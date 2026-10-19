"""Create a balanced real-input artifact once, without occupying the training GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from aperture_vla.performance import digest,validate_input_artifact
from measure_residual_sensitivity import collect_calibration_batch,normalize_instruction,SafeMixedRLDSDataset,benchmark
from turbovla.evaluation.policy import load_preprocessor_config,build_dinov3_manual_processor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dirs',required=True)
    parser.add_argument('--stats-path',type=Path,required=True)
    parser.add_argument('--stats-key',required=True)
    parser.add_argument('--dinov3-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh artifact output path')
    torch.set_num_threads(1)
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tasks={}
    for suite_name in ['libero_spatial','libero_object','libero_goal','libero_10']:
        suite=benchmark.get_benchmark_dict()[suite_name]()
        for i in range(suite.n_tasks):
            key=normalize_instruction(suite.get_task(i).language)
            if key in tasks:raise ValueError('Duplicate instruction')
            tasks[key]={'suite':suite_name,'task_id':i}
    dataset=SafeMixedRLDSDataset(dataset_dir=None,dataset_dirs=args.dataset_dirs,
        LOCAL_DINOV3_PATH=str(args.dinov3_path),stats_path=str(args.stats_path),stats_key=args.stats_key,
        normalize_binary_gripper='false',rank=0,world_size=1,chunk_size=12,split='train',
        shuffle_buffer=8,shuffle_steps_within_episode=False,step_mix_buffer_size=0,
        seed=args.seed,local_files_only=True,expected_image_size=256)
    collated,manifest=collect_calibration_batch(dataset,80,tasks,args.seed)
    samples,instructions,states,_,_=collated
    pixels=samples['dinov3'].float()
    cfg=load_preprocessor_config(str(args.dinov3_path))
    mean=torch.tensor(cfg.get('image_mean',[0.485,0.456,0.406])).view(1,1,3,1,1)
    std=torch.tensor(cfg.get('image_std',[0.229,0.224,0.225])).view(1,1,3,1,1)
    factor=float(cfg.get('rescale_factor',1/255))
    rgb=((pixels*std+mean)/factor).round().clamp(0,255).to(torch.uint8).permute(0,1,3,4,2).contiguous()
    processor=build_dinov3_manual_processor(str(args.dinov3_path))
    max_error=0.0
    for i in range(len(manifest)):
        rebuilt=processor([rgb[i,0].numpy(),rgb[i,1].numpy()])['pixel_values']
        max_error=max(max_error,float((rebuilt-pixels[i]).abs().max()))
    if max_error>1e-5:raise ValueError('RGB inverse preprocessing did not reproduce native tensors')
    raw_states=states*(dataset.proprio_std+1e-6)+dataset.proprio_mean
    payload={'evidence':'balanced_libero_performance_inputs_v1','normalized_pixels':pixels,
        'raw_rgb':rgb,'normalized_states':states,'raw_states':raw_states,
        'instructions':list(instructions),'manifest':manifest,'seed':args.seed,
        'stats_sha256':digest(args.stats_path),'stats_key':args.stats_key,
        'preprocessor_sha256':digest(args.dinov3_path/'preprocessor_config.json')}
    validate_input_artifact(payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_name(args.output.name+'.partial')
    torch.save(payload,partial)
    partial.replace(args.output)
    report={k:v for k,v in payload.items() if not torch.is_tensor(v) and k!='instructions'}
    report.update({'code_commit':code_commit,'dataset_dirs':args.dataset_dirs,
        'artifact':str(args.output),'artifact_sha256':digest(args.output),
        'rgb_roundtrip_max_absolute_error':max_error,
        'scope':'80 real training-demonstration frames; pre-rotated RGB and matching normalized inputs. No latency or success measurement.'})
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='manifest'}))


if __name__=='__main__':main()
