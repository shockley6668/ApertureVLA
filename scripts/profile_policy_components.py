"""Profile TurboVLA component latency on one fixed real input."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from aperture_vla.performance import (
    assert_no_other_exclusive_compute_process,
    digest,
    latency_statistics,
    mainline_gpu_lease,
)
from aperture_vla.fast_inference import GraphedForward, enable_batched_views, enable_text_cache
from aperture_vla.policy import ApertureVLAPolicy
from turbovla.evaluation.suite_policy import TurboVLAPolicy


def measure(name, function, warmup, measurements, device):
    elapsed = []
    last_output = None
    with torch.inference_mode():
        for index in range(warmup + measurements):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            last_output = function()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if index >= warmup:
                elapsed.append((time.perf_counter() - started) * 1000)
    result = latency_statistics(elapsed)
    result["latency_all_ms"] = elapsed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dense", "early"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measurements", type=int, default=200)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--cache-text", action="store_true")
    parser.add_argument("--compile-vision", action="store_true")
    parser.add_argument("--batch-views", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true", help="Also time CUDA-graph replay of the full forward")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh profile output path")
    if args.warmup < 1 or args.measurements < 2:
        parser.error("Invalid warmup or measurement count")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    artifact = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if args.sample_index < 0 or args.sample_index >= len(artifact["instructions"]):
        parser.error("Sample index is outside the input artifact")

    stats = Path("third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json")
    processor = Path("assets/models/dinov3-vitb16/preprocessor_config.json")
    if digest(stats) != artifact["stats_sha256"]:
        raise ValueError("Stats artifact mismatch")
    if digest(processor) != artifact["preprocessor_sha256"]:
        raise ValueError("Vision preprocessor artifact mismatch")

    device = torch.device(args.device)
    lease = mainline_gpu_lease(ROOT) if device.type == "cuda" else nullcontext()
    with lease:
        if device.type == "cuda" and not args.allow_shared_gpu:
            assert_no_other_exclusive_compute_process(os.getpid())

        policy_class = TurboVLAPolicy if args.kind == "dense" else ApertureVLAPolicy
        policy = policy_class(
            ckpt_path=args.checkpoint,
            dinov3_path="assets/models/dinov3-vitb16",
            bert_path="assets/models/bert-base-uncased",
            stats_path=stats,
            stats_key=artifact["stats_key"],
            normalize_binary_gripper="false",
            device=device,
            precision=args.precision,
            verbose=False,
        )
        model = policy.model
        model.eval()
        if args.cache_text:
            enable_text_cache(model)
        if args.batch_views:
            enable_batched_views(model)
        if args.compile_vision:
            model.vision_encoder = torch.compile(model.vision_encoder)

        sample = args.sample_index
        instruction = artifact["instructions"][sample]
        samples = {
            "dinov3": artifact["normalized_pixels"][sample : sample + 1].to(
                device=device,
                dtype=policy.model_dtype,
            )
        }
        state = artifact["normalized_states"][sample : sample + 1].to(
            device=device,
            dtype=policy.model_dtype,
        )

        with torch.inference_mode():
            text_tokens, text_mask, text_self_mask = model.text_encoder(
                [instruction], device=device
            )
            if args.kind == "early":
                deep, detail = model.vision_encoder(samples["dinov3"])
                projected = model.vision_projection(deep)
                visual_tokens = model._position_visual_tokens(projected).flatten(1, 2)
                fused_visual, fused_text = model.vision_language_interaction(
                    visual_tokens=visual_tokens,
                    text_tokens=text_tokens,
                    text_key_padding_mask=text_mask,
                    text_self_attention_masks=text_self_mask,
                )
                condition = torch.cat([fused_visual, fused_text], dim=1)
                detail_tokens = model.shallow_detail_projection(detail.flatten(1, 2))
                action_condition = torch.cat([condition, detail_tokens], dim=1)
            else:
                deep = model.vision_encoder(samples["dinov3"])
                projected = model.vision_projection(deep)
                visual_tokens = model._position_visual_tokens(projected).flatten(1, 2)
                fused_visual, fused_text = model.vision_language_interaction(
                    visual_tokens=visual_tokens,
                    text_tokens=text_tokens,
                    text_key_padding_mask=text_mask,
                    text_self_attention_masks=text_self_mask,
                )
                action_condition = torch.cat([fused_visual, fused_text], dim=1)

            action_dtype = model.action_head.decoder.action_queries.weight.dtype
            action_condition = action_condition.to(dtype=action_dtype)
            action_state = state.to(dtype=action_dtype)

        components = {
            "full_forward": lambda: model([instruction], samples, state),
            "text_encoder": lambda: model.text_encoder([instruction], device=device),
            "vision_encoder": lambda: model.vision_encoder(samples["dinov3"]),
            "vision_projection": lambda: model.vision_projection(deep),
            "vision_language_interaction": lambda: model.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_mask,
                text_self_attention_masks=text_self_mask,
            ),
            "action_head": lambda: model.action_head(action_condition, action_state),
        }
        if args.cuda_graph:
            graphed = GraphedForward(model)
            with torch.inference_mode():
                eager_actions = model([instruction], samples, state).float()
                graph_actions = graphed([instruction], samples, state).float()
            graph_parity = float((eager_actions - graph_actions).abs().max())
            components["full_forward_cuda_graph"] = lambda: graphed([instruction], samples, state)
        else:
            graph_parity = None
        if args.kind == "early":
            components["shallow_detail_projection"] = lambda: model.shallow_detail_projection(
                detail.flatten(1, 2)
            )

        results = {
            name: measure(name, function, args.warmup, args.measurements, device)
            for name, function in components.items()
        }

    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "evidence": "fixed_real_input_component_latency_v1",
        "code_commit": code_commit,
        "kind": args.kind,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "input_artifact_sha256": digest(args.inputs),
        "sample_index": args.sample_index,
        "suite": artifact["manifest"][sample]["suite"],
        "task_id": artifact["manifest"][sample]["task_id"],
        "device": str(device),
        "precision": args.precision,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "shared_gpu_provisional": bool(args.allow_shared_gpu),
        "cache_text": bool(args.cache_text),
        "compile_vision": bool(args.compile_vision),
        "batch_views": bool(args.batch_views),
        "cuda_graph": bool(args.cuda_graph),
        "cuda_graph_max_abs_action_diff": graph_parity,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "warmup": args.warmup,
        "measurements": args.measurements,
        "batch_size": 1,
        "components": results,
        "scope": (
            "One fixed real input, batch one, model components only. "
            "Excludes host image preprocessing and environment action conversion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "latency_all_ms"} for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
