"""Collect real per-step inputs of the instruction-baked BPU graph from float closed-loop rollouts.

The BPU graph has the instruction of one LIBERO task baked in, so calibration and evaluation inputs
must come from that task. Episodes start from seeded random placements (not the fixed evaluation
states); the float policy acts, and every chunk's model inputs and float output are saved:
  <output>/<split>/camera_views/<n>.npy      [2, 3, 256, 256] float32
  <output>/<split>/normalized_state/<n>.npy  [1, 8] float32
  <output>/<split>/float_action/<n>.npy      [1, 12, 7] float32
Even episodes go to ``calibration``, odd episodes to ``evaluation``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from aperture_vla.fast_inference import enable_batched_views
from aperture_vla.policy import ApertureVLAPolicy
from aperture_vla.rl.libero_workers import LiberoWorkers

STATS = ROOT / "third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", type=int, default=7, help="LIBERO-10 task whose instruction is baked in")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.update(CORL_CACHE_TEXT="0", CORL_BATCH_VIEWS="1", CORL_CUDA_GRAPH="0", CORL_REFRESH_MODE="always")
    torch.backends.mha.set_fastpath_enabled(False)
    policy = ApertureVLAPolicy(
        ckpt_path=args.checkpoint, dinov3_path="assets/models/dinov3-vitb16", bert_path="assets/models/bert-base-uncased",
        stats_path=STATS, stats_key="libero_all4_no_noops", normalize_binary_gripper="false",
        device=torch.device("cuda"), precision="fp32", verbose=False)
    model = policy.model.eval()
    enable_batched_views(model)

    workers = LiberoWorkers(args.episodes)
    slots = list(range(args.episodes))
    inits = [workers.call([s], "sample_init", [(args.task, args.seed + s)])[0] for s in slots]
    current = dict(zip(slots, workers.call(slots, "reset", [(args.task, init) for init in inits])))
    counts, results = {"calibration": 0, "evaluation": 0}, {}
    try:
        while current:
            active = list(current)
            env_actions = []
            for slot in active:
                obs = current[slot]
                samples, states = policy._build_batch([obs["primary"]], [obs["wrist"]], [obs["state"]])
                samples, states = policy._prepare_model_inputs(samples, states)
                pixels = samples["dinov3"] if isinstance(samples, dict) else samples
                with torch.inference_mode():
                    pred = model([obs["instruction"]], samples, states).float()
                split = "calibration" if slot % 2 == 0 else "evaluation"
                index = counts[split]
                counts[split] += 1
                for name, value in (("camera_views", pixels.reshape(2, 3, 256, 256)),
                                    ("normalized_state", states.reshape(1, 8)), ("float_action", pred.reshape(1, 12, 7))):
                    folder = args.output / split / name
                    folder.mkdir(parents=True, exist_ok=True)
                    np.save(folder / f"{index}.npy", value.float().cpu().numpy().astype(np.float32))
                env_actions.append(np.stack([policy._normalized_row_to_env_action(r) for r in pred[0].cpu().numpy()]))
            for slot, result in zip(active, workers.call(active, "step", env_actions)):
                if result["finished"]:
                    results[slot] = {"success": result["success"], "steps": result["steps"]}
                    del current[slot]
                else:
                    current[slot] = result
    finally:
        workers.close()
    receipt = {"evidence": "bpu_task_frames_v1", "checkpoint": args.checkpoint, "task": args.task,
               "instruction": obs["instruction"], "seed": args.seed, "frames": counts,
               "episodes": {str(k): v for k, v in sorted(results.items())}}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
