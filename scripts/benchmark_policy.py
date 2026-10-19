"""Measure policy latency on identical real inputs, batch size one.

Kinds used in the paper: ``dense`` (unpooled) and ``early`` (ApertureVLA).
"""
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

import numpy as np
import torch
from aperture_vla.performance import (
    assert_no_other_exclusive_compute_process,
    digest,
    latency_statistics,
    mainline_gpu_lease,
    validate_input_artifact,
)
from aperture_vla.policy import ApertureVLAPolicy
from turbovla.evaluation.suite_policy import TurboVLAPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["dense", "early"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--measurements", type=int, default=400)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh benchmark output path")
    if args.warmup < 1 or args.measurements < 2:
        parser.error("Invalid benchmark counts")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    artifact = torch.load(args.inputs, map_location="cpu", weights_only=True)
    n = validate_input_artifact(artifact)
    stats = Path("third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json")
    processor = Path("assets/models/dinov3-vitb16/preprocessor_config.json")
    if digest(stats) != artifact["stats_sha256"]:
        raise ValueError("Benchmark preprocessing assets differ from the fixed input artifact")
    if digest(processor) != artifact["preprocessor_sha256"]:
        raise ValueError("Encoder preprocessing differs from the fixed input artifact")
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
        forward_pixels, forward_states = artifact["normalized_pixels"], artifact["normalized_states"]

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        results = {}
        with torch.inference_mode():
            for mode in ["model_forward", "host_rgb_to_env_action_chunk"]:
                elapsed = []
                for index in range(args.warmup + args.measurements):
                    i = index % n
                    if mode == "model_forward":
                        pixels, state = forward_pixels[i : i + 1], forward_states[i : i + 1]
                    else:
                        pixels, state = artifact["raw_pixels"][i : i + 1], artifact["raw_states"][i : i + 1]
                    instruction = [artifact["instructions"][i]]
                    synchronize()
                    start = time.perf_counter()
                    if mode == "model_forward":
                        policy.model(instruction, {"dinov3": pixels.to(device)}, state.to(device))
                    else:
                        policy.predict_action_chunk(
                            {"agentview_image": pixels[0, 0].numpy(), "eye_in_hand_image": pixels[0, 1].numpy()},
                            state[0].numpy(),
                            instruction[0],
                        )
                    synchronize()
                    if index >= args.warmup:
                        elapsed.append((time.perf_counter() - start) * 1000.0)
                results[mode] = latency_statistics(elapsed)
        payload = {
            "kind": args.kind,
            "checkpoint": str(args.checkpoint),
            "code_commit": code_commit,
            "device": str(device),
            "precision": args.precision,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
