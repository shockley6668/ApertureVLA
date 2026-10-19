"""Measure the batch-one latency model T = K * tau + sum(c_i) for each execution variant.

For one fixed real input, reports per forward pass: the number of CUDA kernels launched (K), the
summed GPU kernel time (sum c_i, from the profiler) and the synchronized wall time (T). The
difference T - sum c_i is the host-side launch/dispatch overhead the CUDA graph removes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from aperture_vla.fast_inference import GraphedForward, enable_batched_views, enable_text_cache
from aperture_vla.policy import ApertureVLAPolicy
from turbovla.evaluation.suite_policy import TurboVLAPolicy


def wall_ms(fn, repeats=200, warmup=30):
    times = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def kernel_stats(fn, repeats=20):
    with torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    count = len(kernels) / repeats
    busy = sum(e.device_time for e in kernels) / 1000 / repeats  # us -> ms
    return count, busy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/performance/libero_balanced80_v1.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = torch.load(args.inputs, map_location="cpu", weights_only=True)
    common = dict(dinov3_path="assets/models/dinov3-vitb16", bert_path="assets/models/bert-base-uncased",
                  stats_path=ROOT / "third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json",
                  stats_key=artifact["stats_key"], normalize_binary_gripper="false",
                  device=torch.device("cuda"), precision="bf16", verbose=False)
    rows = {}
    for kind, cls, path in (("dense", TurboVLAPolicy, args.dense), ("early2", ApertureVLAPolicy, args.early)):
        policy = cls(ckpt_path=path, **common)
        model = policy.model.eval()
        instruction = [artifact["instructions"][0]]
        samples = {"dinov3": artifact["normalized_pixels"][:1].to("cuda", dtype=policy.model_dtype)}
        state = artifact["normalized_states"][:1].to("cuda", dtype=policy.model_dtype)
        variants = [("eager", None)]
        variants.append(("+text_cache", lambda m: enable_text_cache(m)))
        variants.append(("+batched_views", lambda m: enable_batched_views(m)))
        for name, change in variants:
            if change:
                change(model)
            fn = lambda: model(instruction, samples, state)
            count, busy = kernel_stats(fn)
            rows[f"{kind}/{name}"] = {"kernels": count, "gpu_busy_ms": busy, "wall_ms": wall_ms(fn)}
        graphed = GraphedForward(model)
        fn = lambda: graphed(instruction, samples, state)
        count, busy = kernel_stats(fn)
        rows[f"{kind}/+cuda_graph"] = {"kernels": count, "gpu_busy_ms": busy, "wall_ms": wall_ms(fn)}
        del policy, model, graphed
        torch.cuda.empty_cache()
    for row in rows.values():
        row["overhead_ms"] = row["wall_ms"] - row["gpu_busy_ms"]
    report = {"evidence": "batch_one_latency_model_v1", "hardware": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "batch_size": 1, "rows": rows,
              "note": "Graph replay is profiled as replayed kernels; overhead_ms = wall - summed kernel time."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in rows.items():
        print(f"{name:24s} K={row['kernels']:6.0f}  sum_c={row['gpu_busy_ms']:.2f} ms  T={row['wall_ms']:.2f} ms  overhead={row['overhead_ms']:.2f} ms")


if __name__ == "__main__":
    main()
