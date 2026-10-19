"""Export a per-instruction policy graph for BPU compilation.

Like one CUDA graph per instruction on the GPU, the instruction's cached text features and masks
are baked in as constants, so the graph has only BPU-friendly 4-D/2-D float inputs:
  camera_views      [2, 3, 256, 256]  (third-person, wrist), original preprocessing
  normalized_state  [1, 8]
Output: normalized_action_chunk [1, 12, 7]. Verified against the native PyTorch forward on every
benchmark sample of that instruction (ONNX Runtime with graph optimizations disabled).
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch import nn

from export_cached_text_onnx import CachedTextPolicy, digest
from aperture_vla.fast_inference import enable_batched_views
from aperture_vla.policy import ApertureVLAPolicy


class InstructionPolicy(nn.Module):
    def __init__(self, policy_graph, tokens, padding, self_mask):
        super().__init__()
        self.policy_graph = policy_graph
        self.register_buffer("tokens", tokens)
        self.register_buffer("padding", padding)
        self.register_buffer("self_mask", self_mask)

    def forward(self, camera_views, normalized_state):
        return self.policy_graph(camera_views.unsqueeze(0), normalized_state, self.tokens, self.padding, self.self_mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/performance/libero_balanced80_v1.pt"))
    parser.add_argument("--sample-index", type=int, default=0, help="Benchmark sample whose instruction is baked in")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    torch.backends.mha.set_fastpath_enabled(False)
    artifact = torch.load(args.inputs, map_location="cpu", weights_only=True)
    policy = ApertureVLAPolicy(
        ckpt_path=args.checkpoint, dinov3_path="assets/models/dinov3-vitb16",
        bert_path="assets/models/bert-base-uncased",
        stats_path=ROOT / "third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json",
        stats_key=artifact["stats_key"], normalize_binary_gripper="false",
        device=torch.device("cpu"), precision="fp32", verbose=False,
    )
    model = policy.model.eval()
    enable_batched_views(model)
    instruction = artifact["instructions"][args.sample_index]
    with torch.inference_mode():
        tokens, padding, self_mask = model.text_encoder([instruction], device=torch.device("cpu"))
    graph = InstructionPolicy(CachedTextPolicy(model), tokens.float().clone(), padding.clone(), self_mask.clone()).eval()

    indices = [i for i, text in enumerate(artifact["instructions"]) if text == instruction]
    views = lambda i: artifact["normalized_pixels"][i].float()           # [2, 3, 256, 256]
    state = lambda i: artifact["normalized_states"][i : i + 1].float()  # [1, 8]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(graph, (views(indices[0]), state(indices[0])), str(args.output),
                          input_names=["camera_views", "normalized_state"],
                          output_names=["normalized_action_chunk"], opset_version=args.opset, dynamo=False)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(args.output)))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    errors = []
    for i in indices:
        with torch.inference_mode():
            native = model([instruction], {"dinov3": views(i).unsqueeze(0)}, state(i)).float().numpy()
        out = session.run(None, {"camera_views": views(i).numpy(), "normalized_state": state(i).numpy()})[0]
        errors.append(float(np.abs(out - native).max()))
        np.save(args.output.with_name(f"camera_views_{i}.npy"), views(i).numpy())
        np.save(args.output.with_name(f"normalized_state_{i}.npy"), state(i).numpy())
    receipt = {
        "evidence": "instruction_baked_onnx_export_v1",
        "instruction": instruction,
        "checkpoint_sha256": digest(args.checkpoint),
        "onnx_sha256": digest(args.output),
        "opset": args.opset,
        "inputs": {"camera_views": [2, 3, 256, 256], "normalized_state": [1, 8]},
        "samples_checked": indices,
        "max_abs_onnx_vs_native_forward": max(errors),
        "onnxruntime_graph_optimization": "disabled",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
