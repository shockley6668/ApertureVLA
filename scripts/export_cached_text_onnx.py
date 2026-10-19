"""Export the early-compression policy for edge compilation, with the instruction encoder cached.

On the robot the instruction is fixed per episode, so BERT runs once on the CPU and its
outputs are reused (the same optimization as ``CORL_CACHE_TEXT`` on the GPU). The exported
graph is the per-step part only: DINOv3 (both views batched), projection, vision-language
interaction, shallow-detail branch and action head.

Inputs (all FP32 or bool, static shapes):
  pixel_values                [1, 2, 3, 256, 256]  original two-camera preprocessing
  normalized_state            [1, 8]
  text_tokens                 [1, T, 256]          cached text-encoder output
  text_key_padding_mask       [1, T] bool          True = padding
  text_self_attention_mask    [1, T, T] bool
Output: normalized_action_chunk [1, 12, 7]

The script verifies ONNX Runtime against PyTorch on real benchmark inputs for every
instruction in the input artifact and writes a JSON receipt next to the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from aperture_vla.fast_inference import enable_batched_views
from aperture_vla.policy import ApertureVLAPolicy

INPUTS = ("pixel_values", "normalized_state", "text_tokens", "text_key_padding_mask", "text_self_attention_mask")


class CachedTextPolicy(nn.Module):
    """Per-step policy graph with the text-encoder outputs supplied as inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, normalized_state, text_tokens, text_key_padding_mask, text_self_attention_mask):
        model = self.model
        detail_branch = getattr(model, "shallow_detail_to_action", False)
        if detail_branch:
            deep, detail = model.vision_encoder(pixel_values)
            visual = model._position_visual_tokens(model.vision_projection(deep)).flatten(1, 2)
        else:
            visual = model.encode_vision(pixel_values)
        visual, text = model.vision_language_interaction(
            visual_tokens=visual,
            text_tokens=text_tokens,
            text_key_padding_mask=text_key_padding_mask,
            text_self_attention_masks=text_self_attention_mask,
        )
        condition = torch.cat([visual, text], dim=1)
        if detail_branch:
            condition = torch.cat([condition, model.shallow_detail_projection(detail.flatten(1, 2))], dim=1)
        return model.action_head(condition, normalized_state)


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/performance/libero_balanced80_v1.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(7)
    # The fused nn.MultiheadAttention fast path has no ONNX export; the math path is equivalent.
    torch.backends.mha.set_fastpath_enabled(False)
    artifact = torch.load(args.inputs, map_location="cpu", weights_only=True)
    policy = ApertureVLAPolicy(
        ckpt_path=args.checkpoint,
        dinov3_path="assets/models/dinov3-vitb16",
        bert_path="assets/models/bert-base-uncased",
        stats_path=ROOT / "third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json",
        stats_key=artifact["stats_key"],
        normalize_binary_gripper="false",
        device=torch.device("cpu"),
        precision="fp32",
        verbose=False,
    )
    model = policy.model.eval()
    enable_batched_views(model)
    wrapper = CachedTextPolicy(model).eval()

    def example(index):
        pixels = artifact["normalized_pixels"][index : index + 1].float()
        state = artifact["normalized_states"][index : index + 1].float()
        with torch.inference_mode():
            tokens, padding, self_mask = model.text_encoder([artifact["instructions"][index]], device=torch.device("cpu"))
        return pixels, state, tokens.float().clone(), padding.clone(), self_mask.clone()

    first = example(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, first, str(args.output), input_names=list(INPUTS),
            output_names=["normalized_action_chunk"], opset_version=args.opset, dynamo=False,
        )

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    op_counts = {}
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    # ORT's default CPU graph optimizations change DINOv3 activations by up to ~25% relative
    # (see scripts/debug_onnx_stages.py); with them disabled the graph matches PyTorch to ~1e-6.
    # The vendor compiler parses the ONNX graph itself, so verification uses the unoptimized graph.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])

    checks = []
    seen = set()
    for index, instruction in enumerate(artifact["instructions"]):
        if instruction in seen:
            continue
        seen.add(instruction)
        feeds = example(index)
        with torch.inference_mode():
            reference = wrapper(*feeds).numpy()
            native = model([instruction], {"dinov3": feeds[0]}, feeds[1]).float().numpy()
        onnx_out = session.run(None, {name: value.numpy() for name, value in zip(INPUTS, feeds)})[0]
        checks.append({
            "instruction": instruction,
            "text_length": int(feeds[2].shape[1]),
            "max_abs_onnx_vs_wrapper": float(np.abs(onnx_out - reference).max()),
            "max_abs_wrapper_vs_native_forward": float(np.abs(reference - native).max()),
        })
    lengths = sorted({c["text_length"] for c in checks})
    receipt = {
        "evidence": "cached_text_onnx_export_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "onnx": str(args.output),
        "onnx_sha256": digest(args.output),
        "opset": args.opset,
        "onnxruntime_graph_optimization": "disabled",
        "inputs": {name: list(value.shape) for name, value in zip(INPUTS, first)},
        "text_lengths_seen": lengths,
        "instructions_checked": len(checks),
        "max_abs_onnx_vs_wrapper": max(c["max_abs_onnx_vs_wrapper"] for c in checks),
        "max_abs_wrapper_vs_native_forward": max(c["max_abs_wrapper_vs_native_forward"] for c in checks),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "checks": checks,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("checks", "op_counts")}, indent=2))
    print("ops:", receipt["op_counts"])


if __name__ == "__main__":
    main()
