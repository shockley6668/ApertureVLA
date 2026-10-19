"""Create an early-compression TurboVLA checkpoint from the dense control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-layer", type=int, default=4)
    parser.add_argument("--merge-factor", type=int, default=2)
    parser.add_argument("--no-shallow-detail", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.split_layer < 1 or args.merge_factor < 2:
        raise SystemExit("split-layer must be positive and merge-factor at least 2")

    checkpoint = torch.load(args.dense_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("model_config"):
        raise TypeError("Expected a TurboVLA checkpoint containing model_config")
    if checkpoint.get("compression"):
        raise TypeError("The source must be an uncompressed dense TurboVLA checkpoint")
    state_source = "ema_model_state_dict" if isinstance(
        checkpoint.get("ema_model_state_dict"), dict
    ) else "model_state_dict"
    source = {
        name.removeprefix("module."): value
        for name, value in checkpoint[state_source].items()
    }

    from aperture_vla.early_compression import build_early_compression_turbovla
    from turbovla.models.configuration import TurboVLAConfig

    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    shallow_detail = not args.no_shallow_detail
    torch.manual_seed(args.seed)
    model = build_early_compression_turbovla(
        config,
        split_layer=args.split_layer,
        merge_factor=args.merge_factor,
        shallow_detail_to_action=shallow_detail,
    )
    missing, unexpected = model.load_state_dict(source, strict=False)
    if unexpected:
        raise RuntimeError(f"Dense checkpoint has unexpected tensors: {unexpected[:10]}")
    expected_missing = []
    if shallow_detail:
        expected_missing = [
            name for name in missing if name.startswith("shallow_detail_projection.")
        ]
    if missing != expected_missing:
        raise RuntimeError(f"Unexpected missing tensors after migration: {missing[:20]}")

    compression = {
        "method": "early_dinov3_token_pooling",
        "split_layer": int(args.split_layer),
        "merge_factor": int(args.merge_factor),
        "shallow_detail_to_action": bool(shallow_detail),
        "deep_tokens_per_view": 64,
        "shallow_tokens_per_view": 256,
        "source_state": state_source,
        "seed": int(args.seed),
    }
    payload = {
        "model_config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "compression": compression,
        "migration": {
            "source_checkpoint": str(args.dense_checkpoint),
            "source_sha256": file_sha256(args.dense_checkpoint),
            "source_global_step": checkpoint.get("global_step"),
        },
    }
    reload_model = build_early_compression_turbovla(
        config,
        split_layer=args.split_layer,
        merge_factor=args.merge_factor,
        shallow_detail_to_action=shallow_detail,
    )
    reload_model.load_state_dict(payload["model_state_dict"], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    report = {
        "evidence": "dense_checkpoint_migrated_to_early_compression",
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "source_checkpoint": str(args.dense_checkpoint),
        "source_sha256": payload["migration"]["source_sha256"],
        "source_state": state_source,
        "split_layer": args.split_layer,
        "merge_factor": args.merge_factor,
        "shallow_detail_to_action": shallow_detail,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "candidate_state_tensors": len(payload["model_state_dict"]),
        "added_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("shallow_detail_projection.")
        ),
        "strict_candidate_reload": True,
        "strict_candidate_reload_scope": "in_memory_state_dict_before_atomic_write",
        "scope": "Checkpoint migration only; no training or closed-loop evaluation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
