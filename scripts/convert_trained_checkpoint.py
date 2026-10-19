"""Attach compression metadata to a checkpoint saved by the upstream trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--compressed-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--state-source",
        choices=("ema", "model"),
        default="ema",
        help="Which trained state to evaluate; EMA is preferred when present",
    )
    args = parser.parse_args()

    template = torch.load(args.compressed_template, map_location="cpu", weights_only=True)
    compression = template.get("compression")
    if not isinstance(compression, dict):
        raise KeyError("Compressed template is missing compression metadata")

    trained = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(trained, dict) or not trained.get("model_config"):
        raise TypeError("Expected a TurboVLA training checkpoint with model_config")
    preferred_key = "ema_model_state_dict" if args.state_source == "ema" else "model_state_dict"
    fallback_key = "model_state_dict" if preferred_key == "ema_model_state_dict" else "ema_model_state_dict"
    state = trained.get(preferred_key)
    used_key = preferred_key
    if not isinstance(state, dict):
        state = trained.get(fallback_key)
        used_key = fallback_key
    if not isinstance(state, dict):
        raise KeyError("Training checkpoint has neither EMA nor model state")

    recipe = trained.get("training_recipe") or {}
    if recipe.get("vision_backbone"):
        compression = {**compression, "vision_backbone": recipe["vision_backbone"]}
    output_checkpoint = {
        "model_config": trained["model_config"],
        "model_state_dict": state,
        "compression": compression,
        "training_protocol": trained.get("training_protocol"),
        "training_recipe": trained.get("training_recipe"),
        "training_diagnostics": trained.get("training_diagnostics"),
        "trained_metadata": {
            "source_checkpoint": str(args.trained_checkpoint),
            "global_step": trained.get("global_step"),
            "loss": trained.get("loss"),
            "ema_decay": trained.get("ema_decay"),
            "state_source": used_key,
            "training_seed": trained.get("args", {}).get("seed"),
            "batch_size": trained.get("args", {}).get("batch_size"),
            "grad_accum_steps": trained.get("args", {}).get("grad_accum_steps"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    report = {
        "evidence": "trained_compressed_checkpoint_conversion",
        "trained_checkpoint": str(args.trained_checkpoint),
        "compressed_template": str(args.compressed_template),
        "output_checkpoint": str(args.output),
        "output_checkpoint_sha256": file_sha256(args.output),
        "state_source": used_key,
        "global_step": trained.get("global_step"),
        "loss": trained.get("loss"),
        "compression": compression,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
