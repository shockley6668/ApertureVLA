"""Train a compressed TurboVLA checkpoint with the official trainer.

This wrapper keeps the upstream optimizer, data pipeline, and checkpoint format.
It only replaces model construction so the compressed state is strict-loaded
before training starts. By default, only compressed BERT FFNs are trainable;
--full-text-finetuning also releases the retained BERT weights for recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aperture_vla.replace import replace_linear_layers
from aperture_vla.shared_basis import SharedBasisConfig
from aperture_vla.early_compression import build_early_compression_turbovla
from aperture_vla.training import (train_text_compression, train_full_text,
                                  install_retained_text_lr, use_safe_mixed_dataset)
from aperture_vla.checkpointing import install_checkpoint_writer


def freeze_bert_except_compressed_modules(model) -> None:
    train_text_compression(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressed-checkpoint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-unfreeze-text-compression",
        action="store_true",
        help="Keep every BERT parameter frozen, including compressed FFN modules",
    )
    mode.add_argument("--full-text-finetuning", action="store_true")
    parser.add_argument("--retained-text-lr", type=float, default=1e-5)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--teacher-micro-batch", type=int, default=1)
    parser.add_argument("--teacher-attention-stability", choices=("native", "per_sample"), default="native")
    parser.add_argument(
        "--stale-deltas",
        type=str,
        default="",
        help="Comma-separated frame offsets for staleness-robust Fast-Slow training, e.g. 0,12,24,36",
    )
    parser.add_argument("--onpolicy-dir", type=Path, help="Teacher-labelled student rollouts to mix in")
    parser.add_argument("--onpolicy-selection", choices=("all", "hard", "recovered", "success"), default="hard")
    parser.add_argument("--onpolicy-labels", choices=("teacher", "executed"), default="teacher")
    parser.add_argument("--cvae-latent", type=int, default=0, help="Add an ACT-style CVAE latent of this size")
    parser.add_argument("--cvae-beta", type=float, default=1e-3)
    parser.add_argument("--cvae-prior-weight", type=float, default=0.5)
    parser.add_argument("--vision-backbone", type=Path,
                        help="Swap in another pretrained vision backbone (warm-start the rest)")
    parser.add_argument("--detail-gate-scale", type=float, default=0.0,
                        help="Reparameterize the shallow-detail gate with this scale (0 = unchanged)")
    parser.add_argument("--onpolicy-prob", type=float, default=0.25)
    custom_args, remaining_args = parser.parse_known_args()
    if not math.isfinite(custom_args.distillation_weight) or custom_args.distillation_weight < 0:
        parser.error("Distillation weight must be finite and nonnegative")
    if custom_args.teacher_micro_batch < 1:
        parser.error("Teacher micro-batch must be positive")
    import torch
    checkpoint_probe = torch.load(
        custom_args.compressed_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint_probe, dict) or not checkpoint_probe.get("compression"):
        raise TypeError("Expected a compressed checkpoint with compression metadata")
    compression_method = checkpoint_probe["compression"].get("method")
    if compression_method not in {"shared_basis_plus_residual", "early_dinov3_token_pooling"}:
        raise ValueError(f"Unsupported compressed architecture: {compression_method!r}")
    if (
        custom_args.teacher_checkpoint
        and compression_method != "early_dinov3_token_pooling"
        and not custom_args.full_text_finetuning
    ):
        parser.error("Shared-basis action distillation requires --full-text-finetuning")
    del checkpoint_probe
    if custom_args.full_text_finetuning:
        remaining_args.append("--train_text_encoder")

    # The compressed checkpoint is already complete. Disable the upstream
    # partial-preload path so it cannot overwrite the strict-loaded weights.
    remaining_args.extend(
        [
            "--no_require_feature_enhancer_preload",
            "--no_load_text_projection_from_init",
            "--no_require_text_proj_preload",
        ]
    )
    if "--pretrained_init_ckpt" not in remaining_args:
        remaining_args.extend(
            ["--pretrained_init_ckpt", str(custom_args.compressed_checkpoint)]
        )
    if "--text_layout_path" not in remaining_args:
        remaining_args.extend(
            [
                "--text_layout_path",
                str(
                    ROOT
                    / "third_party/TurboVLA/experiments/libero/configs/online_text_layout.json"
                ),
            ]
        )

    sys.argv = [sys.argv[0], *remaining_args]

    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.training import trainer, train_mixed
    use_safe_mixed_dataset(train_mixed)
    stale_recipe = None
    onpolicy_recipe = None
    if custom_args.onpolicy_dir:
        from aperture_vla.onpolicy_data import install_onpolicy_mixing

        onpolicy_recipe = install_onpolicy_mixing(
            train_mixed.MixedSuiteStatsLiberoRLDSDataset,
            custom_args.onpolicy_dir,
            custom_args.onpolicy_selection,
            custom_args.onpolicy_prob,
            labels=custom_args.onpolicy_labels,
        )
        print(f"[onpolicy] {onpolicy_recipe}", flush=True)
    if custom_args.stale_deltas:
        from aperture_vla.staleness import install_stale_sampling

        deltas = [int(value) for value in custom_args.stale_deltas.split(",")]
        stale_recipe = install_stale_sampling(train_mixed.MixedSuiteStatsLiberoRLDSDataset, deltas)
    if custom_args.full_text_finetuning:
        from turbovla.training import pi05
        install_retained_text_lr(pi05, custom_args.retained_text_lr)

    def build_compressed_model_architecture(args):
        checkpoint = torch.load(custom_args.compressed_checkpoint, map_location="cpu")
        if not isinstance(checkpoint, dict) or not checkpoint.get("compression"):
            raise TypeError("Expected a compressed checkpoint with compression metadata")
        if not checkpoint.get("model_config"):
            raise TypeError("Compressed checkpoint is missing model_config")
        compression = checkpoint["compression"]
        model_config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
        model_config.text.model_name_or_path = args.bert_path
        model_config.vision.model_name_or_path = (
            str(custom_args.vision_backbone) if custom_args.vision_backbone else args.dinov3_path)
        model_config.text.local_files_only = not args.allow_hf_download
        model_config.vision.local_files_only = not args.allow_hf_download
        model_config.vision.frozen = args.freeze_backbones
        model_config.text.frozen = args.freeze_text_encoder
        model_config.vision.compute_precision = getattr(
            args, "dinov3_precision", "bf16_autocast"
        )
        state_source = (
            "ema_model_state_dict"
            if isinstance(checkpoint.get("ema_model_state_dict"), dict)
            else "model_state_dict"
        )
        state = {
            name.removeprefix("module."): value
            for name, value in checkpoint[state_source].items()
        }
        if compression_method == "shared_basis_plus_residual":
            from turbovla.models.turbovla import build_turbovla

            model = build_turbovla(model_config)
            config = SharedBasisConfig(
                shared_rank=int(compression["shared_rank"]),
                residual_rank=int(compression["residual_rank"]),
            )
            rank_map = compression.get("rank_map")
            for pattern in compression["patterns"]:
                replace_linear_layers(model, [pattern], config, rank_map=rank_map, initialize=False)
        else:
            model = build_early_compression_turbovla(
                model_config,
                split_layer=int(compression["split_layer"]),
                merge_factor=int(compression["merge_factor"]),
                shallow_detail_to_action=bool(compression["shallow_detail_to_action"]),
            )
        swapped_keys = []
        if custom_args.vision_backbone:
            # The new backbone keeps its own pretrained weights; modules whose shapes depend on the
            # vision width are re-initialized; fusion, text projection and head are warm-started.
            own = model.state_dict()
            kept = {}
            for name, value in state.items():
                if name.startswith("vision_encoder.backbone.") or name not in own or own[name].shape != value.shape:
                    swapped_keys.append(name)
                else:
                    kept[name] = value
            missing, unexpected = model.load_state_dict(kept, strict=False)
            print(f"[vision-backbone] {custom_args.vision_backbone}: kept {len(kept)} tensors, "
                  f"re-initialized {len(missing)} (non-backbone: "
                  f"{sorted({m.split('.')[0] for m in missing if not m.startswith('vision_encoder.backbone.')})})",
                  flush=True)
        else:
            model.load_state_dict(state, strict=True)
        text_trainability = "frozen"
        recipe = {
            "architecture": compression_method,
            "text_trainability": text_trainability,
            "bert_dropout": "eval",
            "initial_checkpoint": str(custom_args.compressed_checkpoint),
            "initial_state_source": state_source,
            "parent_global_step": checkpoint.get(
                "trained_metadata", checkpoint
            ).get("global_step"),
        }
        if custom_args.full_text_finetuning:
            train_full_text(model)
            text_trainability = "full"
            recipe["retained_text_lr"] = custom_args.retained_text_lr
        elif (
            compression_method == "shared_basis_plus_residual"
            and not custom_args.no_unfreeze_text_compression
        ):
            freeze_bert_except_compressed_modules(model)
            text_trainability = "compressed_ffns"
        recipe["text_trainability"] = text_trainability
        if custom_args.vision_backbone:
            recipe["vision_backbone"] = str(custom_args.vision_backbone)
        if onpolicy_recipe is not None:
            recipe["onpolicy_distillation"] = onpolicy_recipe
        if stale_recipe is not None:
            if compression_method != "early_dinov3_token_pooling":
                raise ValueError("Staleness training needs the early-compression detail branch")
            from aperture_vla.staleness import time_major_hook

            # Registered before the teacher hook so the teacher sees the current frame.
            model.register_forward_pre_hook(time_major_hook)
            recipe["staleness_training"] = stale_recipe
        if custom_args.detail_gate_scale > 0:
            if compression_method != "early_dinov3_token_pooling" or model.shallow_detail_projection is None:
                raise ValueError("--detail-gate-scale needs the early-compression detail branch")
            before = float(model.shallow_detail_projection.effective_gate())
            model.shallow_detail_projection.set_gate_scale(custom_args.detail_gate_scale)
            recipe["detail_gate"] = {"scale": custom_args.detail_gate_scale, "initial_effective_gate": before}
        if custom_args.cvae_latent > 0:
            if compression_method != "early_dinov3_token_pooling":
                raise ValueError("The CVAE latent is implemented for the early-compression model")
            from aperture_vla.cvae_head import attach_cvae, make_cvae_loss

            attach_cvae(model, custom_args.cvae_latent)
            # Installed before the teacher wrapper so KD wraps the CVAE loss, not the reverse.
            trainer.masked_l1_loss = make_cvae_loss(
                model, trainer.masked_l1_loss, beta=custom_args.cvae_beta,
                prior_weight=custom_args.cvae_prior_weight, log_every=args.log_freq,
            )
            recipe["cvae"] = {"latent_dim": custom_args.cvae_latent, "beta": custom_args.cvae_beta,
                              "prior_weight": custom_args.cvae_prior_weight,
                              "injection": "additive on action queries, bias-free projection, gate init 1"}
        distiller = None
        if custom_args.teacher_checkpoint:
            from aperture_vla.distillation import OnlineActionDistillation
            from aperture_vla.teacher_attention import stabilize_teacher_batching
            from turbovla.evaluation.suite_policy import TurboVLAPolicy

            if args.precision != "bf16_amp" or args.resume_mode != "none":
                raise ValueError("The recovery KD comparison requires bf16_amp and fresh optimizer")
            teacher_path = custom_args.teacher_checkpoint
            with teacher_path.open("rb") as handle:
                teacher_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True, mmap=True)
            if teacher_checkpoint.get("compression"):
                raise ValueError("This comparison requires the independent dense teacher")
            if not isinstance(teacher_checkpoint.get("ema_model_state_dict"), dict):
                raise ValueError("The dense teacher must contain its evaluated EMA state")
            del teacher_checkpoint
            # Loading another backbone consumes random initialization draws.
            # Preserve the student's RNG trajectory and subsequent data seed.
            with torch.random.fork_rng(devices=[]):
                teacher_policy = TurboVLAPolicy(
                    ckpt_path=teacher_path, dinov3_path=args.dinov3_path,
                    bert_path=args.bert_path, stats_path=args.stats_path,
                    stats_key=args.stats_key, normalize_binary_gripper="false",
                    device="cpu", precision="bf16", verbose=False,
                )
            stabilized_attention = (stabilize_teacher_batching(teacher_policy.model)
                                    if custom_args.teacher_attention_stability == "per_sample" else [])
            distiller = OnlineActionDistillation(
                teacher_policy.model, trainer.masked_l1_loss,
                weight=custom_args.distillation_weight,
                micro_batch=custom_args.teacher_micro_batch,
                log_every=args.log_freq,
            )
            distiller.attach(model)
            trainer.masked_l1_loss = distiller.loss
            recipe["action_distillation"] = {
                "method": "supervised_masked_l1_plus_teacher_masked_mse",
                "weight": custom_args.distillation_weight,
                "teacher_checkpoint": str(teacher_path), "teacher_sha256": teacher_sha,
                "teacher_state_source": "ema_model_state_dict",
                "teacher_precision": "bf16", "teacher_dropout": "eval",
                "teacher_micro_batch": custom_args.teacher_micro_batch,
                "teacher_attention_stability": custom_args.teacher_attention_stability,
                "teacher_attention_modules": stabilized_attention,
                "teacher_parameters": sum(p.numel() for p in teacher_policy.model.parameters()),
                "teacher_in_student_checkpoint": False,
            }
        install_checkpoint_writer(trainer, model, compression, training_recipe=recipe,
                                  training_diagnostics=distiller.diagnostics if distiller else None)
        return model

    trainer.build_model_architecture = build_compressed_model_architecture
    train_mixed.main()


if __name__ == "__main__":
    main()
