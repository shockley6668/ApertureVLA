"""ApertureVLA policy loader (DINOv3 + BERT, optional mid-encoder pool)."""
from __future__ import annotations

import atexit
import json
import os
import time

import numpy as np
import torch

from turbovla.evaluation.policy import (
    _checkpoint_state_dict,
    _strip_module_prefix,
    sanitize_pred_chunk,
    state_from_libero_obs,
)
from turbovla.evaluation.suite_policy import TurboVLAPolicy

from .replace import replace_linear_layers
from .shared_basis import SharedBasisConfig


from .early_compression import build_early_compression_turbovla
from .memory_head import EpisodeMemory, attach_memory, has_memory_keys, set_memory
from .fast_inference import (
    CachedTextEncoder,
    ConditionRefreshController,
    GraphedForward,
    LoopMonitor,
    RefreshConfig,
    enable_batched_views,
    enable_text_cache,
    set_head_dropout,
    split_condition_forward,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _latency_summary(values):
    if not values:
        return None
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


class ApertureVLAPolicy(TurboVLAPolicy):
    """Strict-load an ApertureVLA or unpooled 0.2B checkpoint."""

    Inference options come from environment variables so the shared LIBERO rollout
    can run every ablation unchanged: CORL_CACHE_TEXT, CORL_BATCH_VIEWS,
    CORL_CUDA_GRAPH, CORL_COMPILE_VISION and CORL_REFRESH_* (see RefreshConfig).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_options = {
            "cache_text": _env_flag("CORL_CACHE_TEXT"),
            "batch_views": _env_flag("CORL_BATCH_VIEWS"),
            "cuda_graph": _env_flag("CORL_CUDA_GRAPH"),
            "compile_vision": _env_flag("CORL_COMPILE_VISION"),
        }
        if self.inference_options["cache_text"]:
            enable_text_cache(self.model)
        if self.inference_options["batch_views"]:
            enable_batched_views(self.model)
        if self.inference_options["compile_vision"]:
            self.model.vision_encoder = torch.compile(self.model.vision_encoder)
        self.refresh = ConditionRefreshController(
            RefreshConfig(
                mode=os.environ.get("CORL_REFRESH_MODE", "always"),
                every_k=int(os.environ.get("CORL_REFRESH_EVERY_K", "2")),
                max_reuse=int(os.environ.get("CORL_REFRESH_MAX_REUSE", "2")),
                position_threshold=float(os.environ.get("CORL_REFRESH_POS", "0.03")),
                rotation_threshold=float(os.environ.get("CORL_REFRESH_ROT", "0.15")),
            )
        )
        # "all": reuse the whole action-head condition (slow path only).
        # "deep": Fast-Slow -- reuse deep semantic tokens, recompute shallow detail every chunk.
        self.refresh_scope = os.environ.get("CORL_REFRESH_SCOPE", "all")
        if self.refresh_scope not in {"all", "deep"}:
            raise ValueError(f"Unknown CORL_REFRESH_SCOPE {self.refresh_scope!r}")
        if self.refresh_scope == "deep" and not hasattr(self.model, "encode_detail"):
            raise ValueError("CORL_REFRESH_SCOPE=deep needs the early-compression detail branch")
        self._encode, self._decode = split_condition_forward(self.model)
        self._action_dtype = self.model.action_head.decoder.action_queries.weight.dtype
        # Persistent per-instruction condition buffers keep fixed addresses for graph replay.
        self._condition_buffers = {}
        self._has_condition = False
        self._runner = self.model
        self._refresh_runner = self._refresh_step
        self._reuse_runner = self._reuse_step
        if self.inference_options["cuda_graph"] and self.device.type == "cuda":
            self._runner = GraphedForward(self.model)
            self._refresh_runner = GraphedForward(self.model, function=self._refresh_step)
            self._reuse_runner = GraphedForward(self.model, function=self._reuse_step)
        self._chunk_latency_ms = {"refresh": [], "reuse": []}
        # Loop-triggered stochastic recovery (off unless CORL_LOOP_DROPOUT is set).
        loop_dropout = os.environ.get("CORL_LOOP_DROPOUT")
        self.loop_dropout = float(loop_dropout) if loop_dropout else None
        self.loop_monitor = LoopMonitor(int(os.environ.get("CORL_LOOP_CLOSES", "4")))
        self.loop_chunks = 0
        if self.loop_dropout is not None and self.refresh.config.mode != "always":
            raise ValueError("Loop recovery is only implemented for CORL_REFRESH_MODE=always")
        # Diagnostic handoff: once the loop monitor fires, another checkpoint (the dense teacher)
        # drives the rest of the episode. Measures whether the teacher can recover from student loops.
        handoff = os.environ.get("CORL_HANDOFF_CKPT")
        self.handoff = None
        self.handoff_chunks = 0
        if handoff:
            if self.refresh.config.mode != "always" or self.loop_dropout is not None:
                raise ValueError("Handoff is only implemented for always-refresh without loop dropout")
            self.handoff = TurboVLAPolicy(
                ckpt_path=handoff, dinov3_path=self.dinov3_path, bert_path=self.bert_path,
                stats_path=self.stats_path, stats_key=self.stats_key, normalize_binary_gripper="false",
                device=self.device, precision=self.precision, verbose=False,
            )
        stats_path = os.environ.get("CORL_INFERENCE_STATS")
        if stats_path:
            atexit.register(self._write_inference_stats, stats_path)

    def reset(self) -> None:
        """Start a new episode: drop the cached condition."""
        self.refresh.reset()
        self.loop_monitor.reset()
        self._has_condition = False
        self._episode_index = getattr(self, "_episode_index", -1) + 1
        self._cycle_rng = np.random.default_rng(1234 + self._episode_index)
        self._grasp_starts = []
        self._last_grip = -1.0
        self._episode_escapes = 0
        self.episode_memory = EpisodeMemory()

    @property
    def uses_memory(self) -> bool:
        return hasattr(self.model.action_head.state_projection, "memory_input")

    def predict_env_action_chunk(self, primary_image, wrist_image, instruction, state_or_obs, execute_steps=None):
        """Limit-cycle escape (CORL_CYCLE_EPS, meters): a deterministic memoryless policy that returns to
        the same observation repeats the same failed grasp. When a chunk would start a grasp within
        ``eps`` of where the previous grasp started, replace it by an escape: open, lift, shift."""
        if not hasattr(self, "episode_memory"):
            self.reset()
        raw = state_from_libero_obs(state_or_obs) if isinstance(state_or_obs, dict) else np.asarray(state_or_obs)
        self._memory_features = self.episode_memory.features(raw) if self.uses_memory else None
        actions = super().predict_env_action_chunk(primary_image, wrist_image, instruction, state_or_obs,
                                                   execute_steps=execute_steps)
        if self.uses_memory:
            self.episode_memory.update(raw, actions[:, 6])
        eps = os.environ.get("CORL_CYCLE_EPS")
        if not eps:
            return actions
        if not hasattr(self, "_grasp_starts"):
            self.reset()
        raw = state_from_libero_obs(state_or_obs) if isinstance(state_or_obs, dict) else np.asarray(state_or_obs)
        signs = np.concatenate([[self._last_grip], np.where(actions[:, 6] > 0, 1.0, -1.0)])
        starts_grasp = bool(np.any((signs[:-1] < 0) & (signs[1:] > 0)))
        if starts_grasp:
            position = np.asarray(raw[:3], dtype=np.float64)
            repeated = bool(self._grasp_starts) and np.linalg.norm(position - self._grasp_starts[-1]) < float(eps)
            self._grasp_starts.append(position)
            if repeated and len(self._grasp_starts) >= 3 and self._episode_escapes < 3:
                self._episode_escapes += 1
                self.cycle_escapes = getattr(self, "cycle_escapes", 0) + 1
                actions = self._escape_chunk(len(actions))
        self._last_grip = 1.0 if actions[-1, 6] > 0 else -1.0
        return actions

    def _escape_chunk(self, length):
        """Open the gripper, lift, then shift laterally in a random direction (env action units)."""
        lift = float(os.environ.get("CORL_CYCLE_LIFT", "0.5"))
        shift = float(os.environ.get("CORL_CYCLE_SHIFT", "0.4"))
        angle = self._cycle_rng.uniform(0.0, 2.0 * np.pi)
        chunk = np.zeros((length, 7), dtype=np.float32)
        chunk[:, 6] = -1.0
        third = max(1, length // 3)
        chunk[:third, 2] = lift
        chunk[third : 2 * third, 0] = shift * np.cos(angle)
        chunk[third : 2 * third, 1] = shift * np.sin(angle)
        return chunk

    def _write_inference_stats(self, path: str) -> None:
        refresh = self._chunk_latency_ms["refresh"]
        reuse = self._chunk_latency_ms["reuse"]
        payload = {
            "inference_options": self.inference_options,
            "refresh_scope": self.refresh_scope,
            "refresh_config": vars(self.refresh.config),
            "refresh_counts": dict(self.refresh.stats),
            "loop_recovery": {"dropout": self.loop_dropout, "closes_threshold": self.loop_monitor.threshold,
                              "triggered_chunks": self.loop_chunks},
            "handoff": {"checkpoint": os.environ.get("CORL_HANDOFF_CKPT"), "chunks": self.handoff_chunks},
            "cycle_escape": {"eps_m": os.environ.get("CORL_CYCLE_EPS"), "escapes": getattr(self, "cycle_escapes", 0),
                             "lift": os.environ.get("CORL_CYCLE_LIFT", "0.5"),
                             "shift": os.environ.get("CORL_CYCLE_SHIFT", "0.4")},
            "chunk_latency": {
                "all": _latency_summary(refresh + reuse),
                "refresh": _latency_summary(refresh),
                "reuse": _latency_summary(reuse),
            },
            "scope": "Synchronized model-call time per action chunk on closed-loop LIBERO inputs; "
            "excludes host image preprocessing.",
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _dummy_samples(self):
        if not hasattr(self, "_dummy_pixels"):
            size = self.model.config.vision.image_size
            self._dummy_pixels = torch.zeros(
                1, self.model.num_views, 3, size, size, device=self.device, dtype=self.model_dtype
            )
        return {"dinov3": self._dummy_pixels}

    def _normalized_state_tensor(self, raw_state):
        norm = (np.asarray(raw_state, dtype=np.float32).reshape(-1) - self.proprio_mean) / (
            self.proprio_std + 1e-6
        )
        return torch.from_numpy(norm).unsqueeze(0).to(self.device, dtype=self.model_dtype)

    def _reuse_step(self, instructions, samples, state):
        condition = self._condition_buffers[tuple(instructions)]
        if self.refresh_scope == "deep":
            detail = self.model.encode_detail(samples).to(dtype=self._action_dtype)
            condition = torch.cat([condition[:, : condition.shape[1] - detail.shape[1]], detail], dim=1)
        return self._decode(condition, state)

    def _refresh_step(self, instructions, samples, state):
        key = tuple(instructions)
        condition = self._encode(instructions, samples)
        buffer = self._condition_buffers.get(key)
        if buffer is None:
            # First sight of this instruction: allocate outside any graph capture.
            buffer = self._condition_buffers[key] = condition.clone()
        elif buffer.shape != condition.shape:
            raise RuntimeError(f"Condition shape changed for {key!r}: {buffer.shape} -> {condition.shape}")
        else:
            buffer.copy_(condition)
        return self._decode(buffer, state)

    def predict_normalized_action_chunk(self, primary_image, wrist_image, instruction, state_or_obs):
        if isinstance(state_or_obs, dict):
            raw_state = state_from_libero_obs(state_or_obs)
        else:
            raw_state = np.asarray(state_or_obs, dtype=np.float32)
        always = self.refresh.config.mode == "always"
        refreshed = always or not self._has_condition or self.refresh.should_refresh(raw_state)
        if refreshed or self.refresh_scope == "deep":
            samples, states = self._build_batch([primary_image], [wrist_image], [raw_state])
            samples, states = self._prepare_model_inputs(samples, states)
        else:
            # Images are not read on a full-condition reuse; keep a dummy for the graph key.
            samples, states = self._dummy_samples(), self._normalized_state_tensor(raw_state)

        cuda = self.device.type == "cuda"
        with torch.inference_mode():
            if cuda:
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            cvae_scale = os.environ.get("CORL_CVAE_SAMPLE")
            if cvae_scale and hasattr(self.model, "cvae") and self.loop_monitor.triggered:
                # Suspected regrasp loop: decode with a fresh latent sample instead of z = 0.
                from .cvae_head import set_latent

                generator = getattr(self, "_cycle_rng", None) or np.random.default_rng(0)
                z = torch.from_numpy(generator.standard_normal((1, self.model.cvae.latent_dim)).astype(np.float32))
                set_latent(self.model, float(cvae_scale) * z.to(self.device))
                try:
                    pred = self.model([instruction], samples, states)
                finally:
                    set_latent(self.model, None)
                self.loop_chunks += 1
            elif self.handoff is not None and self.loop_monitor.triggered:
                pred = self.handoff.model([instruction], samples, states)
                self.handoff_chunks += 1
            elif always and self.loop_dropout is not None and self.loop_monitor.triggered:
                # Suspected regrasp loop: sample from the head instead of repeating its mode.
                set_head_dropout(self.model.action_head, self.loop_dropout)
                try:
                    pred = self.model([instruction], samples, states)
                finally:
                    set_head_dropout(self.model.action_head, None)
                self.loop_chunks += 1
            elif always and getattr(self, "_memory_features", None) is not None:
                memory = torch.from_numpy(self._memory_features).unsqueeze(0).to(self.device)
                set_memory(self.model.action_head, memory)
                try:
                    pred = self.model([instruction], samples, states)
                finally:
                    set_memory(self.model.action_head, None)
            elif always:
                pred = self._runner([instruction], samples, states)
            else:
                if refreshed:
                    if (instruction,) not in self._condition_buffers:
                        self._refresh_step([instruction], samples, states)
                    pred = self._refresh_runner([instruction], samples, states)
                    self._has_condition = True
                else:
                    pred = self._reuse_runner([instruction], samples, states)
            if cuda:
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - started) * 1000

        self._chunk_latency_ms["refresh" if refreshed else "reuse"].append(elapsed)
        self.refresh.record(refreshed, raw_state)
        chunk = sanitize_pred_chunk(pred.detach().float().cpu().numpy()[0])
        self.refresh.note_gripper(chunk[:, 6])
        self.loop_monitor.update(chunk[:, 6])
        return chunk

    def _load_checkpoint(self) -> None:
        compression = self._checkpoint.get("compression") if isinstance(self._checkpoint, dict) else None
        if not isinstance(compression, dict):
            # Plain dense checkpoint (e.g. the official release): load as-is so the same
            # execution path (text cache, batched views, CUDA graph) applies to it.
            super()._load_checkpoint()
            return
        method = compression.get("method")
        load_summary = {"method": method}
        if method == "shared_basis_plus_residual":
            shared_rank = int(compression["shared_rank"])
            residual_rank = int(compression["residual_rank"])
            config = SharedBasisConfig(shared_rank=shared_rank, residual_rank=residual_rank)
            rank_map = compression.get("rank_map")
            for pattern in compression["patterns"]:
                replace_linear_layers(self.model, [pattern], config, rank_map=rank_map, initialize=False)
            load_summary.update(
                shared_rank=shared_rank,
                residual_rank=residual_rank,
                rank_map="yes" if rank_map else "no",
            )
        elif method == "early_dinov3_token_pooling":
            if compression.get("vision_backbone"):
                # Image preprocessing stays the same; only the backbone weights/architecture change.
                self.model.config.vision.model_name_or_path = compression["vision_backbone"]
                load_summary["vision_backbone"] = compression["vision_backbone"]
            self.model = build_early_compression_turbovla(
                self.model.config,
                split_layer=int(compression["split_layer"]),
                merge_factor=int(compression["merge_factor"]),
                shallow_detail_to_action=bool(compression["shallow_detail_to_action"]),
            )
            load_summary.update(
                split_layer=int(compression["split_layer"]),
                merge_factor=int(compression["merge_factor"]),
                shallow_detail_to_action=bool(compression["shallow_detail_to_action"]),
            )
        else:
            raise ValueError(f"Unsupported compression method: {method!r}")

        source_state = _strip_module_prefix(_checkpoint_state_dict(self._checkpoint))
        if "shallow_detail_projection.gate_scale" in source_state:
            self.model.shallow_detail_projection.set_gate_scale(
                float(source_state["shallow_detail_projection.gate_scale"]))
            load_summary["detail_gate_scale"] = float(source_state["shallow_detail_projection.gate_scale"])
        if any(name.startswith("cvae.") for name in source_state):
            from .cvae_head import attach_cvae

            attach_cvae(self.model, int(source_state["cvae.to_query.weight"].shape[1]))
            load_summary["cvae_latent"] = int(source_state["cvae.to_query.weight"].shape[1])
        if has_memory_keys(source_state):
            attach_memory(self.model.action_head)
            load_summary["episode_memory"] = True
        self.model.load_state_dict(source_state, strict=True)
        if self.verbose:
            print(
                "[ApertureVLAPolicy] strict compressed load: "
                f"{self.ckpt_path} ({len(source_state)} tensors; {load_summary})",
                flush=True,
            )
        del self._checkpoint
