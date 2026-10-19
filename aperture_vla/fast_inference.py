"""Execution-aware inference for TurboVLA-style policies.

Every optimization here is opt-in so each one can be ablated:

- ``CachedTextEncoder``: instructions are fixed within an episode, so BERT runs once.
- ``enable_batched_views``: encode both camera views in one DINOv3 call instead of a loop.
- ``GraphedForward``: replay the whole condition + action forward from a CUDA graph to
  remove the per-kernel launch overhead that dominates batch-one latency.
- ``ConditionRefreshController``: reuse the vision-language condition across action
  chunks and only refresh it when the robot state says the scene is changing.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import os

import numpy as np
import torch
from torch import nn


class CachedTextEncoder(nn.Module):
    """Cache deterministic text tokens for repeated instructions."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cache = {}

    def forward(self, instructions, device):
        key = (tuple(instructions), str(device))
        if key not in self.cache:
            self.cache[key] = self.encoder(instructions, device=device)
        return self.cache[key]


def enable_text_cache(model) -> None:
    if not isinstance(model.text_encoder, CachedTextEncoder):
        model.text_encoder = CachedTextEncoder(model.text_encoder)


def enable_batched_views(model) -> None:
    """Fold the view axis into the batch axis for one DINOv3 call per step."""
    encoder = model.vision_encoder
    encoder = getattr(encoder, "_orig_mod", encoder)
    if hasattr(encoder, "early_split_layer"):
        encoder.batch_views = True
    else:
        encoder.config.encode_views_separately = False


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    pixels: torch.Tensor
    state: torch.Tensor
    output: torch.Tensor


class GraphedForward:
    """Replay ``function(instructions, samples, state)`` (default: the full model) from CUDA graphs.

    One graph is captured per (instruction, input shape). Text tokens must come from
    ``CachedTextEncoder`` because tokenization runs on the host and cannot be captured.
    """

    def __init__(self, model, function=None, max_graphs: int | None = None, warmup: int = 3) -> None:
        enable_text_cache(model)
        self.model = model
        self.function = function or (lambda instructions, samples, state: model(instructions, samples, state))
        # One episode uses one instruction, so a small LRU suffices on the robot; benchmarks that
        # cycle through many instructions must raise CORL_MAX_GRAPHS or they measure re-capture.
        self.max_graphs = int(max_graphs if max_graphs is not None else os.environ.get("CORL_MAX_GRAPHS", "4"))
        self.warmup = int(warmup)
        self.graphs: OrderedDict[tuple, _GraphEntry] = OrderedDict()

    def _capture(self, instructions, pixels, state) -> _GraphEntry:
        function = self.function
        static_pixels = pixels.clone()
        static_state = state.clone()
        # Fill the text cache and any lazily built buffers outside the graph.
        stream = torch.cuda.Stream(device=pixels.device)
        stream.wait_stream(torch.cuda.current_stream(pixels.device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                function(instructions, {"dinov3": static_pixels}, static_state)
        torch.cuda.current_stream(pixels.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function(instructions, {"dinov3": static_pixels}, static_state)
        return _GraphEntry(graph, static_pixels, static_state, output)

    def __call__(self, instructions, samples, state):
        pixels = self.model._normalize_samples(samples)
        key = (tuple(instructions), tuple(pixels.shape), pixels.dtype, tuple(state.shape), state.dtype)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(instructions, pixels, state)
            self.graphs[key] = entry
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        entry.pixels.copy_(pixels)
        entry.state.copy_(state)
        entry.graph.replay()
        return entry.output.clone()


def split_condition_forward(model):
    """Return (encode, decode) callables that split the forward at the condition tokens."""
    action_dtype = model.action_head.decoder.action_queries.weight.dtype

    def encode(instructions, samples):
        condition = model.encode_condition(instructions, samples)
        if isinstance(condition, tuple):
            condition, detail = condition
            if detail is not None:
                condition = torch.cat([condition, detail], dim=1)
        return condition.to(dtype=action_dtype)

    def decode(condition, state):
        return model.action_head(condition, state.to(dtype=action_dtype))

    return encode, decode


@dataclass
class RefreshConfig:
    """When to recompute the vision-language condition.

    ``mode``: ``always`` (baseline), ``every_k`` or ``adaptive``. In adaptive mode the
    condition is refreshed after ``max_reuse`` reused chunks, on a gripper command flip,
    or when the end-effector moved more than ``position_threshold`` (meters) or rotated
    more than ``rotation_threshold`` (radians, axis-angle norm difference) since the
    last refresh.
    """

    mode: str = "always"
    every_k: int = 2
    max_reuse: int = 2
    position_threshold: float = 0.03
    rotation_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.mode not in {"always", "every_k", "adaptive"}:
            raise ValueError(f"Unknown refresh mode {self.mode!r}")
        if self.every_k < 1 or self.max_reuse < 0:
            raise ValueError("every_k must be >= 1 and max_reuse >= 0")


@dataclass
class ConditionRefreshController:
    """Decide per action chunk whether the cached condition is still usable."""

    config: RefreshConfig
    _anchor_state: np.ndarray | None = None
    _last_gripper: float | None = None
    _reused: int = 0
    stats: dict = field(default_factory=lambda: {"refresh": 0, "reuse": 0})

    def reset(self) -> None:
        self._anchor_state = None
        self._last_gripper = None
        self._reused = 0

    def should_refresh(self, raw_state: np.ndarray) -> bool:
        """``raw_state`` is the unnormalized 8-D LIBERO state: pos(3), axis-angle(3), gripper(2)."""
        config = self.config
        if config.mode == "always" or self._anchor_state is None:
            return True
        if config.mode == "every_k":
            return self._reused + 1 >= config.every_k
        if self._reused >= config.max_reuse:
            return True
        delta = raw_state - self._anchor_state
        if float(np.linalg.norm(delta[:3])) > config.position_threshold:
            return True
        if float(np.linalg.norm(delta[3:6])) > config.rotation_threshold:
            return True
        return False

    def note_gripper(self, gripper_commands) -> None:
        """Force a refresh at the next chunk when the commanded gripper changes sign."""
        signs = np.sign(np.asarray(gripper_commands, dtype=np.float32).reshape(-1))
        if self._last_gripper is not None:
            signs = np.concatenate([[self._last_gripper], signs])
        if np.any(signs[1:] != signs[:-1]):
            self._reused = max(self.config.max_reuse, self.config.every_k)
        self._last_gripper = float(signs[-1])

    def record(self, refreshed: bool, raw_state: np.ndarray) -> None:
        if refreshed:
            self._anchor_state = np.asarray(raw_state, dtype=np.float32).copy()
            self._reused = 0
            self.stats["refresh"] += 1
        else:
            self._reused += 1
            self.stats["reuse"] += 1


class LoopMonitor:
    """Count grasp attempts from the gripper command and flag a suspected regrasp loop.

    LIBERO gripper commands are -1 (open) and +1 (close). Every open->close transition across
    consecutive chunks is one grasp attempt. ``triggered`` becomes true at the ``threshold``-th
    attempt and stays true until ``reset`` (a new episode). A bare count is a weak detector --
    successful episodes also regrasp -- so callers must make the triggered behaviour mild.
    """

    def __init__(self, threshold: int = 4) -> None:
        if threshold < 1:
            raise ValueError("threshold must be positive")
        self.threshold = int(threshold)
        self.reset()

    def reset(self) -> None:
        self.closes = 0
        self._last = None

    @property
    def triggered(self) -> bool:
        return self.closes >= self.threshold

    def update(self, gripper_norm) -> None:
        commands = np.where(np.asarray(gripper_norm, dtype=np.float32).reshape(-1) >= 0, 1.0, -1.0)
        if self._last is not None:
            commands = np.concatenate([[self._last], commands])
        self.closes += int(np.sum((commands[:-1] < 0) & (commands[1:] > 0)))
        self._last = float(commands[-1])


def set_head_dropout(action_head, probability: float | None) -> None:
    """Enable Monte-Carlo dropout in the action head (``None`` restores eval behaviour)."""
    if probability is None:
        action_head.eval()
        return
    action_head.train()
    for module in action_head.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(probability)
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = float(probability)
