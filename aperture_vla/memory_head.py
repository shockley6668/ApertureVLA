"""Episode memory input for the action head.

A memoryless policy a = pi(o) repeats a failed grasp when the scene returns to the same observation.
The memory input m_t summarises the episode's grasp history from signals the policy already has:

    m_t = [ min(k_t, 8) / 8,                       k_t: grasp onsets so far
            min(c_t, 20) / 20,                     c_t: chunks since the last onset
            clip(10 * (p_t - p_last), -1, 1),      end-effector offset (m) from the last onset
            1[k_t > 0] ]

It enters as an additive offset on the state tokens, produced by an MLP whose last layer is zero, so a
freshly attached memory leaves the policy exactly unchanged.
"""
from __future__ import annotations

import types

import numpy as np
import torch
from torch import nn

MEMORY_DIM = 6


class EpisodeMemory:
    """Tracks grasp onsets over one episode from executed gripper commands (> 0 closes)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.onsets = 0
        self.chunks_since = 0
        self.last_position = None
        self.last_grip = -1.0

    def features(self, raw_state) -> np.ndarray:
        position = np.asarray(raw_state[:3], dtype=np.float64)
        offset = np.zeros(3) if self.last_position is None else np.clip(10.0 * (position - self.last_position), -1, 1)
        return np.asarray([min(self.onsets, 8) / 8.0, min(self.chunks_since, 20) / 20.0, *offset,
                           float(self.onsets > 0)], dtype=np.float32)

    def update(self, raw_state, gripper_commands) -> None:
        """Record one executed chunk; ``raw_state`` is the state the chunk started from."""
        signs = np.where(np.asarray(gripper_commands) > 0, 1.0, -1.0)
        previous = np.concatenate([[self.last_grip], signs[:-1]])
        if np.any((previous < 0) & (signs > 0)):
            self.onsets += 1
            self.chunks_since = 0
            self.last_position = np.asarray(raw_state[:3], dtype=np.float64)
        else:
            self.chunks_since += 1
        self.last_grip = float(signs[-1])


class MemoryInput(nn.Module):
    def __init__(self, num_tokens: int, hidden_dim: int, width: int = 128) -> None:
        super().__init__()
        self.num_tokens, self.hidden_dim = int(num_tokens), int(hidden_dim)
        self.net = nn.Sequential(nn.Linear(MEMORY_DIM, width), nn.GELU(), nn.Linear(width, num_tokens * hidden_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.net(memory).view(memory.shape[0], self.num_tokens, self.hidden_dim)


def _state_forward(self, state: torch.Tensor) -> torch.Tensor:
    tokens = self._memoryless_forward(state)
    memory = getattr(self, "memory", None)
    if memory is None:
        return tokens
    module = self.memory_input
    offset = module(memory.to(device=tokens.device, dtype=module.net[0].weight.dtype))
    return tokens + offset.to(dtype=tokens.dtype)


def attach_memory(action_head) -> MemoryInput:
    """Register ``action_head.state_projection.memory_input`` and route state tokens through it."""
    projection = action_head.state_projection
    if not hasattr(projection, "memory_input"):
        reference = projection.position
        projection.memory_input = MemoryInput(projection.num_tokens, projection.hidden_dim).to(
            device=reference.device, dtype=reference.dtype)
        projection._memoryless_forward = projection.forward
        projection.forward = types.MethodType(_state_forward, projection)
    projection.memory = None
    return projection.memory_input


def set_memory(action_head, memory: torch.Tensor | None) -> None:
    """Decode with memory features ``[B, MEMORY_DIM]``; None gives the memoryless path."""
    action_head.state_projection.memory = memory


def has_memory_keys(state_dict) -> bool:
    return any(".state_projection.memory_input." in name for name in state_dict)
