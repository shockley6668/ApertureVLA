"""Shared-basis linear layers for candidate A.

This module implements the parameterization
``W_l = U diag(s_l) V.T + A_l B_l.T``. ``U`` and ``V`` are shared across a
group of layers. The original dense ``W_l`` is not retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class SharedBasisConfig:
    shared_rank: int = 64
    residual_rank: int = 8

    def __post_init__(self) -> None:
        if self.shared_rank < 1:
            raise ValueError("shared_rank must be positive")
        if self.residual_rank < 0:
            raise ValueError("residual_rank cannot be negative")


class SharedBasisLinear(nn.Module):
    """A single linear layer backed by group-shared basis matrices."""

    def __init__(
        self,
        shared_u: nn.Parameter,
        shared_v: nn.Parameter,
        bias: torch.Tensor | None,
        layer_index: int,
        config: SharedBasisConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if shared_u.ndim != 2 or shared_v.ndim != 2:
            raise ValueError("Shared basis matrices must be rank-2 tensors")
        if shared_u.shape[1] != shared_v.shape[1]:
            raise ValueError("Shared basis matrices must have the same rank")
        if shared_u.shape[1] < config.shared_rank:
            raise ValueError("shared_rank exceeds the available shared basis rank")
        self.layer_index = int(layer_index)
        self.config = config
        self.shared_u = shared_u
        self.shared_v = shared_v
        self.shared_scale = nn.Parameter(
            torch.zeros(config.shared_rank, device=device, dtype=dtype)
        )
        max_residual_rank = min(shared_u.shape[0], shared_v.shape[0])
        residual_rank = min(config.residual_rank, max_residual_rank)
        if residual_rank:
            self.residual_a = nn.Parameter(
                torch.empty(shared_u.shape[0], residual_rank, device=device, dtype=dtype)
            )
            self.residual_b = nn.Parameter(
                torch.empty(shared_v.shape[0], residual_rank, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_a", None)
            self.register_parameter("residual_b", None)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(device=device, dtype=dtype).clone())
        else:
            self.register_parameter("bias", None)
        self.register_buffer("fused_left", None, persistent=False)
        self.register_buffer("fused_right", None, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.shared_v.shape[0])
        nn.init.uniform_(self.shared_scale, -bound, bound)
        if self.residual_a is not None:
            nn.init.zeros_(self.residual_a)
            nn.init.zeros_(self.residual_b)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def in_features(self) -> int:
        return self.shared_v.shape[0]

    @property
    def out_features(self) -> int:
        return self.shared_u.shape[0]

    def effective_weight(self) -> torch.Tensor:
        shared = self.shared_u * self.shared_scale
        weight = shared @ self.shared_v.transpose(0, 1)
        if self.residual_a is not None and self.residual_b is not None:
            weight = weight + self.residual_a @ self.residual_b.transpose(0, 1)
        return weight

    @property
    def fused_rank(self) -> int:
        return int(self.shared_u.shape[1] + (
            self.residual_a.shape[1] if self.residual_a is not None else 0
        ))

    def fuse_for_inference(self) -> None:
        """Cache a single concatenated-rank factorization for inference."""
        shared_left = (self.shared_u * self.shared_scale).detach()
        if self.residual_a is None:
            self.fused_left = shared_left.contiguous()
            self.fused_right = self.shared_v.detach().contiguous()
            return
        self.fused_left = torch.cat(
            (shared_left, self.residual_a), dim=1
        ).detach().contiguous()
        self.fused_right = torch.cat(
            (self.shared_v.detach(), self.residual_b.detach()), dim=1
        ).contiguous()

    def invalidate_inference_fusion(self) -> None:
        self.fused_left = None
        self.fused_right = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fused_left is not None and self.fused_right is not None:
            if self.training:
                self.invalidate_inference_fusion()
            else:
                projected = inputs @ self.fused_right
                output = projected @ self.fused_left.transpose(0, 1)
                if self.bias is not None:
                    output = output + self.bias
                return output

        # y = x @ V diag(s) @ U.T + x @ B @ A.T + b.
        shared = (inputs @ self.shared_v) * self.shared_scale
        output = shared @ self.shared_u.transpose(0, 1)
        if self.residual_a is not None and self.residual_b is not None:
            residual = inputs @ self.residual_b
            output = output + residual @ self.residual_a.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output


def fuse_shared_basis_for_inference(root: nn.Module) -> int:
    """Materialize rank-concatenated caches on every shared-basis layer."""
    count = 0
    for module in root.modules():
        if isinstance(module, SharedBasisLinear):
            module.eval()
            module.fuse_for_inference()
            count += 1
    return count


class SharedBasisLinearGroup:
    """Factory and container for a group of shared-basis linear layers."""

    def __init__(self, layers: Sequence[SharedBasisLinear]) -> None:
        if not layers:
            raise ValueError("A shared-basis group needs at least one layer")
        first = layers[0]
        for layer in layers:
            if layer.shared_u is not first.shared_u or layer.shared_v is not first.shared_v:
                raise ValueError("All layers must share the same U and V parameters")
        self.layers = list(layers)

    @classmethod
    def from_dense_layers(
        cls,
        weights: Sequence[torch.Tensor],
        biases: Sequence[torch.Tensor | None],
        config: SharedBasisConfig,
        layer_names: Sequence[str] | None = None,
        rank_map: dict[str, int] | None = None,
        initialize: bool = True,
    ) -> "SharedBasisLinearGroup":
        if len(weights) != len(biases):
            raise ValueError("weights and biases must have the same length")
        if not weights:
            raise ValueError("weights cannot be empty")
        if rank_map is not None:
            if layer_names is None or len(layer_names) != len(weights):
                raise ValueError("layer_names is required when rank_map is provided")
            missing = sorted(set(layer_names) - set(rank_map))
            if missing:
                raise ValueError(f"rank_map is missing layers: {missing}")
            invalid = {
                name: rank for name, rank in rank_map.items() if rank < 0
            }
            if invalid:
                raise ValueError(f"rank_map contains negative ranks: {invalid}")
        shapes = {tuple(weight.shape) for weight in weights}
        if len(shapes) != 1:
            raise ValueError(f"All dense weights must have the same shape, got {shapes}")
        rows, columns = weights[0].shape
        maximum_rank = min(rows, columns)
        shared_rank = min(config.shared_rank, maximum_rank)
        default_residual_rank = min(config.residual_rank, maximum_rank)
        effective_config = SharedBasisConfig(shared_rank=shared_rank, residual_rank=default_residual_rank)

        if initialize:
            stacked = torch.stack([weight.detach().float() for weight in weights])
            mean_weight = stacked.mean(dim=0)
            left, _, right_t = torch.linalg.svd(mean_weight, full_matrices=False)
            shared_u = nn.Parameter(left[:, :shared_rank].clone())
            shared_v = nn.Parameter(right_t[:shared_rank].transpose(0, 1).clone())
        else:
            # Strict checkpoint restoration needs shapes/aliases, not a new SVD.
            shared_u = nn.Parameter(torch.zeros(rows, shared_rank, device=weights[0].device))
            shared_v = nn.Parameter(torch.zeros(columns, shared_rank, device=weights[0].device))

        layers = []
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            layer_name = layer_names[layer_index] if layer_names is not None else None
            if rank_map is not None and layer_name is not None:
                layer_residual_rank = min(int(rank_map[layer_name]), maximum_rank)
                layer_config = SharedBasisConfig(
                    shared_rank=shared_rank,
                    residual_rank=layer_residual_rank,
                )
            else:
                layer_config = effective_config
            layer_residual_rank = layer_config.residual_rank
            if initialize:
                current = weight.detach().float()
                projected_scale = torch.einsum(
                    "mn,mr,nr->r", current, shared_u, shared_v
                )
                approximate = shared_u @ torch.diag(projected_scale) @ shared_v.transpose(0, 1)
                residual = current - approximate
                residual_left, singular_values, residual_right_t = torch.linalg.svd(
                    residual, full_matrices=False
                )
                scale = singular_values[:layer_residual_rank].sqrt()
            layer = SharedBasisLinear(
                shared_u=shared_u,
                shared_v=shared_v,
                bias=bias,
                layer_index=layer_index,
                config=layer_config,
                device=weight.device,
                dtype=weight.dtype,
            )
            if initialize:
                layer.shared_scale.data.copy_(projected_scale.to(layer.shared_scale.dtype))
            layer_residual_rank = layer_config.residual_rank
            if layer_residual_rank and initialize:
                layer.residual_a.data.copy_(
                    (residual_left[:, :layer_residual_rank] * scale).to(
                        layer.residual_a.dtype
                    )
                )
                layer.residual_b.data.copy_(
                    (residual_right_t[:layer_residual_rank].transpose(0, 1) * scale).to(
                        layer.residual_b.dtype
                    )
                )
            elif not layer_residual_rank:
                layer.residual_a = None
                layer.residual_b = None
            if bias is not None:
                layer.bias.data.copy_(bias.to(layer.bias.device, dtype=layer.bias.dtype))
            layers.append(layer)
        return cls(layers)
