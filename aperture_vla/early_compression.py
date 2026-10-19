from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Mapping, Sequence

import torch
from torch import nn

from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA
from turbovla.models.vision_encoder import DINOv3VisionEncoder


class EarlyDINOv3VisionEncoder(DINOv3VisionEncoder):
    """Run shallow full-grid layers, then pooled deep-grid layers."""

    def __init__(self, config: TurboVLAConfig, split_layer: int = 4, merge_factor: int = 2) -> None:
        super().__init__(config.vision)
        if split_layer < 1:
            raise ValueError("split_layer must be positive")
        if merge_factor < 2:
            raise ValueError("merge_factor must be at least 2")
        layers = self._backbone_layers()
        if split_layer >= len(layers):
            raise ValueError("split_layer must be before the final DINOv3 layer")
        patches = config.vision.image_size // self.patch_size
        if patches < merge_factor or patches % merge_factor:
            raise ValueError("patch grid must divide evenly by merge_factor")
        self.early_split_layer = int(split_layer)
        self.early_merge_factor = int(merge_factor)
        self.early_grid_size = patches // merge_factor
        self.batch_views = False
        self._pooled_rope_cache = {}
        # DINOv3 applies RoPE in every layer; DINOv2-style backbones (DINOv2, WebSSL) add learned
        # absolute positions once at the input, so pooled tokens keep their averaged position.
        self.backbone_kind = "dinov3" if hasattr(self.backbone, "rope_embeddings") else "dinov2"
        if self.backbone_kind == "dinov2":
            self.prefix_tokens = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)

    def _backbone_layers(self):
        if hasattr(self.backbone, "layer"):
            return self.backbone.layer
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
            return self.backbone.model.layer
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        raise AttributeError("Unsupported vision backbone: no layer list")

    @staticmethod
    def _run_layer(layer, hidden, **kwargs):
        output = layer(hidden, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = patches.shape
        grid = int(math.isqrt(tokens))
        if grid * grid != tokens:
            raise ValueError(f"Expected a square patch grid, got {tokens} tokens")
        factor = self.early_merge_factor
        if grid % factor:
            raise ValueError(f"Patch grid {grid} does not divide by merge factor {factor}")
        pooled_grid = grid // factor
        return patches.reshape(
            batch, pooled_grid, factor, pooled_grid, factor, width
        ).mean(dim=(2, 4)).reshape(batch, pooled_grid * pooled_grid, width)

    def _pooled_position_embeddings(self, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        key = (batch, pixel_values.device, pixel_values.dtype)
        if not self.training and key in self._pooled_rope_cache:
            return self._pooled_rope_cache[key]
        size = self.early_grid_size * self.patch_size
        probe = torch.empty(
            (batch, 3, size, size),
            device=pixel_values.device,
            dtype=pixel_values.dtype if pixel_values.is_floating_point() else torch.float32,
        )
        position = self.backbone.rope_embeddings(probe)
        if not self.training:
            self._pooled_rope_cache[key] = position
        return position

    def _encode_images_with_detail(self, pixel_values: torch.Tensor, shallow_only: bool = False):
        height, width = pixel_values.shape[-2:]
        if self.backbone_kind == "dinov3" and (height % self.patch_size or width % self.patch_size):
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        # DINOv2-style patch embedding is a strided convolution: a 256 px image with 14 px patches
        # gives an 18 x 18 grid (the last 4 px are dropped), with interpolated position embeddings.
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        grad_context = torch.no_grad() if self.config.frozen else nullcontext()

        backbone = self.backbone
        with grad_context:
            with autocast_context:
                patch_weight = getattr(backbone.embeddings.patch_embeddings, "weight", None)
                if patch_weight is None:
                    patch_weight = backbone.embeddings.patch_embeddings.projection.weight
                pixel_values = pixel_values.to(patch_weight.dtype)
                hidden = backbone.embeddings(pixel_values)
                rope = self.backbone_kind == "dinov3"
                full_position = backbone.rope_embeddings(pixel_values) if rope else None
                layers = self._backbone_layers()
                for layer in layers[: self.early_split_layer]:
                    hidden = (self._run_layer(layer, hidden, position_embeddings=full_position) if rope
                              else self._run_layer(layer, hidden))
                if hidden.shape[1] != expected_patches + self.prefix_tokens:
                    raise RuntimeError(
                        f"DINOv3 produced {hidden.shape[1]} tokens; expected "
                        f"{expected_patches + self.prefix_tokens} including prefixes"
                    )
                shallow_detail = hidden[:, self.prefix_tokens :, :]
                if shallow_only:
                    return None, shallow_detail
                prefix = hidden[:, : self.prefix_tokens, :]
                patches = hidden[:, self.prefix_tokens :, :]
                hidden = torch.cat((prefix, self._pool_patches(patches)), dim=1)
                if rope:
                    pooled_position = self._pooled_position_embeddings(pixel_values)
                    for layer in layers[self.early_split_layer :]:
                        hidden = self._run_layer(layer, hidden, position_embeddings=pooled_position)
                    deep = backbone.norm(hidden)
                else:
                    for layer in layers[self.early_split_layer :]:
                        hidden = self._run_layer(layer, hidden)
                    deep = backbone.layernorm(hidden)
        return deep, shallow_detail

    def forward(self, pixel_values: torch.Tensor, shallow_only: bool = False):
        """Return (deep, detail) per view; ``shallow_only`` stops after the split layer (deep=None)."""
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.batch_views:
            deep, detail = self._encode_images_with_detail(pixel_values.flatten(0, 1), shallow_only)
            detail = detail.view(batch_size, num_views, *detail.shape[1:])
            if deep is not None:
                deep = deep.view(batch_size, num_views, *deep.shape[1:])
            return deep, detail
        deep, detail = zip(
            *(
                self._encode_images_with_detail(pixel_values[:, view], shallow_only)
                for view in range(num_views)
            ),
            strict=True,
        )
        detail = torch.stack(detail, dim=1)
        return (None if shallow_only else torch.stack(deep, dim=1)), detail


class ShallowDetailProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def set_gate_scale(self, scale: float) -> None:
        """Reparameterize the gate as ``scale * g`` without changing its current value.

        A zero-initialized gate trained with the head learning rate barely opens (0.016 after 10k
        steps), so the detail bypass stays almost unused. Scaling multiplies the gate's effective
        learning rate by ``scale``. The scale is a persistent buffer so checkpoints restore it.
        """
        current = float(self.effective_gate())
        if "gate_scale" not in self._buffers:
            self.register_buffer("gate_scale", torch.ones((), device=self.gate.device))
        self.gate_scale.fill_(float(scale))
        with torch.no_grad():
            self.gate.fill_(current / float(scale))

    def effective_gate(self) -> torch.Tensor:
        scale = self._buffers.get("gate_scale")
        return self.gate if scale is None else self.gate * scale.to(self.gate.dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.net(self.input_norm(tokens))
        if tokens.shape[-1] == projected.shape[-1]:
            projected = projected + tokens
        return self.effective_gate() * self.output_norm(projected)


class EarlyCompressionTurboVLA(TurboVLA):
    """TurboVLA with pooled deep vision tokens and optional action detail tokens."""

    def __init__(
        self,
        config: TurboVLAConfig,
        split_layer: int = 4,
        merge_factor: int = 2,
        shallow_detail_to_action: bool = True,
    ) -> None:
        super().__init__(config)
        self.vision_encoder = EarlyDINOv3VisionEncoder(
            config, split_layer=split_layer, merge_factor=merge_factor
        )
        self.early_compression_metadata = {
            "method": "early_dinov3_token_pooling",
            "split_layer": int(split_layer),
            "merge_factor": int(merge_factor),
            "shallow_detail_to_action": bool(shallow_detail_to_action),
        }
        self.shallow_detail_to_action = bool(shallow_detail_to_action)
        if shallow_detail_to_action:
            self.shallow_detail_projection = ShallowDetailProjection(
                input_dim=self.vision_encoder.hidden_size,
                output_dim=config.interaction.hidden_dim,
                dropout=config.vision.dropout,
            )
        else:
            self.register_parameter("shallow_detail_projection", None)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        deep, _ = self.vision_encoder(pixel_values)
        tokens = self.vision_projection(deep)
        return self._position_visual_tokens(tokens).flatten(1, 2)

    def encode_condition(self, instructions: Sequence[str], samples):
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions, device=device
            )
            if text_tokens.shape[0] != pixel_values.shape[0]:
                raise ValueError("instruction batch size does not match image batch size")
            deep, detail = self.vision_encoder(pixel_values)
            deep = self.vision_projection(deep)
            visual_tokens = self._position_visual_tokens(deep).flatten(1, 2)
            visual_tokens, text_tokens = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
            )
            condition = torch.cat([visual_tokens, text_tokens], dim=1)
            if self.shallow_detail_to_action:
                detail_tokens = self.shallow_detail_projection(detail.flatten(1, 2))
                detail_tokens = detail_tokens.to(device=condition.device, dtype=condition.dtype)
            else:
                detail_tokens = None
            return condition, detail_tokens

    def encode_detail(self, samples) -> torch.Tensor:
        """Fast path: shallow DINOv3 layers only, projected to action-head detail tokens."""
        if not self.shallow_detail_to_action:
            raise RuntimeError("encode_detail requires the shallow detail branch")
        pixel_values = self._normalize_samples(samples)
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            _, detail = self.vision_encoder(pixel_values, shallow_only=True)
            return self.shallow_detail_projection(detail.flatten(1, 2))

    def forward(self, instructions, samples, state):
        pixels = samples["dinov3"] if isinstance(samples, Mapping) else samples
        if pixels.ndim == 6 and pixels.shape[1] == 2 and self.shallow_detail_to_action:
            # Staleness training: [B, (stale, current), V, 3, H, W].
            condition, _ = self.encode_condition(instructions, pixels[:, 0])
            detail_tokens = self.encode_detail(pixels[:, 1]).to(dtype=condition.dtype)
            condition = torch.cat([condition, detail_tokens], dim=1)
            action_dtype = self.action_head.decoder.action_queries.weight.dtype
            return self.action_head(condition.to(dtype=action_dtype), state.to(dtype=action_dtype))
        condition, detail_tokens = self.encode_condition(instructions, samples)
        if detail_tokens is not None:
            condition = torch.cat([condition, detail_tokens], dim=1)
        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        return self.action_head(condition.to(dtype=action_dtype), state.to(dtype=action_dtype))


def build_early_compression_turbovla(
    config: TurboVLAConfig | Mapping,
    split_layer: int = 4,
    merge_factor: int = 2,
    shallow_detail_to_action: bool = True,
) -> EarlyCompressionTurboVLA:
    if not isinstance(config, TurboVLAConfig):
        config = TurboVLAConfig.from_mapping(config)
    return EarlyCompressionTurboVLA(
        config,
        split_layer=split_layer,
        merge_factor=merge_factor,
        shallow_detail_to_action=shallow_detail_to_action,
    )
