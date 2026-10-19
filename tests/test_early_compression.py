from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "TurboVLA"))

from aperture_vla.early_compression import (
    EarlyDINOv3VisionEncoder,
    ShallowDetailProjection,
)


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = self.FakeEmbeddings()
        self.model = SimpleNamespace(layer=[self.FakeLayer() for _ in range(8)])
        self.norm = nn.LayerNorm(8)
        self.config = SimpleNamespace(patch_size=16, image_size=64)

    class FakeLayer(nn.Module):
        def forward(self, hidden_states, position_embeddings=None):
            return hidden_states

    class FakeEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embeddings = SimpleNamespace(weight=torch.empty(8, 3))

        def forward(self, pixel_values):
            tokens = (pixel_values.shape[-1] // 16) ** 2
            return torch.randn(pixel_values.shape[0], tokens, 8)

    def rope_embeddings(self, pixel_values):
        count = (pixel_values.shape[-1] // 16) ** 2
        return torch.ones(1, count, 8), torch.zeros(1, count, 8)


def make_encoder():
    encoder = EarlyDINOv3VisionEncoder.__new__(EarlyDINOv3VisionEncoder)
    nn.Module.__init__(encoder)
    encoder.config = SimpleNamespace(
        num_views=1,
        frozen=False,
        compute_precision="fp32",
    )
    encoder.backbone = FakeDINO()
    encoder.hidden_size = 8
    encoder.patch_size = 16
    encoder.prefix_tokens = 0
    encoder.num_patches = 16
    encoder.early_split_layer = 4
    encoder.early_merge_factor = 2
    encoder.early_grid_size = 2
    encoder.batch_views = False
    encoder._pooled_rope_cache = {}
    encoder.backbone_kind = "dinov3"
    return encoder


def test_pool_patches_averages_two_by_two_cells():
    encoder = make_encoder()
    patches = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).expand(1, 16, 8)
    pooled = encoder._pool_patches(patches)
    assert pooled.shape == (1, 4, 8)
    expected = torch.tensor([2.5, 4.5, 10.5, 12.5], dtype=torch.float32)
    assert torch.allclose(pooled[:, :, 0], expected)


def test_encoder_output_shapes_and_dtype():
    encoder = make_encoder()
    pixels = torch.randn(2, 1, 3, 64, 64)
    deep, detail = encoder(pixels)
    assert deep.shape == (2, 1, 4, 8)
    assert detail.shape == (2, 1, 16, 8)
    assert deep.dtype == detail.dtype


def test_shallow_projection_has_nonzero_gradient():
    projection = ShallowDetailProjection(8, 16, 0.0)
    with torch.no_grad():
        projection.gate.fill_(1.0)
    tokens = torch.randn(2, 16, 8, requires_grad=True)
    projection(tokens).square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert projection.net[0].weight.grad is not None
    assert projection.net[0].weight.grad.abs().sum() > 0


def test_shallow_projection_starts_zero_and_can_learn():
    projection = ShallowDetailProjection(8, 16, 0.0)
    tokens = torch.randn(2, 16, 8)
    initial = projection(tokens)
    assert torch.count_nonzero(initial) == 0
    with torch.no_grad():
        projection.gate.fill_(0.1)
    learned = projection(tokens)
    assert learned.abs().sum() > 0
    learned.square().mean().backward()
    assert projection.gate.grad is not None
    assert projection.gate.grad.abs() > 0
    assert torch.isfinite(projection.gate.grad).all()


def test_pooling_rejects_non_square_grid():
    encoder = make_encoder()
    try:
        encoder._pool_patches(torch.randn(2, 15, 8))
    except ValueError:
        return
    raise AssertionError("Expected non-square pooling to fail")


def test_shallow_only_skips_deep_layers():
    encoder = make_encoder()
    pixels = torch.randn(2, 1, 3, 64, 64)
    deep, detail = encoder(pixels, shallow_only=True)
    assert deep is None
    assert detail.shape == (2, 1, 16, 8)


def test_batched_views_matches_shapes_of_looped_views():
    encoder = make_encoder()
    encoder.config.num_views = 2
    pixels = torch.randn(3, 2, 3, 64, 64)
    looped_deep, looped_detail = encoder(pixels)
    encoder.batch_views = True
    batched_deep, batched_detail = encoder(pixels)
    assert batched_deep.shape == looped_deep.shape == (3, 2, 4, 8)
    assert batched_detail.shape == looped_detail.shape == (3, 2, 16, 8)


def test_detail_gate_scale_preserves_output_and_round_trips():
    projection = ShallowDetailProjection(8, 16, 0.0)
    with torch.no_grad():
        projection.gate.fill_(0.016)
    tokens = torch.randn(2, 16, 8)
    before = projection(tokens)
    projection.set_gate_scale(50.0)
    assert torch.allclose(projection(tokens), before, atol=1e-6)
    assert abs(float(projection.gate) - 0.016 / 50) < 1e-9
    restored = ShallowDetailProjection(8, 16, 0.0)
    restored.set_gate_scale(50.0)
    restored.load_state_dict(projection.state_dict(), strict=True)
    assert torch.allclose(restored(tokens), before, atol=1e-6)
