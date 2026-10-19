import copy

import numpy as np
import torch

from aperture_vla.memory_head import MEMORY_DIM, EpisodeMemory, attach_memory, has_memory_keys, set_memory
from turbovla.models.action_head import TurboVLAActionHead
from turbovla.models.configuration import ActionHeadConfig


def _head():
    torch.manual_seed(0)
    config = ActionHeadConfig(num_layers=1, mlp_hidden_dim=32, state_hidden_dim=16, dropout=0.0)
    return TurboVLAActionHead(config, hidden_dim=32, nheads=4, dim_feedforward=64).eval()


def test_fresh_memory_leaves_head_unchanged():
    head = _head()
    original = copy.deepcopy(head)
    attach_memory(head)
    tokens, state = torch.randn(3, 5, 32), torch.randn(3, 8)
    set_memory(head, torch.rand(3, MEMORY_DIM))
    with torch.no_grad():
        assert torch.equal(head(tokens, state), original(tokens, state))


def test_memory_changes_output_after_update_and_reloads():
    head = _head()
    module = attach_memory(head)
    torch.nn.init.normal_(module.net[-1].weight, std=0.1)
    tokens, state = torch.randn(2, 5, 32), torch.randn(2, 8)
    with torch.no_grad():
        set_memory(head, torch.zeros(2, MEMORY_DIM))
        base = head(tokens, state)
        set_memory(head, torch.ones(2, MEMORY_DIM))
        moved = head(tokens, state)
    assert not torch.allclose(base, moved)
    state_dict = head.state_dict()
    assert has_memory_keys({"action_head." + k: v for k, v in state_dict.items()})
    clone = _head()
    attach_memory(clone)
    clone.load_state_dict(state_dict, strict=True)
    set_memory(clone, torch.ones(2, MEMORY_DIM))
    with torch.no_grad():
        assert torch.equal(clone(tokens, state), moved)


def test_episode_memory_counts_grasp_onsets():
    memory = EpisodeMemory()
    start = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0], dtype=np.float32)
    assert memory.features(start)[0] == 0.0
    memory.update(start, [-1, -1, 1, 1])
    assert memory.onsets == 1
    memory.update(start, [1, 1, 1])
    assert memory.onsets == 1 and memory.chunks_since == 1
    memory.update(start + 0.05, [-1, 1])
    assert memory.onsets == 2
    moved = start.copy()
    moved[0] += 0.02
    features = memory.features(moved)
    assert features[0] == 2 / 8 and features[5] == 1.0
    np.testing.assert_allclose(features[2:5], np.clip(10 * (moved[:3] - (start[:3] + 0.05)), -1, 1), rtol=1e-5)
    memory.reset()
    assert memory.onsets == 0 and memory.last_position is None
