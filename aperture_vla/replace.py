"""Replace selected dense linear layers with shared-basis layers."""
from __future__ import annotations

import fnmatch
import re
from typing import Sequence

from torch import nn

from .shared_basis import SharedBasisConfig, SharedBasisLinearGroup


def _replace_child(root: nn.Module, path: str, layer: nn.Module) -> None:
    parent_path, child_name = path.rsplit(".", 1)
    parent = root.get_submodule(parent_path)
    if child_name.isdigit():
        parent[int(child_name)] = layer
    else:
        setattr(parent, child_name, layer)


def replace_linear_layers(
    model: nn.Module,
    patterns: Sequence[str],
    config: SharedBasisConfig,
    rank_map: dict[str, int] | None = None,
    initialize: bool = True,
) -> tuple[SharedBasisLinearGroup, list[tuple[str, nn.Linear]]]:
    """Replace every matching ``nn.Linear`` with a shared-basis layer.

    Patterns use :mod:`fnmatch` syntax by default. A pattern prefixed with
    ``regex:`` is matched with a full regular expression instead. Matching
    layers are grouped together and share one pair of basis matrices. The
    original modules are returned only for diagnostics; they are no longer
    registered in ``model``.
    """
    if not patterns:
        raise ValueError("At least one pattern is required")
    def matches_pattern(name: str, pattern: str) -> bool:
        if pattern.startswith("regex:"):
            return re.fullmatch(pattern.removeprefix("regex:"), name) is not None
        return fnmatch.fnmatchcase(name, pattern)

    matches = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=True)
        if isinstance(module, nn.Linear)
        and any(matches_pattern(name, pattern) for pattern in patterns)
    ]
    if not matches:
        raise ValueError(f"No linear layers matched patterns: {list(patterns)}")
    matches.sort(key=lambda item: item[0])
    weights = [module.weight for _, module in matches]
    biases = [module.bias for _, module in matches]
    layer_names = [name for name, _ in matches]
    group = SharedBasisLinearGroup.from_dense_layers(
        weights,
        biases,
        config,
        layer_names=layer_names,
        rank_map=rank_map,
        initialize=initialize,
    )
    for (path, original), replacement in zip(matches, group.layers):
        if original.weight.shape != (replacement.out_features, replacement.in_features):
            raise RuntimeError("Replacement changed a layer's weight shape")
        _replace_child(model, path, replacement)
    return group, matches
