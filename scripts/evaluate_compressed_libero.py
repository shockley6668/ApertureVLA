"""Run LIBERO evaluation with an ApertureVLA (or unpooled) checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libero_compat import apply_libero_compat


apply_libero_compat()

import vla_adapter.rollout as rollout
from aperture_vla.policy import ApertureVLAPolicy
from turbovla.evaluation.suite_policy import (
    get_libero_dummy_action,
    rotate_libero_image,
    set_seed_everywhere,
)


def _import_compressed_adapter():
    return (
        ApertureVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    )


rollout._import_turbovla_adapter = _import_compressed_adapter

_run_episode = rollout._run_episode


def _gripper_events(executed_actions):
    """Step indices where the executed gripper command flips (+1 closes, -1 opens in LIBERO)."""
    if not executed_actions:
        return [], []
    gripper = np.sign(np.asarray(executed_actions, dtype=np.float32)[:, 6])
    flips = np.nonzero(gripper[1:] != gripper[:-1])[0] + 1
    closes = [int(i) for i in flips if gripper[i] > 0]
    opens = [int(i) for i in flips if gripper[i] < 0]
    return closes, opens


def _run_episode_with_reset(cfg, env, policy, task_description, *args, **kwargs):
    # Cached vision-language conditions must never leak across episodes.
    if hasattr(policy, "reset"):
        policy.reset()
    result = _run_episode(cfg, env, policy, task_description, *args, **kwargs)
    log_path = os.environ.get("CORL_EPISODE_LOG")
    if log_path:
        success, _, executed = result
        closes, opens = _gripper_events(executed)
        row = {
            "suite": cfg.task_suite_name,
            "task": task_description,
            "success": bool(success),
            "steps": len(executed),
            "close_steps": closes,
            "open_steps": opens,
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return result


rollout._run_episode = _run_episode_with_reset

if __name__ == "__main__":
    rollout.main()
