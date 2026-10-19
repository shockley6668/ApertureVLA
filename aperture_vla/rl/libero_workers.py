"""Parallel LIBERO environment workers for on-policy RL rollouts.

Each worker process owns one LIBERO environment (recreated only when its task changes). The parent
batches policy inference on the GPU and sends action chunks; workers return policy-orientation RGB
views, the raw 8-D state and termination flags. Initial states are sampled from fresh random
placements (``env.reset`` with a seed) and replayed with ``set_init_state`` so a GRPO group can share
one start; the benchmark's fixed evaluation initial states are never used here.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _worker(conn, suite_name: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero_compat import apply_libero_compat

    apply_libero_compat()
    import vla_adapter.rollout as rollout
    from turbovla.evaluation.policy import state_from_libero_obs
    from turbovla.evaluation.suite_policy import get_libero_dummy_action, rotate_libero_image

    cfg = SimpleNamespace(env_img_res=256, seed=0, control_mode="relative", libero_root=str(ROOT / "third_party/LIBERO"),
                          mujoco_gl="egl", pyopengl_platform="egl", osmesa_library=None)
    rollout._ensure_libero_import_path(cfg)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    max_steps = rollout.TASK_MAX_STEPS[suite_name]
    dummy = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    env, env_task, instruction, steps = None, None, None, 0

    def ensure(task_id):
        nonlocal env, env_task, instruction
        if env_task != task_id:
            if env is not None:
                env.close()
            env, instruction = rollout._make_libero_env(suite.get_task(task_id), cfg)
            env_task = task_id

    def observe(obs):
        return {"primary": rotate_libero_image(obs["agentview_image"]).astype(np.uint8),
                "wrist": rotate_libero_image(obs["robot0_eye_in_hand_image"]).astype(np.uint8),
                "state": state_from_libero_obs(obs).astype(np.float32), "instruction": instruction}

    while True:
        command, payload = conn.recv()
        if command == "sample_init":
            task_id, seed = payload
            ensure(task_id)
            env.seed(int(seed))
            np.random.seed(int(seed))
            env.reset()
            conn.send(env.get_sim_state().copy())
        elif command == "reset":
            task_id, init_state = payload
            ensure(task_id)
            env.reset()
            obs = env.set_init_state(init_state)
            for _ in range(10):  # let objects settle, as in the evaluator
                obs, _, _, _ = env.step(dummy.tolist())
            steps = 0
            conn.send(observe(obs))
        elif command == "step":
            done = False
            for action in payload:
                obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
                steps += 1
                if done or steps >= max_steps:
                    break
            conn.send({**observe(obs), "success": bool(done), "finished": bool(done or steps >= max_steps),
                       "steps": steps})
        elif command == "close":
            if env is not None:
                env.close()
            conn.close()
            return


class LiberoWorkers:
    """A pool of environment processes driven in lock step by the parent."""

    def __init__(self, count: int, suite: str = "libero_10") -> None:
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, suite), daemon=True)
            process.start()
            self.pipes.append(parent)
            self.processes.append(process)

    def __len__(self) -> int:
        return len(self.pipes)

    def call(self, indices, command, payloads):
        for index, payload in zip(indices, payloads):
            self.pipes[index].send((command, payload))
        return [self.pipes[index].recv() for index in indices]

    def close(self) -> None:
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
