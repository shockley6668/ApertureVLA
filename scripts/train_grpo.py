"""GRPO fine-tuning of the compact VLA's action head with binary task success.

The deterministic regression head becomes the mean of a Gaussian policy over the normalized
12x7 action chunk, a = clip(mu(o) + sigma * eps, -1, 1), with a learnable per-dimension sigma.
Vision encoder and vision-language fusion stay frozen, so the action-head input (condition tokens)
is computed once during the rollout and cached; updates only re-run the head.

Per iteration: sample ``groups`` fresh random placements over the chosen tasks, roll out
``group_size`` stochastic episodes from each, reward 1 on success, and normalize rewards within
each group (GRPO). Chunks of groups with identical outcomes carry no signal and are skipped. The
loss is the PPO-clipped surrogate per chunk plus an L2 anchor between the new and the initial head
means. Checkpoints are written in the compressed format so the standard evaluator loads them.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from aperture_vla.fast_inference import enable_batched_views, enable_text_cache, split_condition_forward
from aperture_vla.memory_head import EpisodeMemory, attach_memory, set_memory
from aperture_vla.policy import ApertureVLAPolicy
from aperture_vla.rl.libero_workers import LiberoWorkers

STATS = ROOT / "third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json"


def load_policy(checkpoint):
    os.environ.update(CORL_CACHE_TEXT="1", CORL_BATCH_VIEWS="1", CORL_CUDA_GRAPH="0", CORL_REFRESH_MODE="always")
    policy = ApertureVLAPolicy(
        ckpt_path=checkpoint, dinov3_path="assets/models/dinov3-vitb16", bert_path="assets/models/bert-base-uncased",
        stats_path=STATS, stats_key="libero_all4_no_noops", normalize_binary_gripper="false",
        device=torch.device("cuda"), precision="bf16", verbose=False)
    model = policy.model.eval()
    enable_text_cache(model)
    enable_batched_views(model)
    return policy, model


class GaussianHead(nn.Module):
    """Wraps the action head: mean from the head, learnable log std per action dimension."""

    def __init__(self, action_head, action_dim, init_std):
        super().__init__()
        self.head = action_head
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

    def mean(self, tokens, state):
        # The head trains in fp32: bf16 weights would round away updates at small learning rates.
        return self.head(tokens.float(), state.float()).float()

    def std(self):
        return self.log_std.clamp(math.log(1e-3), math.log(0.5)).exp()

    def log_prob(self, mean, action):
        std = self.std()
        return (-0.5 * ((action - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(dim=(-1, -2))


@torch.inference_mode()
def encode(policy, encode_fn, observations):
    samples, states = policy._build_batch([o["primary"] for o in observations], [o["wrist"] for o in observations],
                                          [o["state"] for o in observations])
    samples, states = policy._prepare_model_inputs(samples, states)
    tokens = encode_fn([o["instruction"] for o in observations], samples)
    return tokens, states


def rollout_iteration(args, policy, encode_fn, gaussian, workers, rng, iteration):
    tasks = [int(t) for t in args.tasks.split(",")]
    groups = []
    for g in range(args.groups):
        task = tasks[rng.randrange(len(tasks))]
        seed = (args.seed + iteration * 1009 + g * 7919) % (2**31 - 1)
        init_state = workers.call([0], "sample_init", [(task, seed)])[0]
        groups.append((task, init_state))
    episodes = [(gi, task, init) for gi, (task, init) in enumerate(groups) for _ in range(args.group_size)]
    chunks, rewards, lengths = [], [None] * len(episodes), [0] * len(episodes)
    memories = [EpisodeMemory() for _ in episodes]
    for wave in range(0, len(episodes), len(workers)):
        ids = list(range(wave, min(wave + len(workers), len(episodes))))
        slots = list(range(len(ids)))
        obs = workers.call(slots, "reset", [(episodes[i][1], episodes[i][2]) for i in ids])
        active = dict(zip(slots, ids))
        current = dict(zip(slots, obs))
        while active:
            slot_list = list(active)
            tokens, states = encode(policy, encode_fn, [current[s] for s in slot_list])
            features = np.stack([memories[active[s]].features(current[s]["state"]) for s in slot_list])
            with torch.inference_mode():
                if args.memory:
                    set_memory(gaussian.head, torch.from_numpy(features).cuda())
                mean = gaussian.mean(tokens, states)
                set_memory(gaussian.head, None)
                action = (mean + gaussian.std() * torch.randn_like(mean)).clamp(-1.0, 1.0)
                logp = gaussian.log_prob(mean, action)
            env_actions = []
            for row, slot in enumerate(slot_list):
                chunk = action[row].cpu().numpy()
                env_actions.append(np.stack([policy._normalized_row_to_env_action(r) for r in chunk]))
                memories[active[slot]].update(current[slot]["state"], env_actions[-1][:, 6])
                chunks.append({"episode": active[slot], "tokens": tokens[row].to("cpu"), "state": states[row].to("cpu"),
                               "memory": torch.from_numpy(features[row]),
                               "action": action[row].to("cpu"), "logp": float(logp[row])})
            results = workers.call(slot_list, "step", env_actions)
            for slot, result in zip(slot_list, results):
                episode = active[slot]
                if result["finished"]:
                    rewards[episode] = 1.0 if result["success"] else 0.0
                    lengths[episode] = result["steps"]
                    del active[slot]
                else:
                    current[slot] = result
    # Group-relative advantages; groups with identical outcomes give zero signal.
    advantages = [0.0] * len(episodes)
    useful_groups = 0
    for gi in range(args.groups):
        members = [i for i, e in enumerate(episodes) if e[0] == gi]
        values = np.asarray([rewards[i] for i in members])
        if values.std() < 1e-6:
            continue
        useful_groups += 1
        for i, v in zip(members, values):
            advantages[i] = float((v - values.mean()) / (values.std() + 1e-6))
    for chunk in chunks:
        chunk["advantage"] = advantages[chunk["episode"]]
    stats = {"episodes": len(episodes), "success_rate": float(np.mean(rewards)), "useful_groups": useful_groups,
             "chunks": len(chunks), "mean_steps": float(np.mean(lengths))}
    return [c for c in chunks if c["advantage"] != 0.0], stats


def update(args, gaussian, reference, optimizer, chunks):
    if not chunks:
        return {"updates": 0}
    device = gaussian.log_std.device
    losses, ratios, clipped = [], [], []
    for _ in range(args.epochs):
        random.shuffle(chunks)
        for start in range(0, len(chunks), args.minibatch):
            batch = chunks[start:start + args.minibatch]
            tokens = torch.stack([c["tokens"] for c in batch]).to(device)
            state = torch.stack([c["state"] for c in batch]).to(device)
            action = torch.stack([c["action"] for c in batch]).to(device).float()
            old_logp = torch.tensor([c["logp"] for c in batch], device=device)
            advantage = torch.tensor([c["advantage"] for c in batch], device=device)
            if args.memory:
                set_memory(gaussian.head, torch.stack([c["memory"] for c in batch]).to(device))
            mean = gaussian.mean(tokens, state)
            set_memory(gaussian.head, None)
            with torch.no_grad():
                ref_mean = reference(tokens.float(), state.float()).float()
            ratio = torch.exp(gaussian.log_prob(mean, action) - old_logp)
            surrogate = torch.min(ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage)
            anchor = (mean - ref_mean).pow(2).mean()
            loss = -surrogate.mean() + args.anchor * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussian.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
            ratios.append(float(ratio.mean()))
            clipped.append(float(((ratio - 1).abs() > args.clip).float().mean()))
    return {"updates": len(losses), "loss": float(np.mean(losses)), "ratio": float(np.mean(ratios)),
            "clip_frac": float(np.mean(clipped)), "std": gaussian.std().detach().cpu().tolist()}


def unwrap_state_keys(state):
    """Drop the inference-time text-cache wrapper so keys match the evaluator's strict load."""
    prefix = "text_encoder.encoder."
    return {("text_encoder." + k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}


def save(args, policy_checkpoint, model, gaussian, iteration):
    template = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
    state = unwrap_state_keys({k: v.detach().cpu() for k, v in model.state_dict().items()})
    out = dict(template)
    out["model_state_dict"] = state
    out.pop("ema_model_state_dict", None)
    out["grpo"] = {"iteration": iteration, "log_std": gaussian.log_std.detach().cpu().tolist(), "args": vars(args)}
    path = Path(args.output) / f"iter{iteration:04d}" / "evaluation_ema.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--std-lr", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--anchor", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--memory", action="store_true", help="add the zero-initialised episode-memory input")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy, model = load_policy(args.checkpoint)
    encode_fn, _ = split_condition_forward(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.action_head.float()
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    reference = copy.deepcopy(model.action_head).eval().requires_grad_(False)
    if args.memory:
        attach_memory(model.action_head).requires_grad_(True)
    gaussian = GaussianHead(model.action_head, model.action_dim, args.init_std).cuda()
    optimizer = torch.optim.Adam([{"params": list(model.action_head.parameters()), "lr": args.lr},
                                  {"params": [gaussian.log_std], "lr": args.std_lr}])
    workers = LiberoWorkers(args.workers)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    log = open(Path(args.output) / "train_log.jsonl", "a", encoding="utf-8")
    try:
        for iteration in range(1, args.iterations + 1):
            started = time.time()
            chunks, stats = rollout_iteration(args, policy, encode_fn, gaussian, workers, rng, iteration)
            rollout_seconds = time.time() - started
            stats.update(update(args, gaussian, reference, optimizer, chunks))
            stats.update(iteration=iteration, rollout_s=round(rollout_seconds, 1), total_s=round(time.time() - started, 1))
            print(json.dumps(stats), flush=True)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            if iteration % args.save_every == 0 or iteration == args.iterations:
                print("saved", save(args, args.checkpoint, model, gaussian, iteration), flush=True)
    finally:
        workers.close()
        log.close()


if __name__ == "__main__":
    main()
