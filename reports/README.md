# Reports

Small files that back the paper tables. Checkpoints and full logs stay on the training machine.

| Path | What it supports |
|---|---|
| `formal_early_always/` | Four-suite LIBERO, seed 7, 2,000 episodes |
| `grpo_round1/` | Head-only GRPO with and without episode memory |
| `profile_4090_v2/` | RTX 4090 latency ablation (eager / cache / batched views / CUDA graph) |
| `edge/bpu_v370_fastperf/` | RDK S100P `fast-perf` board latency |
