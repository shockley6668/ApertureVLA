# GRPO post-training round 1 (episode memory vs. memoryless)

Base: `outputs/checkpoints/early_compression_split4_pool2_detail_10k_v1/evaluation_ema.pth`
(ApertureVLA, formal result in `reports/formal_early_always`). Code: `scripts/train_grpo.py`,
`scripts/run_grpo_round.sh`, commit bf5fa6c, RTX 5090.

Training: action head only (vision encoder, text encoder and fusion frozen), Gaussian policy around the
regression head (init std 0.1), GRPO with PPO clip 0.2, L2 anchor 1.0 to the initial head, lr 1e-5,
40 iterations x 16 groups x 8 rollouts = 5,120 LIBERO-10 episodes, task list 8,8,8,6,6,0,9,3,2,7.
Training placements come from the environment's seeded random initializer, not the 50 fixed
evaluation initial states. `grpo_mem1` adds the zero-initialised 6-d episode-memory input
(66,944 parameters); `grpo1` is identical without it. One training seed each.

Evaluation: seed 7, 50 fixed initial states x 10 tasks per suite, chunks of 12, refresh every chunk.

| Suite | Base | GRPO, no memory | GRPO + memory |
|---|---:|---:|---:|
| Spatial | 468 | 477 | 484 |
| Object | 487 | 490 | 493 |
| Goal | 483 | 477 | 475 |
| LIBERO-10 | 455 | 452 | 459 |
| **Total /2000** | **1893** | **1896** | **1911** |

LIBERO-10 per task (/50), iteration 40:

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 45 | 49 | 47 | 47 | 49 | 48 | 42 | 47 | 35 | 46 |
| No memory | 37 | 49 | 45 | 45 | 46 | 46 | 43 | 44 | 48 | 49 |
| Memory | 43 | 49 | 44 | 49 | 44 | 46 | 47 | 47 | 43 | 47 |

Iteration 20 on LIBERO-10: no memory 447, memory 452.

Reading: the memory run raises the worst LIBERO-10 task from 35 to 43 and the four-suite total by 18
episodes; the memoryless run fixes task 8 but loses task 0 (45 to 37). The LIBERO-10 total changes are
within single-seed noise. Memory evaluation runs the head eagerly (outside the CUDA graph), so no
latency is reported for it.

The detail-bypass gate of the base checkpoint is 0.0157 after training (unchanged by GRPO, which does
not train it).
