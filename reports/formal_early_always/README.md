# ApertureVLA 10k checkpoint, optimized inference, formal 50-trial evaluation

RTX 4090, seed 7, 50 trials x 10 tasks per suite, `CORL_CACHE_TEXT=1 CORL_BATCH_VIEWS=1 CORL_CUDA_GRAPH=1`,
refresh every chunk (no reuse). Checkpoint: `early_compression_split4_pool2_detail_10k_v1/evaluation_ema.pth`.

| Suite | Early-compression | Dense 216M (`reports/dense_mixed40k_v2/formal_50trials`) |
|---|---:|---:|
| Spatial | 468/500 (93.6%) | 491/500 (98.2%) |
| Object | 487/500 (97.4%) | 496/500 (99.2%) |
| Goal | 483/500 (96.6%) | 475/500 (95.0%) |
| LIBERO-10 | 455/500 (91.0%) | 463/500 (92.6%) |
| **Total** | **1893/2000 (94.65%)** | **1925/2000 (96.25%)** |

Per-chunk model time (synchronized, excludes host preprocessing): p50 2.47-2.50 ms, p95 4.86-4.98 ms.
The dense run used the older eager path on a different GPU; the success comparison is valid, the latency one is not.
