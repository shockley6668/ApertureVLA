# ApertureVLA

A 216M-parameter VLA for onboard closed-loop control. Two camera views go through **DINOv3 ViT-B/16**; the instruction goes through **BERT-base**. After ViT layer 4 the patch grid is pooled 2×2, so layers 5–12 run on an 8×8 grid. Each control step is a static graph: instruction cached once per episode, both views in one encoder call, CUDA-graph replay on GPU or one compiled HBM on an RDK S100P.

## Results (this checkout)

| Model | Params | LIBERO avg (2,000 ep.) | RTX 4090, CUDA graph | RDK S100P BPU |
|---|---:|---:|---:|---:|
| Unpooled (same architecture, no pool) | 0.2B | 96.25% | 2.63 ms | 8.72 ms |
| ApertureVLA | 0.2B | 94.65% | **2.25 ms** | **5.64 ms** |
| ApertureVLA + memory GRPO | 0.2B | 95.55% | — | — |

S100P numbers are `fast-perf` board measurements (latency only). Default PTQ vs float action is **not** a usable accuracy result yet.

Sources: `reports/formal_early_always/`, `reports/grpo_round1/`, `reports/edge/bpu_v370_fastperf/`.

## Layout

```
aperture_vla/          pooling, static-graph inference, episode memory, GRPO
scripts/               train, evaluate, profile, ONNX / PTQ export
tests/                 unit tests that do not need GPU weights
figures/               architecture, pipeline, board, LIBERO rollout
reports/               LIBERO / latency / board JSON (no checkpoints)
third_party/TurboVLA   pinned submodule
third_party/LIBERO     pinned submodule
```

Weights, LIBERO data, and encoder snapshots stay out of git (`assets/`, `data/`, `outputs/`).

## Setup

```bash
git clone --recurse-submodules https://github.com/shockley6668/ApertureVLA.git
cd ApertureVLA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-baseline.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
export PYTHONPATH="$PWD/third_party/TurboVLA:$PYTHONPATH"
```

Pinned upstream:

- TurboVLA `a3365e8f1787d30bafd99d2806edbf4dc82b5370`
- LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`

Place DINOv3 ViT-B/16 and BERT-base under `assets/models/dinov3-vitb16` and `assets/models/bert-base-uncased`.

## Reproduce

**Unpooled baseline** — TurboVLA layout, 40k steps on the four LIBERO suites (see `scripts/run_early_compression_pipeline.sh` for the dense checkpoint path).

**ApertureVLA** — insert the pool, then recover:

```bash
python scripts/build_early_compression_checkpoint.py \
  --dense-checkpoint outputs/checkpoints/dense_mixed40k_v2/model_40000.pth \
  --output outputs/checkpoints/aperture_init.pth \
  --report reports/aperture_init.json \
  --split-layer 4 --merge-factor 2

# then scripts/train_compressed.py (10k steps, l1 + distillation) and
# scripts/convert_trained_checkpoint.py
```

**LIBERO** (seed 7, 50 trials × 10 tasks, chunk 12):

```bash
export CORL_CACHE_TEXT=1 CORL_BATCH_VIEWS=1 CORL_CUDA_GRAPH=1
python scripts/evaluate_compressed_libero.py \
  --ckpt_path outputs/checkpoints/early_compression_split4_pool2_detail_10k_v1/evaluation_ema.pth \
  --dinov3_path assets/models/dinov3-vitb16 --bert_path assets/models/bert-base-uncased \
  --task_suite_name libero_spatial --num_trials_per_task 50 --seed 7
```

**Latency**: `scripts/profile_policy_components.py` with text cache, batched views, and `--cuda-graph`.

**GRPO + episode memory**: `scripts/train_grpo.py` / `scripts/run_grpo_round.sh`.

**Edge**: `scripts/export_instruction_onnx.py`, then the OE 3.7.0 Docker image and `hb_compile` for `march nash-m`. Board timing used `hrt_model_exec`. Quantised action accuracy is still open.

```bash
python -m pytest -q tests/test_early_compression.py tests/test_memory_head.py
```

Do not treat this README as OE toolchain documentation; S-series compile flags must be checked against the current official OE docs.
