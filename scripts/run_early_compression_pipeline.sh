#!/usr/bin/env bash
# Build, speed-gate, and queue the first 2k early-compression recovery run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/data outputs/checkpoints outputs/libero reports
exec 9>outputs/data/early_compression_pipeline.lock
flock -n 9 || { echo 'An early-compression pipeline is already running' >&2; exit 2; }
trap 'rc=$?; printf "%s\n" "$rc" >"outputs/data/${CANDIDATE_RUN}_exit_code.txt"' EXIT
if [[ -f .env ]]; then source .env; fi
export PYTHONPATH="$PWD/third_party/TurboVLA${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1

: "${DENSE_CHECKPOINT:=outputs/checkpoints/dense_mixed40k_v2/model_40000.pth}"
: "${CANDIDATE_RUN:=early_compression_split4_pool2_detail_v1}"
: "${TRAIN_STEPS:=2000}"
: "${PERFORMANCE_INPUTS:=outputs/performance/libero_balanced80_v1.pt}"
GATE_ID="${GATE_ID:-early_compression_gate_v1}"
[[ "$CANDIDATE_RUN" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
[[ ! -e "outputs/checkpoints/$CANDIDATE_RUN" && ! -e "outputs/libero/$CANDIDATE_RUN" ]] || {
  echo 'Use a fresh run name' >&2; exit 2
}
[[ -f "$DENSE_CHECKPOINT" ]] || { echo "Missing dense checkpoint: $DENSE_CHECKPOINT" >&2; exit 2; }
[[ -f "$PERFORMANCE_INPUTS" ]] || { echo "Missing performance inputs: $PERFORMANCE_INPUTS" >&2; exit 2; }

printf 'unit_tests\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
.venv/bin/python -m pytest -q tests/test_early_compression.py

candidate="outputs/checkpoints/early_compression_split4_pool2_detail_v1.pth"
if [[ ! -f "$candidate" ]]; then
  printf 'building_candidate\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
  .venv/bin/python scripts/build_early_compression_checkpoint.py \
    --dense-checkpoint "$DENSE_CHECKPOINT" \
    --output "$candidate" \
    --report "reports/early_compression_candidate_v1.json" \
    --split-layer 4 --merge-factor 2
fi

printf 'waiting_for_gpu\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
while true; do
  memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "$memory" =~ ^[0-9]+$ ]] && (( memory < 2000 )); then
    if .venv/bin/python - <<'PY'
import subprocess
import sys
import xml.etree.ElementTree as ET
xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
busy = any(p.findtext('type', '').strip() == 'C'
           for p in ET.fromstring(xml).findall('.//processes/process_info'))
sys.exit(int(busy))
PY
    then
      break
    fi
  fi
  sleep 60
done

printf 'speed_gate\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
dense_perf="outputs/performance/${CANDIDATE_RUN}_dense_gpu.json"
early_perf="outputs/performance/${CANDIDATE_RUN}_early_gpu.json"
if [[ ! -f "$dense_perf" ]]; then
  .venv/bin/python scripts/benchmark_policy.py \
    --kind dense --checkpoint "$DENSE_CHECKPOINT" \
    --inputs "$PERFORMANCE_INPUTS" --output "$dense_perf" \
    --device cuda --precision bf16 --warmup 80 --measurements 400
fi
if [[ ! -f "$early_perf" ]]; then
  .venv/bin/python scripts/benchmark_policy.py \
    --kind early --checkpoint "$candidate" \
    --inputs "$PERFORMANCE_INPUTS" --output "$early_perf" \
    --device cuda --precision bf16 --warmup 80 --measurements 400
fi
.venv/bin/python scripts/check_early_speed_gate.py \
  --dense-report "$dense_perf" --early-report "$early_perf" \
  --output "outputs/performance/${CANDIDATE_RUN}_speed_gate.json" \
  --metric host_rgb_to_env_action_chunk --minimum-improvement 0.10

printf 'training\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
mkdir -p "outputs/checkpoints/$CANDIDATE_RUN" "outputs/libero/$CANDIDATE_RUN"
git rev-parse HEAD >"outputs/libero/$CANDIDATE_RUN/code_commit.txt"
git submodule status >"outputs/libero/$CANDIDATE_RUN/upstream_commits.txt"
sha256sum "$DENSE_CHECKPOINT" "$candidate" >"outputs/libero/$CANDIDATE_RUN/initial_checkpoint_sha256.txt"

printf 'waiting_for_exclusive_gpu\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
until .venv/bin/python - <<'PY'
import subprocess
from aperture_vla.performance import parse_exclusive_compute_processes
xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True)
raise SystemExit(0 if not parse_exclusive_compute_processes(xml) else 1)
PY
do
  sleep 60
done

printf 'speed_gate_benchmarks\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
benchmark_inputs=outputs/performance/libero_balanced80_v1.pt
dense_benchmark="outputs/performance/${GATE_ID}_dense.json"
early_benchmark="outputs/performance/${GATE_ID}_early.json"
.venv/bin/python scripts/benchmark_policy.py --kind dense \
  --checkpoint "$DENSE_CHECKPOINT" --inputs "$benchmark_inputs" \
  --output "$dense_benchmark"
.venv/bin/python scripts/benchmark_policy.py --kind early \
  --checkpoint "$candidate" --inputs "$benchmark_inputs" \
  --output "$early_benchmark"

printf 'speed_gate_decision\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
.venv/bin/python scripts/check_early_speed_gate.py \
  --dense-report "$dense_benchmark" --early-report "$early_benchmark" \
  --output "outputs/data/${GATE_ID}.json" --minimum-improvement 0.10

printf 'training\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
exec 8>outputs/data/corrected_mainline.lock
flock 8
.venv/bin/python scripts/train_compressed.py \
  --candidate-checkpoint "$candidate" \
  --teacher-checkpoint "$DENSE_CHECKPOINT" \
  --distillation-weight 1.0 --teacher-micro-batch 1 \
  --teacher-attention-stability per_sample \
  --dataset_dirs data/libero/libero_spatial_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_10_no_noops/1.0.0 \
  --stats_path third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops --normalize_binary_gripper false \
  --dinov3_path assets/models/dinov3-vitb16 --bert_path assets/models/bert-base-uncased \
  --checkpoint_dir "outputs/checkpoints/$CANDIDATE_RUN" --checkpoint_prefix model \
  --max_steps "$TRAIN_STEPS" --lr_schedule_steps "$TRAIN_STEPS" --warmup_steps 100 \
  --min_lr_ratio 0.1 --batch_size 8 --grad_accum_steps 4 --num_workers 1 \
  --shuffle_buffer 8 --step_mix_buffer_size 64 --precision bf16_amp \
  --dinov3_lr 0.00001 --head_lr 0.00005 \
  --save_steps 500 --seed 42 --resume_mode none

printf 'converting_checkpoint\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
.venv/bin/python scripts/convert_trained_checkpoint.py \
  --trained-checkpoint "outputs/checkpoints/$CANDIDATE_RUN/model_${TRAIN_STEPS}.pth" \
  --compressed-template "$candidate" \
  --output "outputs/checkpoints/$CANDIDATE_RUN/evaluation_ema.pth" \
  --report "outputs/libero/$CANDIDATE_RUN/checkpoint_conversion.json" --state-source ema

printf 'four_suite_smoke\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
.venv/bin/python scripts/setup_libero_config.py >/dev/null
for suite in libero_spatial libero_object libero_goal libero_10; do
  .venv/bin/python scripts/evaluate_compressed_libero.py \
    --ckpt_path "outputs/checkpoints/$CANDIDATE_RUN/evaluation_ema.pth" \
    --dinov3_path assets/models/dinov3-vitb16 --bert_path assets/models/bert-base-uncased \
    --stats_path third_party/TurboVLA/experiments/libero/configs/libero_all4_stats.json \
    --stats_key libero_all4_no_noops --task_suite_name "$suite" --num_trials_per_task 5 \
    --seed 7 --libero_root third_party/LIBERO --mujoco_gl egl --pyopengl_platform egl \
    --precision bf16 --result_json_path "outputs/libero/$CANDIDATE_RUN/${suite}_10tasks_5trials.json"
done
.venv/bin/python scripts/aggregate_results.py "outputs/libero/$CANDIDATE_RUN"/*_10tasks_5trials.json \
  --output "outputs/libero/$CANDIDATE_RUN/summary.json"
flock -u 8

printf 'trained_model_benchmark\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
.venv/bin/python scripts/benchmark_policy.py --kind early \
  --checkpoint "outputs/checkpoints/$CANDIDATE_RUN/evaluation_ema.pth" \
  --inputs "$benchmark_inputs" --output "outputs/performance/${GATE_ID}_trained_early.json"
printf 'completed\n' >"outputs/data/${CANDIDATE_RUN}_stage.txt"
