#!/bin/bash
# GRPO post-training round: memory and memoryless runs in sequence, then formal evaluation of both.
# Usage: scripts/run_grpo_round.sh   (single instance enforced by a lock file)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/queue outputs/rl
exec 9>outputs/queue/grpo_round.lock
flock -n 9 || { echo "another GRPO round is running"; exit 1; }
source .env
export PYTHONPATH="$PWD/third_party/TurboVLA:$PWD/third_party/TurboVLA/third_party/vla_adapter:$PWD"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero" USE_TF=0
base=outputs/checkpoints/early_compression_split4_pool2_detail_10k_v1/evaluation_ema.pth
iterations=${GRPO_ITERATIONS:-40}

train() {
  local run=$1; shift
  echo "[$(date +%T)] train $run"
  .venv/bin/python scripts/train_grpo.py "$@" --checkpoint "$base" --output "outputs/rl/$run" \
    --tasks 8,8,8,6,6,0,9,3,2,7 --workers 16 --groups 16 --group-size 8 \
    --iterations "$iterations" --save-every 5 > "outputs/rl/$run.log" 2>&1 \
    || echo "[$(date +%T)] $run FAILED"
}

evaluate() {
  local run=$1 iter=$2; shift 2
  local ckpt="outputs/rl/$run/$iter/evaluation_ema.pth"
  [ -f "$ckpt" ] || { echo "[$(date +%T)] missing $ckpt"; return; }
  for suite in "$@"; do
    CORL_EVAL_CKPT=$ckpt CORL_CACHE_TEXT=1 CORL_BATCH_VIEWS=1 CORL_CUDA_GRAPH=1 CORL_REFRESH_MODE=always \
      scripts/run_refresh_eval.sh "formal_${run}_${iter}" 50 "$suite" > "outputs/queue/formal_${run}_${iter}_$suite.log" 2>&1 &
  done
}

train grpo_mem1 --memory
train grpo1
final=$(printf "iter%04d" "$iterations")
middle=$(printf "iter%04d" $((iterations / 2)))
echo "[$(date +%T)] evaluate"
for run in grpo_mem1 grpo1; do
  evaluate "$run" "$final" libero_spatial libero_object libero_goal libero_10
  evaluate "$run" "$middle" libero_10
done
wait
echo "[$(date +%T)] GRPO_ROUND_DONE"
