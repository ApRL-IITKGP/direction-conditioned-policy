#!/usr/bin/env bash
# Train DCP and CRL on open-room puck pushing (10M steps, 20 evaluations,
# 3 seeds). Runs are written to runs/run_<agent>_push_bot_open_s_<seed>/.
#
# Usage: bash scripts/controlled/train_push_bot.sh
#   SEEDS="1" AGENTS="dcp" LOG_WANDB=1 bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=${SEEDS:-"1 42 123"}
AGENTS=${AGENTS:-"dcp crl"}
if [[ "${LOG_WANDB:-0}" == 1 ]]; then wandb=--log-wandb; else wandb=--no-log-wandb; fi

for seed in ${SEEDS}; do
  for agent in ${AGENTS}; do
    name="${agent}_push_bot_open"
    run_dir="runs/run_${name}_s_${seed}"
    if [[ -f "${run_dir}/ckpt/final" ]]; then
      echo "skip ${run_dir} (complete)"
      continue
    fi
    mkdir -p "${run_dir}"
    echo "train ${run_dir}"
    MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python run.py "${agent}" \
      --env push_bot_open \
      --total-env-steps 10000000 \
      --num-envs 512 \
      --num-evals 20 \
      --episode-length 1000 \
      --batch-size 256 \
      --discounting 0.99 \
      --unroll-length 62 \
      --min-replay-size 1000 \
      --max-replay-size 10000 \
      --contrastive-loss-fn bwd_infonce \
      --energy-fn norm \
      --train-step-multiplier 1 \
      "${wandb}" \
      --seed "${seed}" \
      --exp-name "${name}" \
      2>&1 | tee "${run_dir}/train.log"
  done
done
