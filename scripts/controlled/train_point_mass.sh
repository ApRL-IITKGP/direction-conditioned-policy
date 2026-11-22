#!/usr/bin/env bash
# Train CRL, Direct-DCP, SSGC, and DCP in the point-mass open room and U-maze
# (10M steps, 20 evaluations, 5 seeds). Runs are written to
# runs/run_<variant>_<env>_s_<seed>/ with the training log in train.log.
#
# Usage: bash scripts/controlled/train_point_mass.sh
#   SEEDS="42 1" VARIANTS="dcp crl" ENVS="simple_u_maze" LOG_WANDB=1 bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=${SEEDS:-"42 1 123 256 1234"}
ENVS=${ENVS:-"simple_u_maze simple_open"}
VARIANTS=${VARIANTS:-"dcp crl direct_dcp ssgc"}
if [[ "${LOG_WANDB:-0}" == 1 ]]; then wandb=--log-wandb; else wandb=--no-log-wandb; fi

for seed in ${SEEDS}; do
  for env in ${ENVS}; do
    for variant in ${VARIANTS}; do
      case "${variant}" in
        crl) agent=(crl) ;;
        dcp) agent=(dcp) ;;
        direct_dcp) agent=(dcp --use-direct-goal-direction) ;;
        ssgc) agent=(ssgc --waypoint-steps 25 --pool-size 512 --pool-candidates 32
                     --subgoal-scoring dot_product) ;;
        *) echo "unknown variant: ${variant}" >&2; exit 2 ;;
      esac
      name="${variant}_${env}"
      run_dir="runs/run_${name}_s_${seed}"
      if [[ -f "${run_dir}/ckpt/final" ]]; then
        echo "skip ${run_dir} (complete)"
        continue
      fi
      mkdir -p "${run_dir}"
      echo "train ${run_dir}"
      MUJOCO_GL=egl uv run python run.py "${agent[@]}" \
        --env "${env}" \
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
done
