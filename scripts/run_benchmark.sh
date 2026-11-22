#!/usr/bin/env bash
# Train one method on the nine benchmark tasks with the five paper seeds.
#
#   bash scripts/run_benchmark.sh --method dcp
#   bash scripts/run_benchmark.sh --method direct_dcp --env pusher_hard --seed 42
#   bash scripts/run_benchmark.sh --method crl --dry-run
#
# Methods: dcp, direct_dcp, crl, ssgc, crl_wide (CRL with the actor's first layer
# widened to 313 units, matching DCP's actor parameter count).
# Runs are written to runs/run_<method>_<env>_s<seed>_s_<seed>/. Completed runs are skipped.
# Add --log-wandb to log to Weights & Biases.
set -euo pipefail
cd "$(dirname "$0")/.."

method=""
env_filter=all
seed_filter=all
dry_run=false
extra=()
while (($#)); do
  case "$1" in
    --method) method=$2; shift 2 ;;
    --env) env_filter=$2; shift 2 ;;
    --seed) seed_filter=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --log-wandb) extra+=(--log-wandb); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$method" in
  dcp) agent=(dcp) ;;
  direct_dcp) agent=(dcp --use-direct-goal-direction) ;;
  crl) agent=(crl) ;;
  ssgc) agent=(ssgc) ;;
  crl_wide) agent=(crl --actor-first-layer-width 313) ;;
  *) echo "Specify --method dcp|direct_dcp|crl|ssgc|crl_wide" >&2; exit 2 ;;
esac

envs=(ant_big_maze ant_hardest_maze ant_u_maze humanoid_u_maze
      ant_push ant_ball pusher_easy pusher_hard pusher_hard_far)
seeds=(1 42 123 256 1234)

for env in "${envs[@]}"; do
  [[ $env_filter == all || $env_filter == "$env" ]] || continue
  steps=50000000; evals=50; num_envs=512; batch_size=256
  case "$env" in
    humanoid_u_maze) steps=100000000; evals=100; num_envs=128; batch_size=128 ;;
    ant_push|pusher_hard) steps=100000000; evals=100 ;;
  esac

  for seed in "${seeds[@]}"; do
    [[ $seed_filter == all || $seed_filter == "$seed" ]] || continue
    exp_name="${method}_${env}_s${seed}"
    run_dir="runs/run_${exp_name}_s_${seed}"
    if [[ -f "${run_dir}/ckpt/final" ]]; then
      echo "Skipping completed run: $run_dir"
      continue
    fi

    cmd=(uv run python run.py "${agent[@]}"
      --env "$env" --seed "$seed"
      --total-env-steps "$steps" --num-evals "$evals"
      --num-envs "$num_envs" --batch-size "$batch_size"
      --episode-length 1000 --action-repeat 1 --discounting 0.99
      --unroll-length 62 --min-replay-size 1000 --max-replay-size 10000
      --contrastive-loss-fn bwd_infonce --energy-fn norm
      --train-step-multiplier 1 --exp-name "$exp_name" "${extra[@]}")

    printf '%q ' "${cmd[@]}"; echo
    [[ $dry_run == true ]] || MUJOCO_GL=egl "${cmd[@]}"
  done
done
