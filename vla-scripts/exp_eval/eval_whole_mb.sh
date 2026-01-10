#!/usr/bin/env bash
set -euo pipefail

# Run a set of LIBERO eval scripts across multiple seeds.
# Usage: bash vla-scripts/exp_eval/eval_whole_mb.sh

SCRIPTS=(
  # "vla-scripts/exp_eval/sft_mb_paral_goal_seed.sh"
  # "vla-scripts/exp_eval/sft_mb_paral_object_seed.sh"
  # "vla-scripts/exp_eval/sft_mb_paral_spatial_seed.sh"
  "vla-scripts/exp_eval/sft_mb_paral_long_seed.sh"
)

# SEEDS=(2)
SEEDS=(0 1 2)

for script in "${SCRIPTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running ${script} with seed ${seed}"
    bash "${script}" "${seed}"
  done
done

