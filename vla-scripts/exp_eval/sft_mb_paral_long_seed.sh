#!/usr/bin/env bash
set -euo pipefail

# Parallel LIBERO-10 eval for MB-SFT checkpoints.
# Usage: bash vla-scripts/exp_eval/sft_mb_paral_long_seed.sh <SEED>
#
# Override checkpoint by exporting PRETRAINED_CHECKPOINT=/path/to/ckpt

SEED="${1:-0}"

# Split tasks across GPUs in groups, each group runs in parallel.
GPUS=(0 1 2)
TASK_IDS=(0 1 2)

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-internvl_chat/work_dirs/ablaion_mb_long_new_pretrain_old_te_fb/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full}"

EXP_NAME_SUFFIX="${EXP_NAME_SUFFIX:-finetune_long_20ep_seed${SEED}_1epoch_mb_new_pretrain_old_te_fb}"

run_group () {
  local -a gpus=("${!1}")
  local -a task_ids=("${!2}")
  for i in "${!gpus[@]}"; do
    local gpu="${gpus[$i]}"
    local task_id="${task_ids[$i]}"
    local exp_name="finetune_long_task${task_id}_${EXP_NAME_SUFFIX}"
    local log_file="${exp_name}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" python experiments/robot/libero/run_libero_eval_ablation_mb_sft.py \
      --model_family internvl \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      --task_suite_name libero_10 \
      --num_trials_per_task 50 \
      --num_task_for_eval 1 \
      --task_id "${task_id}" \
      --center_crop False \
      --use_nromaction False \
      --use_anyview False \
      --use_2view False \
      --use_multi_size True \
      --use_hist True \
      --hist_version v2 \
      --hist_len 5 \
      --sample_interval 1 \
      --use_image_prompt False \
      --use_depth True \
      --use_proprio False \
      --is_ft True \
      --seed "${SEED}" \
      --exp_name "${exp_name}" \
      --prompt_version v1 \
      > "${log_file}" 2>&1 &
  done
  wait
}

run_group GPUS[@] TASK_IDS[@]

GPUS=(0 1 2)
TASK_IDS=(3 4 5)
run_group GPUS[@] TASK_IDS[@]

GPUS=(3 4 5)
TASK_IDS=(6 7 8)
run_group GPUS[@] TASK_IDS[@]

GPUS=(6)
TASK_IDS=(9)
run_group GPUS[@] TASK_IDS[@]

