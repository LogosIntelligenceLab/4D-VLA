#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${1:-}"
if [[ -z "${SEED}" ]]; then
  echo "Usage: $0 <SEED>" 1>&2
  exit 2
fi

# 3800, 7600, 11400, 15200
GPUS=(0 1 2)
TASK_IDS=(0 1 2)
# PRETRAINED_CHECKPOINT="internvl_chat/work_dirs/ablaion_exp4_spatial_10hist_1img_1sample_interval_ah_p1_3d_prop_pretrain_fb_1epoch_nodrop/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full/checkpoint-5670"
# PRETRAINED_CHECKPOINT="internvl_chat/work_dirs_ssd_group/ablaion_exp4_spatial_10hist_1img_1sample_interval_ah_p1_3d_prop_pretrain/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full/checkpoint-7860"
# PRETRAINED_CHECKPOINT="internvl_chat/work_dirs/ablaion_exp4_spatial_10hist_1img_1sample_interval_ah_p1_3d_prop_pretrain_fb_1epoch_drop/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full/checkpoint-7860"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-internvl_chat/work_dirs/ablaion_exp4_spatial_10hist_1img_1sample_interval_ah_p1_3d_prop_pretrain_fb_1epoch_drop_40ep/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full/checkpoint-15720}"

if [[ -z "${PRETRAINED_CHECKPOINT}" || ! -e "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "[ERROR] PRETRAINED_CHECKPOINT not found: ${PRETRAINED_CHECKPOINT}" 1>&2
  echo "Fix: export PRETRAINED_CHECKPOINT=/path/to/checkpoint then rerun." 1>&2
  exit 2
fi

for i in ${!GPUS[@]}; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python experiments/robot/libero/run_libero_eval_ablation.py \
    --model_family internvl \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --num_task_for_eval 1 \
    --task_id ${TASK_IDS[$i]} \
    --center_crop False \
    --use_nromaction False \
    --use_anyview False \
    --use_2view False \
    --use_multi_size True \
    --use_hist True \
    --hist_version 'v2' \
    --hist_len 10 \
    --sample_interval 1 \
    --use_image_prompt False \
    --use_depth True \
    --use_proprio True \
    --is_ft True \
    --seed "${SEED}" \
    --exp_name "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp" \
    --prompt_version 'v1' \
    > "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp.log" 2>&1 &
done

wait


GPUS=(0 1 2)
TASK_IDS=(3 4 5)

# Iterate over the GPUS and POSES arrays
for i in ${!GPUS[@]}; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python experiments/robot/libero/run_libero_eval_ablation.py \
    --model_family internvl \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --num_task_for_eval 1 \
    --task_id ${TASK_IDS[$i]} \
    --center_crop False \
    --use_nromaction False \
    --use_anyview False \
    --use_2view False \
    --use_multi_size True \
    --use_hist True \
    --hist_version 'v2' \
    --hist_len 10 \
    --sample_interval 1 \
    --use_image_prompt False \
    --use_depth True \
    --use_proprio True \
    --is_ft True \
    --seed "${SEED}" \
    --exp_name "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp" \
    --prompt_version 'v1' \
    > "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp.log" 2>&1 &
done

wait


GPUS=(3 4 5)
TASK_IDS=(6 7 8)

for i in ${!GPUS[@]}; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python experiments/robot/libero/run_libero_eval_ablation.py \
    --model_family internvl \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --num_task_for_eval 1 \
    --task_id ${TASK_IDS[$i]} \
    --center_crop False \
    --use_nromaction False \
    --use_anyview False \
    --use_2view False \
    --use_multi_size True \
    --use_hist True \
    --hist_version 'v2' \
    --hist_len 10 \
    --sample_interval 1 \
    --use_image_prompt False \
    --use_depth True \
    --use_proprio True \
    --is_ft True \
    --seed "${SEED}" \
    --exp_name "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp" \
    --prompt_version 'v1' \
    > "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp.log" 2>&1 &
done

wait

GPUS=(6)
TASK_IDS=(9)

for i in ${!GPUS[@]}; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python experiments/robot/libero/run_libero_eval_ablation.py \
    --model_family internvl \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --num_task_for_eval 1 \
    --task_id ${TASK_IDS[$i]} \
    --center_crop False \
    --use_nromaction False \
    --use_anyview False \
    --use_2view False \
    --use_multi_size True \
    --use_hist True \
    --hist_version 'v2' \
    --hist_len 10 \
    --sample_interval 1 \
    --use_image_prompt False \
    --use_depth True \
    --use_proprio True \
    --is_ft True \
    --seed "${SEED}" \
    --exp_name "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp" \
    --prompt_version 'v1' \
    > "finetune_spatial_task${TASK_IDS[$i]}_40ep_seed${SEED}_1epoch_fb_dp.log" 2>&1 &
done

wait
