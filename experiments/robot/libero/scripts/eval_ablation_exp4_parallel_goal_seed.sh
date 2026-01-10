#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${1:-}"
if [[ -z "${SEED}" ]]; then
  echo "Usage: $0 <SEED>" 1>&2
  echo "" 1>&2
  echo "Env overrides:" 1>&2
  echo "  PRETRAINED_CHECKPOINT=/path/to/checkpoint" 1>&2
  echo "  GPUS=0,1,2  (default: 0,1,2)" 1>&2
  echo "  TASK_IDS=0,1,2,3,4,5,6,7,8,9 (default: all 10 goal tasks)" 1>&2
  echo "  LOG_DIR=experiments/logs/libero_goal" 1>&2
  exit 2
fi

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
if [[ -z "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "[ERROR] Please set PRETRAINED_CHECKPOINT to a local checkpoint directory." 1>&2
  exit 2
fi
if [[ ! -e "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "[ERROR] PRETRAINED_CHECKPOINT not found: ${PRETRAINED_CHECKPOINT}" 1>&2
  exit 2
fi

GPUS_CSV="${GPUS:-0,1,2}"
TASK_IDS_CSV="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
LOG_DIR="${LOG_DIR:-experiments/logs/libero_goal}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_LIST <<<"${GPUS_CSV}"
IFS=',' read -r -a TASK_LIST <<<"${TASK_IDS_CSV}"

if [[ "${#GPU_LIST[@]}" -le 0 ]]; then
  echo "[ERROR] GPUS list is empty." 1>&2
  exit 2
fi
if [[ "${#TASK_LIST[@]}" -le 0 ]]; then
  echo "[ERROR] TASK_IDS list is empty." 1>&2
  exit 2
fi

COMMON_ARGS=(
  --model_family internvl
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --task_suite_name libero_goal
  --num_trials_per_task 50
  --num_task_for_eval 1
  --center_crop False
  --use_nromaction False
  --use_anyview False
  --use_2view False
  --use_multi_size True
  --use_hist True
  --hist_version v2
  --hist_len 10
  --sample_interval 1
  --use_image_prompt False
  --use_depth True
  --use_proprio True
  --is_ft True
  --seed "${SEED}"
  --prompt_version v1
)

for idx in "${!TASK_LIST[@]}"; do
  task_id="${TASK_LIST[$idx]}"
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  exp_name="finetune_goal_task${task_id}_seed${SEED}"
  log_path="${LOG_DIR}/${exp_name}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiments/robot/libero/run_libero_eval_ablation.py \
    "${COMMON_ARGS[@]}" \
    --task_id "${task_id}" \
    --exp_name "${exp_name}" \
    >"${log_path}" 2>&1 &
done

wait
