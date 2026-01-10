#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-34229}"

# This script mirrors:
#   shell/legacy/exp_shell/franka/franka_new_bs32_20ep_baseline_pt_mb_aug_dep_50samples_v5.sh
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"

if (( GLOBAL_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * GPUS) != 0 )); then
  echo "[ERROR] GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by PER_DEVICE_BATCH_SIZE*GPUS ($((PER_DEVICE_BATCH_SIZE * GPUS)))." 1>&2
  exit 2
fi
GRADIENT_ACC=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export LAUNCHER="${LAUNCHER:-pytorch}"

# Backward-compat env aliases used in some legacy scripts.
OUTPUT_DIR="${OUTPUT_DIR:-${output_dir:-work_dirs/4dvla/finetune_franka}}"
META_PATH="${META_PATH:-${meta_path:-}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero_stage1_config.json}"

# Point this to a pretrain checkpoint directory (or a HuggingFace-style local model dir).
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] Please set MODEL_NAME_OR_PATH (e.g. work_dirs/.../checkpoint-XXXXX or a local HF model dir)." 1>&2
  exit 2
fi
if [[ "${MODEL_NAME_OR_PATH}" == /* || "${MODEL_NAME_OR_PATH}" == ./* || "${MODEL_NAME_OR_PATH}" == ../* ]]; then
  if [[ ! -e "${MODEL_NAME_OR_PATH}" ]]; then
    echo "[ERROR] MODEL_NAME_OR_PATH not found: ${MODEL_NAME_OR_PATH}" 1>&2
    exit 2
  fi
fi
if [[ -z "${META_PATH}" ]]; then
  echo "[ERROR] Please set META_PATH (dataset mixture json). Example: META_PATH=shell/exp_data/xxx.json" 1>&2
  exit 2
fi
if [[ ! -e "${META_PATH}" ]]; then
  echo "[ERROR] META_PATH not found: ${META_PATH}" 1>&2
  exit 2
fi

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-40}"
HIST_LEN="${HIST_LEN:-4}"
SAVE_STEPS="${SAVE_STEPS:-1500}"
USE_DYNAMIC_IMAGE_SIZE="${USE_DYNAMIC_IMAGE_SIZE:-False}"
USE_THUMBNAIL="${USE_THUMBNAIL:-False}"

mkdir -p "${OUTPUT_DIR}"

# Basic GPU sanity-check (fail fast with a clear message).
AVAILABLE_GPUS="$(
  python - <<'PY'
import torch
try:
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ "${AVAILABLE_GPUS}" -le 0 ]]; then
  echo "[ERROR] torch reports 0 visible CUDA devices. Run on a GPU node / fix CUDA_VISIBLE_DEVICES." 1>&2
  exit 2
fi
if (( GPUS > AVAILABLE_GPUS )); then
  echo "[ERROR] GPUS (${GPUS}) > visible CUDA devices (${AVAILABLE_GPUS}). Set GPUS accordingly." 1>&2
  exit 2
fi

# Quick dataset sanity-check (fast-fail for common path mistakes).
META_PATH="${META_PATH}" HIST_LEN="${HIST_LEN}" python - <<'PY'
import json
import os
import sys

meta_path = os.environ.get("META_PATH")
hist_len = int(os.environ.get("HIST_LEN", "4"))
expected = hist_len + 1

meta = json.load(open(meta_path, "r"))
if not meta:
    print(f"[ERROR] META_PATH is empty: {meta_path}", file=sys.stderr)
    sys.exit(2)

first_name = next(iter(meta.keys()))
ann = meta[first_name].get("annotation")
if not ann:
    print(f"[ERROR] dataset '{first_name}' missing 'annotation' in {meta_path}", file=sys.stderr)
    sys.exit(2)
if not os.path.exists(ann):
    print(f"[ERROR] annotation not found: {ann}", file=sys.stderr)
    sys.exit(2)

with open(ann, "r") as f:
    line = f.readline().strip()
if not line:
    print(f"[ERROR] annotation is empty: {ann}", file=sys.stderr)
    sys.exit(2)

obj = json.loads(line)
images = obj.get("image")
depths = obj.get("image_depth")

if isinstance(images, list):
    if len(images) != expected:
        print(
            f"[WARN] first sample has {len(images)} frames, expected {expected} for hist_len={hist_len}.",
            file=sys.stderr,
        )
else:
    print("[WARN] first sample 'image' is not a list; this script assumes list-based multi-frame jsonl.", file=sys.stderr)

if isinstance(depths, list) and isinstance(images, list) and len(depths) != len(images):
    print(
        f"[WARN] first sample depth/image count mismatch: len(image_depth)={len(depths)} len(image)={len(images)}",
        file=sys.stderr,
    )
PY

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  internvl/train/finetune_franka_baseline_pt_mb_aug_dep.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --conv_style "phi3-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --force_hist_image_size 224 \
  --hist_len "${HIST_LEN}" \
  --use_multi_size_image True \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 20 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size "${USE_DYNAMIC_IMAGE_SIZE}" \
  --use_thumbnail "${USE_THUMBNAIL}" \
  --ps_version "v2" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "tensorboard"
