set -x

GPUS=8
BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=16
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/ablaion_exp4_goal_10hist_1img_1sample_interval_ah_p1_3d_prop_pretrain_fb_1epoch_drop/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/legacy/ablation_exp4_1img_10hist_3d_prop_pretrain_fixbug.py \
  --model_name_or_path "work_dirs_ssd_group/exp98_droid_pretrain_5hist/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full/checkpoint-17000"  \
  --conv_style "phi3-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "shell/exp_data/ablaion_exp2_goal_10hist_1img.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --force_hist_image_size 224 \
  --hist_len 9 \
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
  --num_train_epochs 20 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1890 \
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
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  # 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
