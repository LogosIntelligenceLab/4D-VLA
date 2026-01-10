#!/bin/bash

# 运行第一个脚本
bash shell/exp_shell/pretrain/exp98_droid_finetune_debug.sh


# 等待 1 分钟
sleep 60
find . -type f -name "*states.pt" -exec rm {} \;

# 运行第二个脚本
bash shell/exp_shell/pretrain/exp97_droid_finetune_debug.sh

echo "Both scripts have completed."
