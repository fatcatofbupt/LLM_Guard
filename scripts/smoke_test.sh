#!/bin/bash
# Qwen3Guard 微调流程验证脚本（Smoke Test）
# 先跑 1 个 epoch 验证环境、数据和训练流程是否正常
# 确认无误后，把 --num_train_epochs 改成 3 即可跑完整训练

set -e

cd "$(dirname "$0")"

# 设置 HuggingFace 镜像（避免初始化时连接 HuggingFace 被墙）
export HF_ENDPOINT=https://hf-mirror.com

# 模型已在本地，强制离线避免网络检查
export HF_HUB_OFFLINE=1

mkdir -p finetune_qwen3guard/output/unsloth_lora

python finetune_qwen3guard/scripts/02_train_unsloth.py \
  --model_path models/Qwen/Qwen3Guard-Gen-0.6B \
  --train_file finetune_qwen3guard/data/train.jsonl \
  --val_file finetune_qwen3guard/data/val.jsonl \
  --output_dir finetune_qwen3guard/output/unsloth_lora \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 2048 \
  --save_merged \
  2>&1 | tee finetune_qwen3guard/output/training.log
