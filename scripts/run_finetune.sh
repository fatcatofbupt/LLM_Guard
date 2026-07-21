#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="logs/finetune.log"
echo "=== Fine-tune started at $(date) ===" | tee -a "$LOG"
echo "PID: $$" | tee -a "$LOG"

# Trap signals to find what's killing us
trap 'echo "SIGTERM received at $(date)" >> "$LOG"; exit 143' TERM
trap 'echo "SIGINT received at $(date)" >> "$LOG"; exit 130' INT
trap 'echo "SIGHUP received at $(date)" >> "$LOG"; exit 129' HUP
trap 'echo "SIGQUIT received at $(date)" >> "$LOG"; exit 131' QUIT
trap 'echo "SIGKILL received at $(date)" >> "$LOG"' KILL
trap 'echo "SIGUSR1 received at $(date)" >> "$LOG"' USR1
trap 'echo "SIGUSR2 received at $(date)" >> "$LOG"' USR2

source /data/ai_phone/miniconda/etc/profile.d/conda.sh
conda activate py311
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python3 finetune_qwen3guard/scripts/02_train_lora.py \
    --train_file finetune_qwen3guard/data/train_v5.jsonl \
    --val_file finetune_qwen3guard/data/val_v5.jsonl \
    --num_train_epochs 3 \
    --output_dir finetune_qwen3guard/output/lora_v5_1 \
    --save_merged \
    2>&1 | tee -a "$LOG"

echo "=== Fine-tune finished at $(date) ===" | tee -a "$LOG"
