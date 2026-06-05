#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Qwen3Guard-8B Batch Inference ==="
echo "Start: $(date)"
python3 answer_questions.py \
    --variant 8B \
    --batch-size 8 \
    --max-new-tokens 64
echo "End: $(date)"
