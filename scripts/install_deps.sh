#!/bin/bash
# Install all Python dependencies for LLM_Guard
# Usage: ./install_deps.sh

MIRROR="https://mirrors.cloud.tencent.com/pypi/simple"

echo "=== Installing LLM_Guard dependencies ==="
pip install -r requirements.txt -i "$MIRROR"

echo ""
echo "=== Done ==="
echo ""
echo "=== Installed packages ==="
pip list | grep -E "transformers|torch|tqdm|datasets|peft|trl|unsloth|fastapi|uvicorn|openai|loguru"
