#!/usr/bin/env bash
# 在远程服务器上安装 Unsloth 微调所需依赖。
# 建议先激活你的 conda 环境，再执行此脚本。
#
# 用法:
#   conda activate your_env
#   bash install_unsloth.sh

set -euo pipefail

echo "[1/4] Upgrading pip ..."
python -m pip install --upgrade pip

echo "[2/4] Installing PyTorch (CUDA 12.1) ..."
# 如果你的服务器 CUDA 版本不同，请修改 index-url
# CUDA 11.8: https://download.pytorch.org/whl/cu118
# CPU only:  https://download.pytorch.org/whl/cpu
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing Unsloth + training deps ..."
python -m pip install -r requirements_unsloth.txt

echo "[4/4] Verifying installation ..."
python - <<'PY'
import torch
import transformers
import unsloth
import trl
import peft
import datasets

print(f"torch        = {torch.__version__}")
print(f"transformers = {transformers.__version__}")
print(f"unsloth      = {unsloth.__version__}")
print(f"trl          = {trl.__version__}")
print(f"peft         = {peft.__version__}")
print(f"datasets     = {datasets.__version__}")
print(f"CUDA available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device    = {torch.cuda.get_device_name(0)}")
PY

echo "[done] Installation complete."
