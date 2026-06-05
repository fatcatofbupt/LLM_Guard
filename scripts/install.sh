#!/usr/bin/env bash
# Install Python dependencies for answer_questions.py.
# Activate your venv/conda env before running, or run inside one.
#
# Override the PyTorch CUDA build if needed:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118 ./install.sh   # CUDA 11.8
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu   ./install.sh   # CPU only

set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

python -m pip install --upgrade pip
python -m pip install torch --index-url "$TORCH_INDEX"
python -m pip install "transformers>=4.51.0" tqdm

python - <<'PY'
import torch, transformers
print(f"torch       = {torch.__version__}")
print(f"transformers= {transformers.__version__}")
print(f"cuda.available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda.device  = {torch.cuda.get_device_name(0)}")
PY
