#!/usr/bin/env bash
# =============================================================================
# Qwen3Guard 远程 GPU 服务器环境准备脚本（当前活跃方案：peft + trl LoRA）
# 适用于：中国大陆服务器、NVIDIA GPU、conda
#
# 用法：
#   bash finetune_qwen3guard/setup_remote.sh
#
# 可用环境变量覆盖（均有默认值）：
#   CONDA_ENV        conda 环境名，默认 py311；环境不存在时自动创建
#   GPU_ID           训练用 GPU 编号，默认 0
#   TORCH_INDEX      PyTorch 安装源，默认 cu121
#   PROJECT_DIR      项目根目录，默认取脚本所在仓库根
#   训练超参：NUM_TRAIN_EPOCHS / LEARNING_RATE / LORA_R / LORA_ALPHA /
#             BATCH_SIZE / GRAD_ACCUM
# =============================================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# 配置（可用环境变量覆盖）
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-py311}"
GPU_ID="${GPU_ID:-0}"

MODEL_DIR="${PROJECT_DIR}/models/Qwen/Qwen3Guard-Gen-0.6B"
OUTPUT_DIR="${PROJECT_DIR}/finetune_qwen3guard/output/lora_v5_1"

# 训练超参
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# 国内 PyPI 镜像（腾讯云源）
PIP_INDEX="https://mirrors.cloud.tencent.com/pypi/simple/"
PIP_TRUSTED="mirrors.cloud.tencent.com"

# PyTorch 安装源（按服务器 CUDA 版本覆盖，如 TORCH_INDEX=https://download.pytorch.org/whl/cu118）
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

# 国内 ModelScope 缓存目录
export MODELSCOPE_CACHE="${PROJECT_DIR}/.modelscope_cache"

log_info "========================================"
log_info "Qwen3Guard 远程环境准备（peft + trl LoRA）"
log_info "========================================"
log_info "项目目录:   ${PROJECT_DIR}"
log_info "Conda 环境: ${CONDA_ENV}"
log_info "GPU 编号:   ${GPU_ID}"
log_info "PyPI 镜像:  ${PIP_INDEX}"
log_info "Torch 源:   ${TORCH_INDEX}"

# ---------------------------------------------------------------------------
# Step 0: 检查环境
# ---------------------------------------------------------------------------
log_info "Step 0/5: 检查环境..."

if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi 未找到，请确认 NVIDIA 驱动已安装"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_info "GPU 信息（共 ${GPU_COUNT} 张）:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if [ "${GPU_ID}" -ge "${GPU_COUNT}" ]; then
    log_error "GPU_ID=${GPU_ID} 超出可用范围（共 ${GPU_COUNT} 张，编号 0-$((GPU_COUNT - 1))）"
    exit 1
fi

# 探测 conda 安装目录
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "${CONDA_BASE}" ]; then
    for cand in "${HOME}/miniconda3" "${HOME}/miniconda" "${HOME}/anaconda3" "/opt/conda"; do
        if [ -x "${cand}/bin/conda" ]; then
            CONDA_BASE="${cand}"
            break
        fi
    done
fi

if [ -z "${CONDA_BASE}" ] || [ ! -x "${CONDA_BASE}/bin/conda" ]; then
    log_error "未找到 conda。请安装 miniconda，或用 CONDA_BASE=/path/to/conda 指定"
    exit 1
fi

CONDA_BIN="${CONDA_BASE}/bin/conda"
log_info "Conda 路径: ${CONDA_BIN}"

# 环境不存在则自动创建
if ! "${CONDA_BIN}" env list | grep -qE "^${CONDA_ENV}\s"; then
    log_warn "Conda 环境 '${CONDA_ENV}' 不存在，自动创建（Python 3.11）..."
    "${CONDA_BIN}" create -n "${CONDA_ENV}" python=3.11 -y
fi

CONDA_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
CONDA_PIP="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"

log_info "Python 路径: ${CONDA_PYTHON}"
"${CONDA_PYTHON}" --version

# ---------------------------------------------------------------------------
# Step 1: 安装依赖
# ---------------------------------------------------------------------------
log_info "Step 1/5: 安装 Python 依赖..."

export PIP_DEFAULT_TIMEOUT=300

"${CONDA_PIP}" install --upgrade pip -i "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"

log_info "[1/4] 安装 PyTorch（${TORCH_INDEX}）..."
"${CONDA_PIP}" install torch --index-url "${TORCH_INDEX}"

log_info "[2/4] 安装训练依赖（peft / trl / accelerate / datasets / bitsandbytes）..."
"${CONDA_PIP}" install \
    "transformers>=4.51.0" \
    peft \
    trl \
    accelerate \
    datasets \
    bitsandbytes \
    tqdm \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    --timeout 300 \
    --retries 5

log_info "[3/4] 安装 modelscope（用于下载模型）..."
"${CONDA_PIP}" install \
    modelscope \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    --timeout 300 \
    --retries 5

log_info "[4/4] 已安装的关键包:"
"${CONDA_PIP}" list | grep -iE "torch|transformers|trl|peft|accelerate|datasets|bitsandbytes|modelscope" || true

# ---------------------------------------------------------------------------
# Step 2: 验证关键包
# ---------------------------------------------------------------------------
log_info "Step 2/5: 验证关键包..."

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_PYTHON}" -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA 可用:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本:    {torch.version.cuda}')
    print(f'当前 GPU:     {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers: {transformers.__version__}')

for pkg in ('trl', 'peft', 'accelerate', 'datasets', 'modelscope'):
    try:
        __import__(pkg)
        print(f'{pkg:<13} OK')
    except Exception as e:
        print(f'{pkg:<13} 导入失败 - {e}')
"

# ---------------------------------------------------------------------------
# Step 3: 下载模型（ModelScope，权重缺失时）
# ---------------------------------------------------------------------------
if [ -f "${MODEL_DIR}/model.safetensors" ] || [ -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    log_info "Step 3/5: 模型权重已存在，跳过下载"
else
    log_info "Step 3/5: 下载 Qwen3Guard-Gen-0.6B 模型（ModelScope）..."

    mkdir -p "${PROJECT_DIR}/models/Qwen"

    "${CONDA_PYTHON}" -c "
from modelscope import snapshot_download

model_id = 'Qwen/Qwen3Guard-Gen-0.6B'
local_dir = '${MODEL_DIR}'

print(f'正在从 ModelScope 下载: {model_id}')
print(f'目标目录: {local_dir}')

snapshot_download(
    model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print('模型下载完成')
"
fi

# 验证模型文件
log_info "验证模型文件..."
if [ -f "${MODEL_DIR}/model.safetensors" ] || [ -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    log_info "模型权重文件存在 ✓"
else
    log_error "未找到模型权重文件，下载不完整"
    exit 1
fi

for f in config.json tokenizer_config.json tokenizer.json; do
    if [ -f "${MODEL_DIR}/${f}" ]; then
        log_info "${f} 存在 ✓"
    else
        log_warn "${f} 缺失"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: 执行训练
# ---------------------------------------------------------------------------
log_info "Step 4/5: 开始训练（GPU ${GPU_ID}）..."

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"${CONDA_PYTHON}" finetune_qwen3guard/scripts/02_train_lora.py \
    --model_path "${MODEL_DIR}" \
    --train_file "${PROJECT_DIR}/finetune_qwen3guard/data/train_v5.jsonl" \
    --val_file "${PROJECT_DIR}/finetune_qwen3guard/data/val_v5.jsonl" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_merged

# ---------------------------------------------------------------------------
# Step 5: 验证效果
# ---------------------------------------------------------------------------
log_info "Step 5/5: 验证训练效果（合并模型冒烟测试）..."

cd "${PROJECT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MERGED_MODEL_PATH="${OUTPUT_DIR}/merged_model" \
"${CONDA_PYTHON}" finetune_qwen3guard/scripts/test_merged_model.py

log_info "========================================"
log_info "全部完成！"
log_info "合并模型路径: ${OUTPUT_DIR}/merged_model"
log_info "========================================"
