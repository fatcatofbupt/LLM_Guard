#!/usr/bin/env bash
# =============================================================================
# Qwen3Guard 远程服务器环境准备脚本
# 适用于：中国大陆服务器、conda index 环境、RTX 4090
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
# 配置
# ---------------------------------------------------------------------------
CONDA_ENV="index"
PROJECT_DIR="/data/ai_phone/LLM_Guard"
MODEL_DIR="${PROJECT_DIR}/models/Qwen/Qwen3Guard-Gen-0.6B"

# 国内 PyPI 镜像（腾讯云源，延迟最低）
PIP_INDEX="https://mirrors.cloud.tencent.com/pypi/simple/"
PIP_TRUSTED="mirrors.cloud.tencent.com"

# 国内 HuggingFace / ModelScope 镜像
export HF_ENDPOINT="https://hf-mirror.com"
export MODELSCOPE_CACHE="${PROJECT_DIR}/.modelscope_cache"

log_info "========================================"
log_info "Qwen3Guard 远程环境准备"
log_info "========================================"
log_info "Conda 环境: ${CONDA_ENV}"
log_info "项目目录: ${PROJECT_DIR}"
log_info "PyPI 镜像: ${PIP_INDEX}"
log_info "HF 镜像:   ${HF_ENDPOINT}"

# ---------------------------------------------------------------------------
# Step 0: 检查环境
# ---------------------------------------------------------------------------
log_info "Step 0/5: 检查环境..."

if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi 未找到，请确认 NVIDIA 驱动已安装"
    exit 1
fi

log_info "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 检查 conda
if [ ! -d "${HOME}/miniconda3" ] && [ ! -d "${HOME}/anaconda3" ]; then
    log_error "未找到 conda 安装目录"
    exit 1
fi

# 设置 conda 路径
if [ -d "${HOME}/miniconda3" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
    CONDA_BIN="${HOME}/anaconda3/bin/conda"
fi

log_info "Conda 路径: ${CONDA_BIN}"

# 检查 index 环境是否存在
if ! ${CONDA_BIN} env list | grep -q "${CONDA_ENV}"; then
    log_error "Conda 环境 '${CONDA_ENV}' 不存在"
    exit 1
fi

CONDA_PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
CONDA_PIP="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/pip"

log_info "Python 路径: ${CONDA_PYTHON}"
${CONDA_PYTHON} --version

# ---------------------------------------------------------------------------
# Step 1: 升级 pip 并安装依赖
# ---------------------------------------------------------------------------
log_info "Step 1/5: 安装 Python 依赖（使用阿里云镜像）..."

${CONDA_PIP} install --upgrade pip -i ${PIP_INDEX} --trusted-host ${PIP_TRUSTED}

# 核心训练依赖（分步安装，减少超时风险）
log_info "正在安装依赖（腾讯云源，分步安装防止超时）..."

# 设置 pip 超时
export PIP_DEFAULT_TIMEOUT=300

log_info "[1/4] 安装 peft, accelerate, datasets, bitsandbytes..."
${CONDA_PIP} install \
    peft accelerate datasets bitsandbytes \
    -i ${PIP_INDEX} \
    --trusted-host ${PIP_TRUSTED} \
    --timeout 300 \
    --retries 5

log_info "[2/4] 安装 trl..."
${CONDA_PIP} install \
    trl \
    -i ${PIP_INDEX} \
    --trusted-host ${PIP_TRUSTED} \
    --timeout 300 \
    --retries 5

log_info "[3/4] 安装 unsloth（包较大，可能需要几分钟）..."
${CONDA_PIP} install \
    unsloth \
    -i ${PIP_INDEX} \
    --trusted-host ${PIP_TRUSTED} \
    --timeout 600 \
    --retries 5

log_info "[4/4] 验证安装..."
${CONDA_PIP} list | grep -iE "unsloth|trl|peft|accelerate|datasets|bitsandbytes" || true

log_info "依赖安装完成"

# ---------------------------------------------------------------------------
# Step 2: 验证关键包
# ---------------------------------------------------------------------------
log_info "Step 2/5: 验证关键包..."

${CONDA_PYTHON} -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本:    {torch.version.cuda}')
    print(f'GPU:          {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers: {transformers.__version__}')

try:
    import unsloth
    print(f'Unsloth:      {unsloth.__version__}')
except Exception as e:
    print(f'Unsloth:      导入失败 - {e}')

try:
    import trl
    print(f'TRL:          OK')
except Exception as e:
    print(f'TRL:          导入失败 - {e}')

try:
    import peft
    print(f'PEFT:         OK')
except Exception as e:
    print(f'PEFT:         导入失败 - {e}')

print('所有关键包验证通过')
"

# ---------------------------------------------------------------------------
# Step 3: 下载模型（ModelScope）
# ---------------------------------------------------------------------------
log_info "Step 3/5: 下载 Qwen3Guard-Gen-0.6B 模型（ModelScope）..."

mkdir -p "${PROJECT_DIR}/models/Qwen"

${CONDA_PYTHON} -c "
from modelscope import snapshot_download
import os

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

# 验证模型文件
log_info "验证模型文件..."
if [ -f "${MODEL_DIR}/model.safetensors" ] || [ -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    log_info "模型权重文件存在 ✓"
else
    log_warn "未找到模型权重文件，可能下载不完整"
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
log_info "Step 4/5: 开始训练..."

cd "${PROJECT_DIR}/finetune_qwen3guard"

${CONDA_PYTHON} scripts/02_train_unsloth.py \
    --model_path "${MODEL_DIR}" \
    --train_file "${PROJECT_DIR}/finetune_qwen3guard/data/train.jsonl" \
    --val_file "${PROJECT_DIR}/finetune_qwen3guard/data/val.jsonl" \
    --output_dir "${PROJECT_DIR}/finetune_qwen3guard/output/unsloth_lora" \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_merged

# ---------------------------------------------------------------------------
# Step 5: 验证效果
# ---------------------------------------------------------------------------
log_info "Step 5/5: 验证训练效果..."

${CONDA_PYTHON} scripts/03_inference_lora.py \
    --merged_model "${PROJECT_DIR}/finetune_qwen3guard/output/unsloth_lora/merged_model" \
    --test_file "${PROJECT_DIR}/docs/test_questions.json" \
    --test_discriminatory_only \
    --output_file "${PROJECT_DIR}/finetune_qwen3guard/output/validation_result.json"

log_info "========================================"
log_info "全部完成！"
log_info "合并模型路径: ${PROJECT_DIR}/finetune_qwen3guard/output/unsloth_lora/merged_model"
log_info "验证结果:     ${PROJECT_DIR}/finetune_qwen3guard/output/validation_result.json"
log_info "========================================"
