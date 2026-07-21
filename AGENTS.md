# LLM_Guard 项目指南

> 本文件面向 AI 编程代理。阅读前请假设你对本项目一无所知。以下内容基于仓库实际文件整理，请勿依赖假设。

---

## 1. 项目概述

LLM_Guard 是一个基于 **Qwen3Guard-Gen-0.6B** 的本地 LLM 安全审核研究/工程仓库，用于对中文提示（prompt）和回复（response）进行 **Safe / Unsafe / Controversial** 三档安全分类。

核心用途：

- 对 `all_questions.xlsx` 中的问题进行两阶段批量处理：先由本地安全模型判定是否安全，再对安全题目调用后端大模型生成答案。
- 提供 OpenAI 兼容的在线服务（`/v1/chat/completions`），在转发用户请求前先做安全过滤。
- 通过 LoRA 微调持续优化特定类别（如歧视内容）的检测能力。

**重要前提**：

- 模型权重全部从本地路径加载，不从 Hugging Face Hub 下载。
- 基座模型位于 `models/Qwen/Qwen3Guard-Gen-0.6B/`（约 1.5 GB，safetensors 格式）。
- 当前生产使用的微调合并模型路径为 `finetune_qwen3guard/output/lora_v5_1/merged_model`。`scripts/services/safety_service.py` 也已统一指向该路径。

---

## 2. 技术栈

- **语言**：Python 3.11（通过 conda 环境 `py311` 运行）。
- **深度学习框架**：PyTorch + Hugging Face `transformers`。
- **微调**：LoRA / PEFT / TRL（当前活跃脚本），历史版本使用 Unsloth（已归档）。
- **数据处理**：`pandas`、`openpyxl`。
- **HTTP 后端**：`httpx`（批量脚本）、`fastapi` + `uvicorn` + `openai`（服务）。
- **日志**：`loguru`。
- **无构建系统**：没有 `pyproject.toml`、`setup.py`、`package.json`、`Makefile` 或标准测试框架，所有流程都是直接运行的 ad-hoc Python 脚本。

---

## 3. 项目结构

```
LLM_Guard/
├── README.md                          # 中文项目主文档
├── CLAUDE.md                          # Claude Code 项目指令（偏旧，注意冲突）
├── requirements.txt                   # 根目录依赖（推理 + 微调 + 服务）
│
├── pipeline/                          # 主两阶段处理脚本
│   ├── batch_stage1_safety.py         # Stage 1：本地安全模型批量推理
│   ├── batch_stage2_backend.py        # Stage 2：后端 API 调用
│   ├── export_stage1_for_review.py    # 将 Stage 1 结果导出为 Excel
│   ├── merge_stage1_stage2.py         # 合并 Stage 1 + Stage 2 结果
│   ├── fill_safe_empty.py             # 对 Safe 但答案为空的行补调后端 API
│   ├── process_0721_stage2.py         # 0721 新版测试题 Stage 2 填充
│   └── fill_remaining_0721.py         # 0721 剩余空行回填
│
├── tools/                             # 用户手动运行的 ad-hoc 工具
│   └── answer_非拒答_qwen35.py        # 手动对“非拒答”sheet 调用后端 API（用户脚本，勿改）
│
├── data/                              # 数据与中间结果
│   ├── interim/                       # Stage 1 原始推理结果（pickle）
│   │   ├── .batch_stage1_results.pkl
│   │   └── .batch_stage1_results_0721.pkl
│   ├── raw/                           # 原始输入数据
│   │   ├── 附件4 天津关键词拦截列表_合并去重.xlsx
│   │   ├── 附件4_原始备份.xlsx
│   │   ├── 附件5 天津测试题_合并去重.xlsx
│   │   └── 附件5_天津测试题_最终合并结果.xlsx
│   └── reference/                     # 用于拷贝答案的历史已回答文件
│       └── 附件5 天津测试题_合并去重_已回答.xlsx
│
├── questions/                         # 问题工作簿（输入 + Stage 输出）
│   ├── all_questions.xlsx             # 主输入数据（47,772 条，3 个 sheet）
│   ├── 0721-附件5_测试题.xlsx
│   ├── 0721-附件5_测试题_已回答.xlsx
│   ├── stage1_review.xlsx             # Stage 1 人工审核输出
│   └── stage1_v5_1_raw.xlsx           # Stage 1 原始结果（带后处理）
│
├── models/                            # 本地模型权重（gitignore 忽略大文件）
│   └── Qwen/
│       └── Qwen3Guard-Gen-0.6B/       # 基座模型
│
├── finetune_qwen3guard/               # 微调相关
│   ├── data/
│   │   ├── train_v5.jsonl             # v5 训练数据
│   │   └── val_v5.jsonl               # v5 验证数据
│   ├── output/                        # 训练输出（gitignore 忽略）
│   ├── scripts/
│   │   ├── 01_prepare_data_v5.py      # 构造 v5 数据集
│   │   ├── 02_train_lora.py           # 当前活跃的 LoRA 训练脚本
│   │   ├── augment_dataset.py         # v2 -> v3 数据增强
│   │   ├── merge_adapter.py           # CPU 合并 LoRA adapter
│   │   └── test_merged_model.py       # 合并模型冒烟测试
│   └── scripts/archive/               # 历史脚本（含 02_train_unsloth.py）
│
├── scripts/                           # 辅助脚本与入口
│   ├── install.sh / install_deps.sh   # 依赖安装
│   ├── run_finetune.sh / smoke_test.sh# 训练包装脚本
│   ├── start_server.sh                # 服务启停脚本
│   ├── demos/
│   │   └── hello_qwen3guard.py        # 单条快速测试 demo
│   ├── services/
│   │   └── safety_service.py          # FastAPI 服务入口
│   └── archive/
│       ├── batch_process_two_stage.py # 两阶段合一（旧版参考）
│       ├── run.sh                     # 已失效的包装脚本
│       └── run_8b.sh                  # 已失效的包装脚本
│
├── docs/                              # 评估语料与文档
│   ├── test_questions.json            # 20,910 条评估问题（7 个类别）
│   ├── answer_qwen3_0.6B.json         # 0.6B 模型输出
│   ├── answer_qwen3_8B.json           # 8B 模型输出
│   ├── divergence_report.json         # 模型分歧分析
│   ├── dataset_construction_plan.md   # v2 数据集设计
│   └── project_report_2026-05-18.md   # 项目报告
│
├── results/                           # 批量评估结果（CSV + summary）
└── logs/                              # 服务运行日志
```

---

## 4. 模型与数据

### 4.1 分类标签

模型输出为自由文本，需用正则解析：

```text
Safety: Safe | Unsafe | Controversial
Categories: <category>[, <category>...]
Refusal: Yes | No      # 仅在 response moderation 时出现
```

可用的 `Categories`（来自 `models/Qwen/Qwen3Guard-Gen-0.6B/README.md`）：

- `Violent`
- `Non-violent Illegal Acts`
- `Sexual Content or Sexual Acts`
- `PII`
- `Suicide & Self-Harm`
- `Unethical Acts`（歧视/偏见类统一归为此类）
- `Politically Sensitive Topics`
- `Copyright Violation`
- `Jailbreak`（仅输入）
- `None`

### 4.2 主输入数据 `all_questions.xlsx`

包含 3 个工作表：

| Sheet | 样本数 | 期望标签 | 说明 |
|-------|--------|----------|------|
| 生成类 | 19,220 | Unsafe | 全部拒绝 |
| 拒答 | 20,836 | Unsafe | 全部拒绝 |
| 非拒答 | 7,716 | Safe | 全部通过 |

### 4.3 模型路径

| 用途 | 路径 |
|------|------|
| 基座模型 | `models/Qwen/Qwen3Guard-Gen-0.6B` |
| 当前活跃微调模型 | `finetune_qwen3guard/output/lora_v5_1/merged_model` |

**注意**：所有脚本已统一指向 `lora_v5_1`。历史 Unsloth 模型路径（v2 / v3）仅在 `scripts/archive/` 中保留，不再使用。

---

## 5. 运行命令

所有命令默认在 conda 环境 `py311` 中执行：

```bash
conda run -n py311 python3 <script>.py
```

### 5.1 环境安装

```bash
# 方式 1：安装根目录全部依赖
bash scripts/install_deps.sh

# 方式 2：仅安装推理依赖（使用 CUDA 12.1）
bash scripts/install.sh

# 方式 3：手动安装
pip install -r requirements.txt
```

### 5.2 单条测试

```bash
conda run -n py311 python3 scripts/demos/hello_qwen3guard.py
```

### 5.3 两阶段批量处理（推荐当前流程）

```bash
# Stage 1：本地安全模型推理
conda run -n py311 python3 pipeline/batch_stage1_safety.py

# 导出审核 Excel
conda run -n py311 python3 pipeline/export_stage1_for_review.py

# Stage 2：审核通过后，调用后端大模型（需确认后再运行）
conda run -n py311 python3 pipeline/batch_stage2_backend.py
```

环境变量可覆盖默认配置：

- `INPUT_FILE`：输入 Excel（默认 `questions/all_questions.xlsx`）
- `STAGE1_OUTPUT`：Stage 1 pickle 输出（默认 `data/interim/.batch_stage1_results.pkl`）
- `OUTPUT_FILE`：Stage 2 输出（默认 `questions/all_questions_finished.xlsx`）
- `REVIEW_FILE`：审核 Excel 输出（默认 `questions/stage1_review.xlsx`）

### 5.4 启动服务

```bash
# 前台运行
bash scripts/start_server.sh --help   # 查看用法

# 后台运行
bash scripts/start_server.sh

# 停止
bash scripts/start_server.sh --stop

# 状态
bash scripts/start_server.sh --status
```

服务监听 `0.0.0.0:32469`，使用 GPU 3（`CUDA_VISIBLE_DEVICES=3`）。

端点：

- `GET /health`
- `POST /v1/chat/completions`（OpenAI 兼容，支持流式）

### 5.5 微调

当前活跃训练脚本：

```bash
conda run -n py311 python3 finetune_qwen3guard/scripts/02_train_lora.py \
  --model_path models/Qwen/Qwen3Guard-Gen-0.6B \
  --train_file finetune_qwen3guard/data/train_v5.jsonl \
  --val_file finetune_qwen3guard/data/val_v5.jsonl \
  --output_dir finetune_qwen3guard/output/lora_v5_1 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --save_merged
```

合并 LoRA：

```bash
conda run -n py311 python3 finetune_qwen3guard/scripts/merge_adapter.py
```

冒烟测试：

```bash
conda run -n py311 python3 finetune_qwen3guard/scripts/test_merged_model.py
```

---

## 6. 代码组织原则

- **无包结构**：所有脚本都是可执行文件，通过相对路径（以项目根目录为工作目录）引用数据和模型。
- **配置内联**：各脚本顶部有 `Configuration` 区块，模型路径、API 地址、超时等常数直接写在文件里，未使用配置文件。
- **重复解析逻辑**：安全输出解析正则（`SAFETY_RE`、`CATEGORY_RE`）在多个文件中重复定义（`pipeline/batch_stage1_safety.py`、`pipeline/batch_stage2_backend.py`、`scripts/services/safety_service.py`、`scripts/archive/batch_process_two_stage.py`）。如需修改分类正则，请同步所有文件。
- **用户脚本保护**：`tools/answer_非拒答_qwen35.py` 被 README 明确标注为“用户手动运行，请勿修改”。除非用户主动要求，否则不要编辑该文件。

---

## 7. 代码风格指南

- 注释以中文为主，函数/变量名使用英文 snake_case。
- 每个脚本顶部保留模块级 docstring，说明用途、输入、输出和用法。
- 配置常量集中在文件顶部 `Configuration` 区块。
- 日志统一使用 `loguru`，格式为：`<time> | <level> | <message>`。
- 模型加载使用 `trust_remote_code=True` 和本地路径。
- 批量推理使用 `torch.no_grad()`、`do_sample=False`，保证确定性输出。
- 解析模型输出时优先使用正则，不依赖分类头。

---

## 8. 测试说明

本项目没有单元测试框架。验证方式：

1. **单条推理验证**：`scripts/demos/hello_qwen3guard.py`
2. **合并模型冒烟测试**：`finetune_qwen3guard/scripts/test_merged_model.py`
3. **服务健康检查**：`curl http://localhost:32469/health`
4. **批量结果统计**：运行 `pipeline/batch_stage1_safety.py` 后查看日志中的 Safe/Unsafe/Controversial 计数。
5. **端到端验证**：运行 Stage 1 + Stage 2 后检查 `questions/all_questions_finished.xlsx`。

---

## 9. 部署注意事项

- **GPU 要求**：推理需要 NVIDIA GPU；训练脚本默认使用 `CUDA_VISIBLE_DEVICES=3`（`02_train_lora.py` 第 11 行）。部署前确认 GPU 编号。
- **后端 API**：`pipeline/batch_stage2_backend.py` 和 `scripts/services/safety_service.py` 都调用内网后端 `http://172.31.0.97:3391/v1`，模型名为 `qwen3.5-122b-a10b`。该地址是项目特定环境的一部分，迁移时需修改。
- **API Key**：后端 API Key 硬编码在 `pipeline/batch_stage2_backend.py`、`scripts/services/safety_service.py`、`tools/answer_非拒答_qwen35.py`、`pipeline/fill_safe_empty.py` 中。生产部署应改为环境变量或密钥管理。
- **关键词过滤**：`scripts/services/safety_service.py` 包含一个从 `data/raw/附件4 天津关键词拦截列表_合并去重.xlsx` 加载的关键词预过滤模块，但当前代码中 `keyword_hit=False` 被硬编码关闭。修改前请确认业务需求。

---

## 10. 已知问题与陷阱

1. **`scripts/archive/run.sh` 和 `scripts/archive/run_8b.sh` 已失效**，它们引用了不存在的 `answer_questions.py` 和 8B 模型，仅作归档保留。
2. **`.gitignore` 忽略 `models/` 下所有 safetensors/bin 等大文件**，因此新 clone 的仓库没有模型权重，需要从其他途径恢复。
3. **没有标准测试框架**，所有验证都是脚本式、手动的。
4. **CLAUDE.md 部分信息已过时**。请以本文件和实际代码为准。

---

## 11. 安全注意事项

- 后端 API Key 以明文形式出现在多个脚本中。修改或提交前注意不要泄露到公共仓库。
- 安全模型输出是自由文本，依赖正则解析。修改解析逻辑时务必对照 `models/Qwen/Qwen3Guard-Gen-0.6B/README.md` 中的官方示例，避免破坏分类正确性。
- `batch_stage2_backend.py` 会向内网发送大量用户问题，运行前需确认网络可达和权限。
- 关键词过滤模块当前被显式关闭；如启用，需评估对正常请求的误拦截风险。

---

## 12. 修改前 checklist

- [ ] 确认要使用的模型路径在磁盘上真实存在。
- [ ] 若修改安全输出解析正则，同步所有引用该正则的脚本。
- [ ] 若修改后端 API 地址或 Key，同步 `pipeline/batch_stage2_backend.py`、`scripts/services/safety_service.py`、`tools/answer_非拒答_qwen35.py`、`pipeline/fill_safe_empty.py`。
- [ ] 不要修改 `tools/answer_非拒答_qwen35.py` 除非用户明确要求。
- [ ] 运行修改后先用 `scripts/demos/hello_qwen3guard.py` 或 `finetune_qwen3guard/scripts/test_merged_model.py` 做冒烟验证。
