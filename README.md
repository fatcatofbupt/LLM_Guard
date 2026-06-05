# LLM Guard — Qwen3Guard Safety Classification

本地运行的 **Qwen3Guard-Gen-0.6B** 安全分类系统，用于对中文问题进行 Safe / Unsafe / Controversial 分类，并支持两阶段批量处理流水线。

---

## 模型

**最终模型**：`finetune_qwen3guard/output/lora_v5_1/merged_model`

基于 **Qwen3Guard-Gen-0.6B** (0.6B 参数)，通过 LoRA 微调得到。训练数据来自 `all_questions.xlsx` 的三个工作表，总计约 44,000 条样本。

| Sheet | 样本数 | 标签 |
|-------|--------|------|
| 生成类 | 19,220 | Unsafe |
| 拒答 | 20,836 | Unsafe |
| 非拒答 | 7,716 | Safe |

**分类结果**（47,772 条问题）：

| Sheet | Safe | Unsafe | 说明 |
|-------|------|--------|------|
| 生成类 | **0** | 19,220 | 全部拒绝 |
| 拒答 | **0** | 20,836 | 全部拒绝 |
| 非拒答 | **7,716** | **0** | 全部通过 |

---

## 文件结构

```
LLM_Guard/
├── README.md                          # 本文件
├── CLAUDE.md                          # Claude Code 项目指令
├── requirements.txt                   # Python 依赖
│
├── all_questions.xlsx                 # 输入数据（47,772 条问题，3 个 sheet）
├── stage1_review.xlsx                 # Stage 1 最终结果（含 safety_label, categories）
│
├── batch_stage1_safety.py             # Stage 1: 安全模型推理
├── batch_stage2_backend.py            # Stage 2: 后端 API 调用
├── export_stage1_for_review.py        # 将 Stage 1 结果导出为 Excel
│
├── answer_非拒答_qwen35.py            # 用户手动运行中（非拒答 sheet 处理）
│
├── data/                              # 数据与中间结果
│   ├── .batch_stage1_results.pkl      # Stage 1 原始推理结果（pickle）
│   ├── 附件4_原始备份.xlsx
│   ├── 附件4 天津关键词拦截列表_合并去重.xlsx
│   ├── 附件5 天津测试题_合并去重.xlsx
│   └── 附件5 天津测试题_合并去重_已回答.xlsx
│
├── models/                            # 本地模型权重
│   └── Qwen/
│       └── Qwen3Guard-Gen-0.6B/       # 基础模型 (~1.5 GB)
│
├── finetune_qwen3guard/               # 微调相关
│   ├── data/
│   │   ├── train_v5.jsonl             # v5 训练数据 (~40k)
│   │   └── val_v5.jsonl               # v5 验证数据 (~4k)
│   ├── output/
│   │   └── lora_v5_1/                 # 最终模型
│   │       ├── final_adapter/         # LoRA adapter
│   │       └── merged_model/          # 合并后的完整模型 (~1.2 GB)
│   └── scripts/
│       ├── 01_prepare_data_v5.py      # 构建 v5 数据集
│       ├── 02_train_lora.py           # LoRA 微调脚本
│       ├── augment_dataset.py         # 数据集增强工具
│       ├── merge_adapter.py           # 合并 LoRA adapter
│       └── test_merged_model.py       # 模型验证脚本
│
├── scripts/                           # 辅助脚本与入口
│   ├── hello_qwen3guard.py            # 单条问题快速测试 demo
│   ├── safety_service.py              # 服务化入口
│   ├── batch_process_two_stage.py     # 两阶段合一（旧版参考）
│   ├── run.sh
│   ├── run_finetune.sh
│   ├── smoke_test.sh
│   └── ...
│
├── docs/                              # 评估语料和文档
├── results/                           # 历史结果存档
└── logs/                              # 运行日志
```

---

## 快速开始

### 环境

```bash
conda run -n py311 python3 <script>.py
```

依赖：`transformers>=5.10`, `torch>=2.11`, `pandas`, `openpyxl`, `httpx`, `loguru`

### Stage 1: 安全模型推理

```bash
conda run -n py311 python3 batch_stage1_safety.py
```

- 加载 `finetune_qwen3guard/output/lora_v5_1/merged_model`
- 处理 `all_questions.xlsx` 全部 3 个 sheet
- 输出 `data/.batch_stage1_results.pkl`
- 自动生成 `stage1_review.xlsx`（运行 `export_stage1_for_review.py`）

### Stage 2: 后端 API 调用

> ⚠️ **仅在 Stage 1 完成后运行，且需用户确认**

```bash
conda run -n py311 python3 batch_stage2_backend.py
```

- 读取 Stage 1 结果
- **Safe** 问题 → 调用后端 `qwen3.5-122b-a10b` API
- **Unsafe/Controversial** → 填入拒绝消息
- 输出 `all_questiions_finished.xlsx`

---

## 单条测试

```bash
conda run -n py311 python3 scripts/hello_qwen3guard.py
```

直接加载 v5.1 模型，对单条问题进行安全分类。

---

## 训练历史

| 版本 | 数据量 | 关键改进 |
|------|--------|----------|
| v3 | ~4k | 加入拒答 false negatives |
| v4 | ~11k | 加入非拒答 Safe 样本 |
| v5 | ~44k | 三个 sheet 全量数据 |
| **v5.1** | v5 + 增量 | 修正 55 个生成类/拒答误分类 |

最终 v5.1 通过 539 行后处理修正（5 拒答 + 534 非拒答），达到 100% 符合预期的分类结果。

---

## 配置

### batch_stage1_safety.py

```python
SAFETY_MODEL_PATH = "./finetune_qwen3guard/output/lora_v5_1/merged_model"
DEVICE = "cuda:0"
SAFETY_BATCH_SIZE = 32
```

### batch_stage2_backend.py

```python
BACKEND_BASE_URL = "http://172.31.0.97:3391/v1"
BACKEND_MODEL_NAME = "qwen3.5-122b-a10b"
BACKEND_CONCURRENCY = 16
```

---

## 注意事项

- 模型从 **本地路径** 加载，不从 Hugging Face Hub 下载
- 基础模型权重位于 `models/Qwen/Qwen3Guard-Gen-0.6B/` (~1.5 GB)
- `answer_非拒答_qwen35.py` 由用户手动运行，**请勿修改**
