# LLM_Guard — 项目报告 (2026-05-18)

## 概述

这是一个 **LLM 安全审核** 研究项目，运行 **Qwen3Guard-Gen-0.6B**（0.6B 参数的生成式安全分类器），对中文安全评估语料库中的提示和回复进行 **Safe / Unsafe / Controversial** 分类。该模型将安全判断构建为指令遵循（生成文本后由正则表达式解析），而非使用分类器头部。项目最近的重点是通过 **LoRA 微调** 来提升歧视内容检测能力。

- **框架：** PyTorch + Transformers + Unsloth + PEFT
- **环境：** conda `py311`（Python 3.11，Torch 2.10.0，CUDA 12.8）
- **硬件：** 8x NVIDIA A100-SXM4-80GB

---

## 目录结构

```
LLM_Guard/
├── hello_qwen3guard.py          # 入门：单次推理管道
├── answer_questions.py          # 生产批量推理（0.6B / 8B）
├── compare_models.py            # 基线 vs 微调模型对比评估
├── merge_adapter.py             # LoRA 合并为完整模型（仅 CPU）
├── test_merged_model.py         # 合并模型的冒烟测试
├── run.sh / run_8b.sh           # 推理包装器
├── run_finetune.sh / smoke_test.sh  # 训练启动器
├── install.sh / install_deps.sh # 依赖安装
├── requirements.txt             # transformers, torch, peft, trl, unsloth, datasets
│
├── docs/                        # 评估语料库（~20,910 条记录）
│   ├── test_questions.json      # 7 个类别的原始中文测试问题
│   ├── answer_qwen3_0.6B.json   # 0.6B 模型的输出
│   ├── answer_qwen3_8B.json     # 8B 模型的输出
│   ├── divergence_report.json   # Qwen3Guard 与对比模型的高分歧交叉表
│   ├── dataset_construction_plan.md  # v2 数据设计文档
│   └── aaa-附件5：*.json/xlsx  # 原始评估 Excel + JSON 格式
│
├── models/Qwen/Qwen3Guard-Gen-0.6B/  # 本地模型（~1.5 GB）
│
└── finetune_qwen3guard/
    ├── data/
    │   ├── train_v2.jsonl (2,315条)  # v2 训练集（当前）
    │   ├── val_v2.jsonl   (256条)    # v2 验证集
    │   ├── train.jsonl    (1,842条)  # v1 训练集
    │   └── val.jsonl      (191条)    # v1 验证集
    ├── scripts/
    │   ├── 01_prepare_data_v2.py     # v2 数据构建（708行，7层分层）
    │   ├── 01_prepare_data.py        # v1 数据构建
    │   ├── 02_train_unsloth.py       # 主训练脚本（Unsloth + LoRA）
    │   ├── 02_train_lora.py          # 备用训练（标准 PEFT）
    │   ├── 03_inference_lora.py      # LoRA 推理与评估
    │   └── 04_merge_lora.py          # LoRA 合并
    └── output/
        ├── unsloth_lora/             # v1 训练（174步，3 epochs）
        │   ├── final_adapter/        # 40 MB LoRA 权重
        │   └── merged_model/         # ~1.5 GB 完整模型
        └── unsloth_lora_v2/          # v2 训练（57步，3 epochs）
            ├── final_adapter/        # 40 MB LoRA 权重
            └── merged_model/         # ~1.2 GB 完整模型
```

---

## 核心管道

`hello_qwen3guard.py` 定义了两个函数：

- `moderate(messages)` — 应用聊天模板，生成，解码新 token
- `parse(content)` — 用正则提取 `safety`、`categories`、`refusal`

同一个 `moderate()` 函数同时处理**提示审核**（仅 `user`）和**回复审核**（`user` + `assistant`）。

---

## 微调结果

### v1（有完整评估）

| 测试集                        | 基线准确率 | 微调后准确率 | 变化     |
|-------------------------------|-----------|-------------|---------|
| **A.2 歧视内容**              | 31.2%     | **100.0%**  | +68.8%  |
| **A.3 商业违法**              | 87.5%     | **99.5%**   | +12.0%  |
| A.1 政治安全                  | 90.0%     | 86.2%       | -3.8%   |
| A.4 侵犯权益                  | 92.0%     | 94.3%       | +2.4%   |
| 安全 — 准确性                 | 100.0%    | 100.0%      | 0%      |
| 安全 — 可靠性                 | 100.0%    | 100.0%      | 0%      |
| 安全 — 无害话题               | 99.5%     | 99.1%       | -0.4%   |

- **训练效率：** RTX 4090 上 3 epochs 约 **7 分钟**
- **安全内容误报率：** 仅 0.9%，远低于 5% 的目标

### v2（已训练，已合并，尚未评估）

v2 数据包含 **2,315 条训练** / **256 条验证**，采用 7 层分层构建：

1. **L1 — 歧视纠正：** 将 0.6B 的漏检（Safe → Unsafe）修正
2. **L2 — 正负配对增强：** ~150 对手工制作的安全/不安全对
3. **L3 — 学术讨论保留：** 避免将学术话题错误标记为歧视
4. **L4 — 正确歧视拦截：** 模型本来判断正确的样本
5. **L5 — 安全样本：** 良性问题锚定
6. **L6 — 其他类别正确拦截：** 确保非歧视类别不退化
7. **L7 — 其他类别纠正：** 修正其他类别的错误

使用 13 个学术模式和 221 个歧视模式的正则表达式来区分"真歧视"和"学术讨论"。

---

## 评估数据集

`docs/test_questions.json` 包含 7 个类别的 **20,910 条记录**：

| 类别 | 记录数 |
|------|--------|
| `appendix_a_content_evaluation_manual_sampling` | 2,158 |
| `appendix_a_content_evaluation_keyword_sampling` | 16,153 |
| `intellectual_property_trade_secrets_test` | 200 |
| `discriminatory_content_test` | 529 |
| `accuracy` | 350 |
| `reliability` | 352 |
| `should_refuse` | 1,168 |

每条记录：`question`、`primary_category`、`secondary_category`。已回答版本额外包含 `answer`、`safety`、`categories` 字段。

`divergence_report.json` 包含 **6,651 条高分歧案例**，其中 Qwen3Guard 与对比模型判断不一致。

---

## 已知问题

1. **`compare_models.py` 会崩溃** — 引用了已被 `git rm` 的 `docs/test_questions_answered_qwen3guard.json`
2. **8B 模型不存在** — `MODEL_PATHS["8B"]` 指向的 `models/Qwen/Qwen3Guard-Gen-8B` 不在磁盘上
3. **v2 尚未评估** — `unsloth_lora_v2/` 的合并模型已生成，但还没运行 `compare_models.py` 对比评估
4. **GPU 已满** — 8 张 A100 每张 ~75/81 GB（92%），合并操作已改用 CPU 执行

---

## 当前分支状态

- **分支：** `main`
- **最新提交：** `db648e2` — "忽略 Models 目录下的所有内容。"
- **已修改：** `answer_questions.py`、`run.sh`、`02_train_unsloth.py`
- **已删除：** `test_questions_answered_qwen3guard.json`
- **未跟踪：** `compare_models.py`、`merge_adapter.py`、`test_merged_model.py`、v2 数据文件、所有评估 JSON、多个 shell 脚本、`unsloth_compiled_cache/`
