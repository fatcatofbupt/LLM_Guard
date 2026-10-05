# 实验报告：Qwen3Guard-Gen-0.6B LoRA 微调与 0929 批次问答处理

- 实验日期：2026-10-05
- 实验地点：远端 GPU 服务器（SSH：`ssh tts@112.45.47.21`，免密登录）
- 报告面向：任何没有参与过本项目的第三方读者。阅读本报告无需任何先验知识。

---

## 1. 实验摘要

本项目是一套中文问题的"安全判定 + 自动回答"流水线。本次实验完成三件事：

1. 用 v5 数据集（约 4.4 万条）对安全判定模型 Qwen3Guard-Gen-0.6B 做 LoRA 微调，产出 lora_v5_1 模型；
2. 用微调后的模型处理"0929 测试题"批次（1,200 道题，分三类），产出已回答工作簿；
3. 用未微调的基座模型跑同一批题作为对照实验。

**核心结论**：微调模型对"生成类"和"拒答"两类题目判定更严格（全部拒答，符合业务预期），但对"非拒答"类题目出现明显的过度拦截（400 题只放行 7 题，历史通过率约 96%）。根本原因是训练数据严重偏向 Unsafe 标签（占 82.7%）。基座模型在同批题上放行 356 题，整体更接近历史行为，可作为对照参考。

**最终交付物**：`questions/0929-附件5_A5普通用户_测试题_已回答.xlsx`（按线上规则产出，即微调版判定结果）。

---

## 2. 背景与术语（零先验知识）

### 2.1 业务目标

收到一份 Excel 工作簿，内含 1,200 道中文测试题，分三个工作表（sheet）。要求对每道题给出"模型的答案"：该拒答的拒答，该回答的调用大模型回答。

### 2.2 三类题目

| Sheet 名 | 含义 | 期望行为 |
|---|---|---|
| 生成类 | 诱导生成违规内容的请求 | 一律拒答 |
| 拒答 | 应当拒绝回答的问题 | 一律拒答 |
| 非拒答 | 正常、可以回答的问题 | 调用后端大模型回答 |

### 2.3 两阶段流水线（Pipeline）

一次完整的处理分两步：

- **Stage 1（安全判定）**：本地运行一个小模型（Qwen3Guard-Gen-0.6B，0.6B 参数），对每道题输出 `Safety: Safe` 或 `Safety: Unsafe`（及违规类别）。1,200 题约 20 秒。
- **Stage 2（答案生成）**：Stage 1 判为 Safe 的题，调用远端后端大模型生成答案；判为 Unsafe/Controversial 的题，直接填入固定拒绝语 `根据政策和法律规定，我无法回答你的问题。`

这样设计的原因：后端大模型按 API 调用计费且速度有限，先用本地小模型过滤掉必须拒答的题，可以省钱省时间，也保证违规内容不会送到后端模型。

### 2.4 Qwen3Guard-Gen-0.6B

阿里通义千问团队发布的安全判定小模型。输入一道问题，输出固定格式：

```
Safety: Safe|Unsafe|Controversial
Categories: <违规类别，如 Unethical Acts；Safe 时为 None>
```

模型仅 0.6B 参数（约 6 亿），单张消费级显卡即可运行。本项目的微调（见第 5 节）不改变模型结构，只改变其判定倾向。

### 2.5 LoRA 微调

LoRA（Low-Rank Adaptation）是一种参数高效微调方法：冻结原模型全部权重，仅训练额外插入的低秩矩阵。本次训练只更新约 1,000 万参数（占总参数 606M 的 1.67%）。好处是训练快、显存占用小、 adapter 文件小。

---

## 3. 实验环境

### 3.1 GPU 服务器硬件

| 项目 | 实测值 | 说明 |
|---|---|---|
| GPU | 4 × NVIDIA GeForce RTX 4090 | 每卡显存约 48GB（nvidia-smi 显示 49,140 MiB） |
| CPU | 56 核 | — |
| 内存 | 472GB | — |
| 磁盘 | 491GB（实验时用 38%） | Ubuntu，LVM 分区 |
| 连接方式 | `ssh tts@112.45.47.21` | 已配置免密登录 |
| 项目目录 | `~/LLM_Guard`（即 `/home/tts/LLM_Guard`） | 仓库代码同步于此 |

**GPU 占用纪律（重要）**：这台服务器不是本项目独占。4 张卡上都有其他用户的常驻服务（实测 GPU0 已用 46.4GB，其余三卡各约 13.8GB）。本项目所有任务固定使用 **GPU1**（`CUDA_VISIBLE_DEVICES=1`），该卡空闲显存约 35GB，足够训练与推理。绝不动其他 conda 环境和在跑服务。

### 3.2 软件环境

使用 conda 环境 `py311`（本实验新建；服务器 conda 安装在 `~/miniconda3`，不在默认 PATH 中）：

| 软件 | 版本 |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.1（+ cu130） |
| transformers | 5.18.0 |
| trl | 1.14.1 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.0.1 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| loguru | 0.7.3 |

**中国大陆加速配置**（本项目服务器在国内，默认源很慢）：

- conda 频道：清华 Tuna anaconda 镜像，已写入服务器 `~/.condarc`；
- pip：安装时用 `-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`；
- 模型权重：从 ModelScope（阿里模型仓库）下载，不走 Hugging Face。

### 3.3 模型与数据路径

| 内容 | 服务器路径 | 大小 |
|---|---|---|
| 基座模型权重 | `~/LLM_Guard/models/Qwen/Qwen3Guard-Gen-0.6B/` | 1.5GB |
| 微调产物（LoRA adapter） | `~/LLM_Guard/finetune_qwen3guard/output/lora_v5_1/final_adapter/` | 小（仅 adapter） |
| 微调产物（合并后完整模型） | `~/LLM_Guard/finetune_qwen3guard/output/lora_v5_1/merged_model/` | 1.1GB |
| 训练数据 | `~/LLM_Guard/finetune_qwen3guard/data/train_v5.jsonl` / `val_v5.jsonl` | 8.0MB / 0.9MB |
| 0929 输入工作簿 | `~/LLM_Guard/questions/0929-附件5_A5普通用户_测试题.xlsx` | 73KB |
| 0929 输出工作簿 | `~/LLM_Guard/questions/0929-附件5_A5普通用户_测试题_已回答.xlsx` | 83KB |

合并后的模型已备份到本地 Mac：`finetune_qwen3guard/output/lora_v5_1/merged_model/`（1.1GB，safetensors 格式 + tokenizer）。该目录被 gitignore，仅本地保存。

### 3.4 后端大模型（Stage 2 答案生成）

Stage 2 调用的后端是部署在阿里云 PAI-EAS 上的 vLLM 大模型服务（Qwen3 系列）。访问参数统一从项目根目录的 `config.toml` 读取（模板为 `config.example.toml`；`config.toml` 已 gitignore，API Key 只存本地，**不入 Git**）：

| 配置项 | 说明 |
|---|---|
| `base_url` | OpenAI 兼容接口地址，必须以 `/v1` 结尾 |
| `api_key` | 访问密钥 |
| `model_name` | 必须填服务方指定的名字 `LegalBrain`（不是 URL 里的服务名） |
| `enable_thinking` | 思考型模型的开关；本项目设为 `false`（直接回答，省 token） |

调用细节：HTTP POST 到 `{base_url}/chat/completions`，请求体含 `chat_template_kwargs.enable_thinking`（顶层参数无效），并发 16，总超时 90 秒。

---

## 4. 实验数据

### 4.1 v5 训练集（用于微调）

来源：`questions/all_questions.xlsx` 三个 sheet 全量 + v3 版本独有的 Unsafe 样本，按题目去重（同一题多来源时保留最严格的 Unsafe 标签），90/10 随机切分（seed=42）。

标注规则：生成类、拒答 → `Unsafe / Unethical Acts`；非拒答 → `Safe / None`。

| 数据集 | Safe | Unsafe | 合计 | Safe 占比 |
|---|---|---|---|---|
| train_v5 | 6,893 | 32,960 | 39,853 | 17.3% |
| val_v5 | 789 | 3,639 | 4,428 | 17.8% |
| **合计** | **7,682** | **36,599** | **44,281** | **17.3%** |

每条样本格式（jsonl，一行一条）：

```json
{"messages": [
  {"role": "user", "content": "<题目>"},
  {"role": "assistant", "content": "Safety: Unsafe\nCategories: Unethical Acts"}
]}
```

### 4.2 0929 评测输入

`questions/0929-附件5_A5普通用户_测试题.xlsx`，三个 sheet（生成类 / 拒答 / 非拒答）各 400 题，共 1,200 题。题目为全新题目，训练时未见过。

---

## 5. 训练过程

### 5.1 训练命令

在服务器项目根目录（`~/LLM_Guard`）执行：

```bash
cd ~/LLM_Guard
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/py311/bin/python \
  finetune_qwen3guard/scripts/02_train_lora.py \
  --train_file finetune_qwen3guard/data/train_v5.jsonl \
  --val_file   finetune_qwen3guard/data/val_v5.jsonl \
  --output_dir finetune_qwen3guard/output/lora_v5_1 \
  --save_merged
```

其余超参全部用脚本默认值（见下表）。训练日志：`logs/train_0929.log`（已存本地）。

### 5.2 超参数

| 参数 | 值 |
|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 学习率 | 2e-4，cosine 调度，warmup 50 步 |
| 每卡 batch size | 4 |
| 梯度累积 | 4（等效 batch = 16） |
| 训练轮数 | 3 epochs |
| 最大序列长 | 2048 |
| 精度 | bf16 |
| 评估 / 保存策略 | 每个 epoch 评估并保存，最多留 2 个 checkpoint，按 eval_loss 选最优 |
| 随机种子 | 42（脚本默认） |
| 优化器 | AdamW（transformers 默认） |

### 5.3 训练结果（来自日志，已核实）

| 指标 | 值 |
|---|---|
| 可训练参数 | 10,092,544（占总参数 606,142,464 的 1.67%） |
| 总步数 | 7,473 步（= 39,853 × 3 ÷ 16，吻合） |
| 训练耗时 | 5,772 秒 ≈ **96 分钟**（约 1.29 it/s） |
| 最终 train_loss | 0.00299 |
| 最终 eval_loss | 0.00299 |
| eval token 准确率 | 0.9994 |

训练在 10 月 5 日上午进行（日志文件时间戳 12:24 为训练结束时刻，开始时刻按 96 分钟时长推断约 10:45，未精确记录）。训练完成后脚本自动把 LoRA adapter 合并回基座，保存完整模型到 `merged_model/`。单卡 4090 即可跑完，峰值显存远低于 GPU1 的 35GB 空闲额度。

---

## 6. 推理与回答流程（0929 批次实测）

### Stage 1：安全判定

```bash
cd ~/LLM_Guard
CUDA_VISIBLE_DEVICES=1 INPUT_FILE="questions/0929-附件5_A5普通用户_测试题.xlsx" \
  STAGE1_OUTPUT=data/interim/.batch_stage1_results_0929.pkl \
  DEVICE=cuda:0 \
  ~/miniconda3/envs/py311/bin/python pipeline/batch_stage1_safety.py
```

- 加载 `finetune_qwen3guard/output/lora_v5_1/merged_model`，bf16，batch_size=32，贪心解码，最多生成 128 token；
- 用正则从输出中提取 `Safety:` 判定；
- 1,200 题共 21 秒（12:27:49–12:28:10）；
- 结果存 pickle，供 Stage 2 读取。三个路径/设备参数均可用环境变量覆盖（`SAFETY_MODEL_PATH` / `DEVICE` / `INPUT_FILE` / `STAGE1_OUTPUT`）。

### Stage 2：答案生成

```bash
cd ~/LLM_Guard
INPUT_FILE="questions/0929-附件5_A5普通用户_测试题.xlsx" \
  STAGE1_OUTPUT=data/interim/.batch_stage1_results_0929.pkl \
  OUTPUT_FILE="questions/0929-附件5_A5普通用户_测试题_已回答.xlsx" \
  ~/miniconda3/envs/py311/bin/python pipeline/batch_stage2_backend.py
```

- Unsafe/Controversial → 填固定拒绝语；
- Safe → 异步并发 16 路调用后端 `LegalBrain`；
- 把答案写回"模型的答案"列，输出最终工作簿。

### 基座对照

同一批 1,200 题，用基座模型（`models/Qwen/Qwen3Guard-Gen-0.6B`）重跑 Stage 1（`SAFETY_MODEL_PATH` 指到基座目录即可），结果存 `data/interim/.batch_stage1_results_0929_base.pkl`。基座版 Stage 2（实际生成回答）未跑。

---

## 7. 实验结果

同一批 1,200 题，两个模型的 Safe 判定数（Safe = 放行并送后端回答；数字来自两份 Stage 1 日志，已核实）：

| Sheet（各 400 题） | 基座模型 Safe | 微调模型 Safe | 业务期望 |
|---|---|---|---|
| 生成类 | 89 | **0** | 全部拒答 |
| 拒答 | 22 | **0** | 全部拒答 |
| 非拒答 | 245 | **7** | 几乎全部回答 |
| 合计 | 356 | 7 | — |

### 结果分析

1. **生成类、拒答：微调版更优。** 两类共 800 题，基座漏放 111 题（89+22）到后端，微调版 0 漏放。对"必须拒答"的业务目标，微调版完全符合预期。
2. **非拒答：微调版过度拦截。** 400 题仅放行 7 题（通过率 1.75%），而历史口径下非拒答题的通过率约 96%。基座版放行 245 题（61.3%），虽不及其训练集内的表现，但远好于微调版。
3. **根因是数据不均衡。** 训练集 82.7% 为 Unsafe，模型学到"偏向拒答"的先验。值得注意的是，微调模型在同源的 47,772 题全量集上，非拒答 7,716 题判定为 100% Safe（README 训练历史表）——那是见过的训练数据；本次 0929 是新题泛化测试，暴露了泛化偏差。
4. **基座版 Stage 2 未执行。** 如需基座版最终工作簿，用现成的 `data/interim/.batch_stage1_results_0929_base.pkl` 重跑 Stage 2 即可，约 2 分钟（356 次后端调用）。

### 最终交付

`questions/0929-附件5_A5普通用户_测试题_已回答.xlsx`：生成类 400 条拒绝、拒答 400 条拒绝、非拒答 393 条拒绝 + 7 条后端回答，无空值。已提交 Git。

---

## 8. 已知问题与后续建议

| 问题 | 说明 | 建议 |
|---|---|---|
| 非拒答过度拦截 | 训练数据 Unsafe 占 82.7%，导致泛化时偏向拒答 | 下版数据补充 Safe 样本或降采样 Unsafe，使比例接近线上真实分布 |
| 类别单一 | v5 数据 Unsafe 全部标为 Unethical Acts | 若后续需要按类别统计，应还原 v3 中的细粒度类别 |
| 训练开始时刻未记录 | 日志只有结束时间戳 | 无实质影响，报告中已标注为推断值 |

---

## 9. 复现手册（从零开始）

在一台新的、已装 NVIDIA 驱动和 conda 的 Linux 服务器上：

```bash
# 1. 登录并同步代码（任一方式：git clone 或 scp/rsync 本仓库）
ssh tts@112.45.47.21
cd ~ && git clone <仓库地址> LLM_Guard   # 或从本机 rsync -av LLM_Guard/ tts@112.45.47.21:~/LLM_Guard/

# 2. 配置镜像源（中国大陆）
cat > ~/.condarc <<'EOF'
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/cloud/conda-forge/
show_channel_urls: true
EOF

# 3. 创建 Python 环境（版本见 3.2 节）
~/miniconda3/bin/conda create -n py311 python=3.11 -y
~/miniconda3/envs/py311/bin/pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  torch transformers trl peft accelerate datasets pandas openpyxl httpx loguru

# 4. 下载基座模型（ModelScope，1.5GB）
mkdir -p ~/LLM_Guard/models/Qwen/Qwen3Guard-Gen-0.6B
cd ~/LLM_Guard/models/Qwen/Qwen3Guard-Gen-0.6B
curl -L -o model.safetensors "https://modelscope.cn/models/Qwen/Qwen3Guard-Gen-0.6B/resolve/master/model.safetensors"
# 其余小文件（config.json / tokenizer 等）从 ModelScope 同仓库下载，或整体用 modelscope SDK 拉取

# 5. 配置后端模型
cd ~/LLM_Guard && cp config.example.toml config.toml
# 编辑 config.toml 填入 base_url / api_key / model_name=LegalBrain / enable_thinking=false

# 6. 训练（约 96 分钟，固定用 GPU1）
cd ~/LLM_Guard
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/py311/bin/python \
  finetune_qwen3guard/scripts/02_train_lora.py \
  --train_file finetune_qwen3guard/data/train_v5.jsonl \
  --val_file   finetune_qwen3guard/data/val_v5.jsonl \
  --output_dir finetune_qwen3guard/output/lora_v5_1 \
  --save_merged

# 7. Stage 1 安全判定（约 21 秒）
CUDA_VISIBLE_DEVICES=1 INPUT_FILE="questions/0929-附件5_A5普通用户_测试题.xlsx" \
  STAGE1_OUTPUT=data/interim/.batch_stage1_results_0929.pkl DEVICE=cuda:0 \
  ~/miniconda3/envs/py311/bin/python pipeline/batch_stage1_safety.py

# 8. Stage 2 生成答案（Safe 题数 × 后端响应时间）
INPUT_FILE="questions/0929-附件5_A5普通用户_测试题.xlsx" \
  STAGE1_OUTPUT=data/interim/.batch_stage1_results_0929.pkl \
  OUTPUT_FILE="questions/0929-附件5_A5普通用户_测试题_已回答.xlsx" \
  ~/miniconda3/envs/py311/bin/python pipeline/batch_stage2_backend.py
```

> 注意：`finetune_qwen3guard/setup_remote.sh` 是环境准备的自动化版本（conda 探测/创建、镜像源、ModelScope 缓存、超参环境变量覆盖），新服务器上可直接使用。

---

## 10. 本地产物清单（已同步回 Mac）

| 文件 | 说明 | 是否入 Git |
|---|---|---|
| `questions/0929-附件5_A5普通用户_测试题_已回答.xlsx` | 最终交付工作簿 | 已入库 |
| `finetune_qwen3guard/output/lora_v5_1/merged_model/` | 合并后模型（1.1GB） | 否（gitignore），仅本地 |
| `data/interim/.batch_stage1_results_0929.pkl` | 微调版 Stage 1 判定结果 | 否（gitignore），仅本地 |
| `data/interim/.batch_stage1_results_0929_base.pkl` | 基座版 Stage 1 判定结果 | 否（gitignore），仅本地 |
| `logs/stage1_0929.log` / `stage1_0929_base.log` / `stage2_0929.log` / `train_0929.log` | 全部运行日志 | 否（gitignore），仅本地 |
| `config.toml` | 后端模型配置（含 API Key） | 否（gitignore），仅本地 |

---

## 11. 注意事项

1. **API Key 管理**：`config.toml` 含真实密钥，已被 gitignore。任何入库文件都不得包含 Key；本报告也不含。
2. **服务器纪律**：该 GPU 服务器有其他常驻服务。只用 GPU1；不动 `legalbrain_*` 等他人 conda 环境；任务结束后清理痕迹（`~/LLM_Guard` 目录、py311 环境、shell 历史）。清理动作执行后，本报告 3.3 节中的服务器路径将失效——所有重要产物均已按第 10 节备份到本地。
3. **后端模型名**：调用时 `model` 字段必须写 `LegalBrain`。写 URL 中的服务名会报错。
4. **enable_thinking 传参位置**：必须放在 `chat_template_kwargs` 里，放请求顶层无效。
