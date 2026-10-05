"""
使用 Unsloth 微调 Qwen3Guard-Gen-0.6B（LoRA）。

依赖:
  pip install unsloth
  pip install "trl<0.15.0" peft accelerate bitsandbytes

参考:
  https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama
"""
import argparse
import json
import os
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Unsloth 相关导入
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# 默认超参
# ---------------------------------------------------------------------------
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_WARMUP_STEPS = 5
DEFAULT_MAX_STEPS = 60
DEFAULT_LOGGING_STEPS = 1
DEFAULT_OUTPUT_DIR = "./outputs/qwen3guard-lora"


def load_data(train_path: str, val_path: str = None):
    """加载 sharegpt 格式的 JSON，并转成 Dataset。"""
    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)

    train_ds = Dataset.from_list(train_raw)
    val_ds = None
    if val_path and Path(val_path).exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
        val_ds = Dataset.from_list(val_raw)
    return train_ds, val_ds


def formatting_func(examples, tokenizer):
    """
    将 sharegpt 格式的 conversations 转换为模型输入文本。
    使用 tokenizer 的 apply_chat_template 能力。
    """
    texts = []
    for conv in examples["conversations"]:
        # conv 是 [{"from": "human"/"gpt", "value": "..."}, ...]
        messages = []
        for turn in conv:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # 训练时不需要生成提示
        )
        texts.append(text)
    return {"text": texts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/data/ai_phone/LLM_Guard/models/Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--train-data", default="./data/train.json")
    ap.add_argument("--val-data", default="./data/val.json")
    ap.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    ap.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--num-epochs", type=int, default=None,
                    help="如果设置，优先使用 epoch 而不是 max_steps")
    ap.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--save-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # 1. 加载模型 & Tokenizer（Unsloth 优化版）
    # -----------------------------------------------------------------------
    print(f"[1/6] Loading model from {args.model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,               # None = 自动检测（A100 -> bf16, 其他 -> fp16）
        load_in_4bit=True,        # 4-bit QLoRA，显存友好
    )

    # 设置 chat_template（Qwen3 默认就是 chatml，但显式指定更安全）
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3",    # Unsloth 内置模板名
    )

    # -----------------------------------------------------------------------
    # 2. 添加 LoRA 适配器
    # -----------------------------------------------------------------------
    print(f"[2/6] Adding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha}) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 显存优化，速度 2x
        random_state=args.seed,
    )

    # -----------------------------------------------------------------------
    # 3. 加载数据
    # -----------------------------------------------------------------------
    print(f"[3/6] Loading dataset: {args.train_data} ...")
    train_ds, val_ds = load_data(args.train_data, args.val_data)

    # 标准化 sharegpt 格式（human/gpt -> user/assistant）
    train_ds = standardize_sharegpt(train_ds)
    if val_ds:
        val_ds = standardize_sharegpt(val_ds)

    # 格式化
    train_ds = train_ds.map(
        lambda x: formatting_func(x, tokenizer),
        batched=True,
        desc="Formatting train",
    )
    if val_ds:
        val_ds = val_ds.map(
            lambda x: formatting_func(x, tokenizer),
            batched=True,
            desc="Formatting val",
        )

    print(f"       train samples: {len(train_ds)}")
    if val_ds:
        print(f"       val   samples: {len(val_ds)}")

    # -----------------------------------------------------------------------
    # 4. 训练参数
    # -----------------------------------------------------------------------
    print(f"[4/6] Setting up training arguments ...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.num_epochs is None else -1,
        num_train_epochs=args.num_epochs if args.num_epochs else None,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(output_dir),
        report_to="none",            # 如需 wandb/tensorboard 可改
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.save_steps if val_ds else None,
        load_best_model_at_end=True if val_ds else False,
    )

    # -----------------------------------------------------------------------
    # 5. 开始训练
    # -----------------------------------------------------------------------
    print(f"[5/6] Start training ...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,               # 短序列可设为 True 加速
        args=training_args,
    )

    trainer_stats = trainer.train()
    print(f"[train] final loss: {trainer_stats.training_loss:.4f}")

    # -----------------------------------------------------------------------
    # 6. 保存模型
    # -----------------------------------------------------------------------
    print(f"[6/6] Saving adapters & merged model ...")

    # 6a. 只保存 LoRA 权重
    lora_dir = output_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"       LoRA -> {lora_dir}")

    # 6b. 合并并保存完整模型（可选，显存足够时执行）
    try:
        merged_dir = output_dir / "merged"
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"       Merged 16-bit -> {merged_dir}")
    except Exception as e:
        print(f"       Skip merged save: {e}")

    print("[done] All checkpoints saved.")


if __name__ == "__main__":
    main()
