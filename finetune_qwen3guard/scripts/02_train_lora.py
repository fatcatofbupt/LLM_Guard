#!/usr/bin/env python3
"""
Qwen3Guard-Gen-0.6B LoRA 微调脚本

依赖:
    pip install transformers>=4.51.0 accelerate peft trl

使用 SFTTrainer (trl) + PEFT LoRA 进行高效微调。
0.6B 模型非常小，单卡 8GB 显存即可 LoRA 微调，CPU/MPS 也能跑（较慢）。
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--train_file", default="finetune_qwen3guard/data/train.jsonl")
    ap.add_argument("--val_file", default="finetune_qwen3guard/data/val.jsonl")
    ap.add_argument("--output_dir", default="finetune_qwen3guard/output/lora")
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--save_merged", action="store_true", default=False,
                    help="训练结束后将 LoRA 权重合并到基础模型并保存")
    return ap.parse_args()


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. 加载 Tokenizer & Model
    # ------------------------------------------------------------------
    print(f"[1/5] Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Qwen3Guard 使用左填充（生成模型标准做法）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        # 0.6B 很小，不需要 quantization
    )

    # 自动选择设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"       Using device: {device}")
    model = model.to(device)

    # ------------------------------------------------------------------
    # 2. 加载数据集
    # ------------------------------------------------------------------
    print(f"[2/5] Loading datasets")
    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    print(f"       Train: {len(train_raw)}, Val: {len(val_raw)}")

    # 转换为 HuggingFace Dataset
    train_dataset = Dataset.from_list(train_raw)
    val_dataset = Dataset.from_list(val_raw)

    # ------------------------------------------------------------------
    # 3. LoRA 配置
    # ------------------------------------------------------------------
    print(f"[3/5] Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ------------------------------------------------------------------
    # 4. 训练参数
    # ------------------------------------------------------------------
    print(f"[4/5] Setting up training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16 and (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=not args.bf16 and device == "cuda",
        report_to="none",
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. SFTTrainer
    # ------------------------------------------------------------------
    # 使用 formatting_func 将 messages 转换为模型输入文本
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # response_template: 让 loss 只计算在 assistant 的 Safety 判定输出上
    # Qwen3Guard 模板会在 assistant 后自动生成 <think>\n\n</think>\n 然后接 Safety: ...
    response_template = "Safety:"
    print(f"       Response template for loss masking: '{response_template}'")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_func,
        data_collator=collator,
    )

    # ------------------------------------------------------------------
    # 6. 开始训练
    # ------------------------------------------------------------------
    print(f"[5/5] Starting training...")
    trainer.train()

    # 保存最终 LoRA adapter
    final_adapter_dir = Path(args.output_dir) / "final_adapter"
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\n✅ LoRA adapter saved to: {final_adapter_dir}")

    # ------------------------------------------------------------------
    # 7. 可选：合并并保存完整模型
    # ------------------------------------------------------------------
    if args.save_merged:
        print("\n[Extra] Merging LoRA weights into base model...")
        merged_dir = Path(args.output_dir) / "merged_model"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"✅ Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
