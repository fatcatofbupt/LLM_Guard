#!/usr/bin/env python3
"""
Qwen3Guard-Gen-0.6B Unsloth LoRA 微调脚本

优势（vs 标准 Transformers + PEFT）：
- 2x 训练速度
- 70% 显存节省
- 支持更长的上下文（8x）

环境要求:
- Linux + NVIDIA GPU (CUDA >= 11.8)
- Python 3.10+
- 显存: bf16 LoRA 约需 3-4GB (0.6B 模型非常小)

安装:
    pip install unsloth transformers trl datasets accelerate

运行:
    python finetune_qwen3guard/scripts/02_train_unsloth.py
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Unsloth 必须在 import transformers 之后导入
from unsloth import FastModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--train_file", default="finetune_qwen3guard/data/train.jsonl")
    ap.add_argument("--val_file", default="finetune_qwen3guard/data/val.jsonl")
    ap.add_argument("--output_dir", default="finetune_qwen3guard/output/unsloth_lora")
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_merged", action="store_true", default=False,
                    help="训练结束后合并 LoRA 权重并保存完整模型")
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
    # 1. 使用 Unsloth FastModel 加载模型
    # ------------------------------------------------------------------
    print(f"[1/5] Loading model with Unsloth: {args.model_path}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,      # 0.6B 很小，不需要 4-bit
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
    )
    print(f"       Model dtype: {model.dtype}")
    print(f"       Device: {next(model.parameters()).device}")

    # ------------------------------------------------------------------
    # 2. 附加 LoRA Adapter
    # ------------------------------------------------------------------
    print(f"[2/5] Attaching LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 优化的 gradient checkpointing
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
        # 对 Qwen3 推荐的目标模块
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. 加载数据集
    # ------------------------------------------------------------------
    print(f"[3/5] Loading datasets")
    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    print(f"       Train: {len(train_raw)}, Val: {len(val_raw)}")

    train_dataset = Dataset.from_list(train_raw)
    val_dataset = Dataset.from_list(val_raw)

    # ------------------------------------------------------------------
    # 4. 数据格式化: messages -> 应用 chat_template
    # ------------------------------------------------------------------
    print(f"[4/5] Preparing data formatter")

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # 用一个样本验证输出格式
    sample = formatting_func(train_raw[0])
    print(f"       Sample input length: {len(sample)} chars")
    print(f"       Sample preview:\n{sample[:400]}...")

    # ------------------------------------------------------------------
    # 5. 训练参数
    # ------------------------------------------------------------------
    print(f"[5/5] Configuring trainer")
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
        bf16=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_func,
        # Unsloth 会自动处理 completion-only loss masking
        # 对于 Qwen3Guard，assistant 输出从 "Safety:" 开始
        dataset_text_field=None,
    )

    # ------------------------------------------------------------------
    # 6. 训练
    # ------------------------------------------------------------------
    print(f"\n🚀 Starting training...")
    trainer_stats = trainer.train()

    print(f"\n📊 Training completed!")
    print(f"    Final train loss: {trainer_stats.training_loss:.4f}")
    print(f"    Training time:    {trainer_stats.metrics.get('train_runtime', 0)/60:.1f} min")

    # ------------------------------------------------------------------
    # 7. 保存 LoRA Adapter
    # ------------------------------------------------------------------
    adapter_dir = Path(args.output_dir) / "final_adapter"
    print(f"\n💾 Saving LoRA adapter to: {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ------------------------------------------------------------------
    # 8. 可选: 合并并保存完整模型 (FP16)
    # ------------------------------------------------------------------
    if args.save_merged:
        merged_dir = Path(args.output_dir) / "merged_model"
        print(f"\n🔧 Merging LoRA into base model and saving to: {merged_dir}")
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",  # 合并为 fp16 完整模型
        )
        print(f"✅ Merged model saved!")

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
