#!/usr/bin/env python3
"""
Qwen3Guard-Gen-0.6B LoRA 微调脚本

Follows the exact same pattern as 02_train_unsloth.py:
  1. Build prompt + completion pairs
  2. Use SFTTrainer for loss masking (only completion is learned)
  3. Save LoRA adapter + optional merged model
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


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
    ap.add_argument("--save_merged", action="store_true", default=False)
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

    print(f"[1/5] Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"       Using device: {device}")
    model = model.to(device)

    print(f"[2/5] Loading datasets")
    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    print(f"       Train: {len(train_raw)}, Val: {len(val_raw)}")

    train_dataset = Dataset.from_list(train_raw)
    val_dataset = Dataset.from_list(val_raw)

    # ------------------------------------------------------------------
    # Build prompt + completion pairs (same as original working script)
    # ------------------------------------------------------------------
    print(f"[3/5] Preparing prompt + completion pairs")

    def build_prompt_completion(example):
        user_msg = [m for m in example["messages"] if m["role"] == "user"]
        assistant_msg = [m for m in example["messages"] if m["role"] == "assistant"]
        prompt = tokenizer.apply_chat_template(
            user_msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = assistant_msg[0]["content"] if assistant_msg else ""
        return {"prompt": prompt, "completion": completion}

    train_dataset = train_dataset.map(build_prompt_completion, remove_columns=["messages"])
    val_dataset = val_dataset.map(build_prompt_completion, remove_columns=["messages"])

    sample = train_dataset[0]
    print(f"       Prompt length: {len(sample['prompt'])} chars")
    print(f"       Completion: {sample['completion'][:100]}...")

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    print(f"[4/5] Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training args
    # ------------------------------------------------------------------
    print(f"[5/5] Setting up SFTTrainer")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n🚀 Starting training...")
    trainer.train()

    # Save adapter
    final_adapter_dir = Path(args.output_dir) / "final_adapter"
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\n✅ LoRA adapter saved to: {final_adapter_dir}")

    # Optional: merge
    if args.save_merged:
        print("\n[Extra] Merging LoRA weights into base model...")
        merged_dir = Path(args.output_dir) / "merged_model"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"✅ Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
