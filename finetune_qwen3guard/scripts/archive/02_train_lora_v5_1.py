#!/usr/bin/env python3
"""
Incremental fine-tuning on v5 adapter to fix remaining 60 misclassifications.

Loads the v5 LoRA adapter and trains on:
  - 60 misclassified questions (31 from 生成类 + 29 from 拒答) → Unsafe
  - 120 random non-refusal Safe questions → Safe (to prevent forgetting)

Trains 3 epochs on this small targeted dataset.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import pickle
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

BASE_MODEL = "models/Qwen/Qwen3Guard-Gen-0.6B"
V5_ADAPTER = "finetune_qwen3guard/output/lora_v5/final_adapter"
OUTPUT_DIR = "finetune_qwen3guard/output/lora_v5_1"

random.seed(42)


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    print("=" * 60)
    print("v5.1 Incremental Training")
    print("=" * 60)

    # 1. Load misclassified questions from Stage 1 v5 results
    with open(".batch_stage1_results.pkl", "rb") as f:
        stage1_data = pickle.load(f)

    unsafe_questions = []
    for sheet in ["生成类", "拒答"]:
        results = stage1_data[sheet]["results"]
        questions = stage1_data[sheet]["questions"]
        for q, r in zip(questions, results):
            if r["safety"] == "Safe":
                unsafe_questions.append(q)

    print(f"[load] Misclassified Unsafe questions: {len(unsafe_questions)}")

    # 2. Load non-refusal Safe questions for anti-forgetting
    fei_results = stage1_data["非拒答"]["results"]
    fei_questions = stage1_data["非拒答"]["questions"]
    safe_questions = [
        q for q, r in zip(fei_questions, fei_results)
        if r["safety"] == "Safe"
    ]
    safe_sample = random.sample(safe_questions, min(120, len(safe_questions)))
    print(f"[load] Safe sample for anti-forgetting: {len(safe_sample)}")

    # 3. Build dataset
    items = []
    for q in unsafe_questions:
        items.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "Safety: Unsafe\nCategories: Unethical Acts"},
            ]
        })
    for q in safe_sample:
        items.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "Safety: Safe\nCategories: None"},
            ]
        })
    random.shuffle(items)
    print(f"[build] Total incremental samples: {len(items)}")

    # 4. Load base model + v5 adapter
    print(f"[load] Loading base model + v5 adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda")

    model = PeftModel.from_pretrained(model, V5_ADAPTER, is_trainable=True)
    print(f"[load] Adapter loaded, trainable params:")
    model.print_trainable_parameters()

    # 5. Prepare dataset
    dataset = Dataset.from_list(items)

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

    dataset = dataset.map(build_prompt_completion, remove_columns=["messages"])

    # 6. Train
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n🚀 Starting incremental training...")
    trainer.train()

    # 7. Save
    final_adapter_dir = Path(OUTPUT_DIR) / "final_adapter"
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"\n✅ v5.1 adapter saved to: {final_adapter_dir}")

    # 8. Merge
    print("\n[Extra] Merging v5.1 adapter into base model...")
    merged_dir = Path(OUTPUT_DIR) / "merged_model"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"✅ Merged model saved to: {merged_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
