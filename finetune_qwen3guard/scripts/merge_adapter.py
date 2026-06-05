#!/usr/bin/env python3
"""Merge LoRA adapter into full model using PEFT — CPU-only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "models/Qwen/Qwen3Guard-Gen-0.6B"
ADAPTER_DIR = "finetune_qwen3guard/output/unsloth_lora_v2/final_adapter"
MERGED_DIR = "finetune_qwen3guard/output/unsloth_lora_v2/merged_model"

print("Loading base model (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
    device_map="cpu",
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}...")
model.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Done.")
