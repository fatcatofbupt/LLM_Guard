#!/usr/bin/env python3
"""
将 Unsloth/PEFT 训练好的 LoRA 权重合并到基础模型中，导出为完整模型。

用途:
- 简化部署（不需要再加载 adapter）
- 提高推理速度
- 转换为 GGUF / vLLM 等格式前必须合并

用法:
    python scripts/04_merge_lora.py \
        --base_model models/Qwen/Qwen3Guard-Gen-0.6B \
        --adapter_path finetune_qwen3guard/output/unsloth_lora/final_adapter \
        --output_dir finetune_qwen3guard/output/unsloth_lora/merged_model
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return ap.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[1/3] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    print(f"[2/3] Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print(f"[3/3] Merging and unloading...")
    merged_model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Done! Merged model saved.")


if __name__ == "__main__":
    main()
