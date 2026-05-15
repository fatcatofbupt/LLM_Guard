#!/usr/bin/env python3
"""
加载微调后的 Qwen3Guard-Gen-0.6B (LoRA 或合并版) 进行推理验证。

支持三种模式:
1. 加载合并后的完整模型 (--merged_model)
2. 加载基础模型 + LoRA adapter (--base_model + --adapter_path)
3. 仅加载基础模型做对比 (--base_model)

使用方式:
    # 模式1: 合并模型
    python scripts/03_inference_lora.py --merged_model finetune_qwen3guard/output/unsloth_lora/merged_model

    # 模式2: Base + LoRA
    python scripts/03_inference_lora.py --base_model models/Qwen/Qwen3Guard-Gen-0.6B \
        --adapter_path finetune_qwen3guard/output/unsloth_lora/final_adapter

    # 模式3: 仅 Base (对比基线)
    python scripts/03_inference_lora.py --base_model models/Qwen/Qwen3Guard-Gen-0.6B
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SAFETY_RE = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_model", default="",
                    help="合并后的完整模型路径（模式1）")
    ap.add_argument("--base_model", default="models/Qwen/Qwen3Guard-Gen-0.6B",
                    help="基础模型路径（模式2/3）")
    ap.add_argument("--adapter_path", default="",
                    help="LoRA adapter 路径（模式2）")
    ap.add_argument("--test_file", default="docs/test_questions.json",
                    help="原始测试题文件")
    ap.add_argument("--output_file", default="finetune_qwen3guard/output/validation_result.json",
                    help="验证结果输出路径")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--test_discriminatory_only", action="store_true",
                    help="只验证 discriminatory_content_test（快速验证）")
    return ap.parse_args()


def load_model(args):
    """根据参数加载模型。"""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.merged_model:
        print(f"Loading merged model from: {args.merged_model}")
        model_path = args.merged_model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(device)
    else:
        print(f"Loading base model from: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype="auto",
            trust_remote_code=True,
        ).to(device)

        if args.adapter_path:
            print(f"Loading LoRA adapter from: {args.adapter_path}")
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload()  # 合并后推理更快

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def parse_verdict(content: str) -> dict:
    s = SAFETY_RE.search(content)
    return {
        "safety": s.group(1) if s else None,
        "categories": CATEGORY_RE.findall(content),
    }


@torch.no_grad()
def batch_moderate(model, tokenizer, device, questions: list[str], max_new_tokens: int) -> list[dict]:
    messages = [[{"role": "user", "content": q}] for q in questions]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = enc.input_ids.size(1)
    results = []
    for i in range(outputs.size(0)):
        decoded = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
        results.append({"raw": decoded, **parse_verdict(decoded)})
    return results


def main():
    args = parse_args()
    model, tokenizer, device = load_model(args)

    # 加载测试数据
    with open(args.test_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 只取 discriminatory_content_test 做快速验证
    if args.test_discriminatory_only:
        test_sets = {"discriminatory_content_test": all_data.get("discriminatory_content_test", [])}
    else:
        test_sets = {
            "discriminatory_content_test": all_data.get("discriminatory_content_test", []),
            "should_refuse": [
                q for q in all_data.get("should_refuse", [])
                if q.get("primary_category") == "A.2包含歧视性内容"
            ],
        }

    output_results = {}
    total_questions = 0
    total_correct = 0

    for test_name, questions in test_sets.items():
        print(f"\n🔍 Evaluating: {test_name} ({len(questions)} questions)")
        results = []
        correct = 0

        # 分批推理
        for i in range(0, len(questions), args.batch_size):
            batch = questions[i:i + args.batch_size]
            batch_qs = [q["question"] for q in batch]
            preds = batch_moderate(model, tokenizer, device, batch_qs, args.max_new_tokens)

            for q, pred in zip(batch, preds):
                # 判定是否正确
                expected = "Unsafe"  # 这些测试集中的题目都是有害的
                is_correct = pred["safety"] == expected
                if is_correct:
                    correct += 1

                results.append({
                    "question": q["question"],
                    "primary_category": q.get("primary_category", ""),
                    "secondary_category": q.get("secondary_category", ""),
                    "expected": expected,
                    "predicted": pred["safety"],
                    "categories": pred["categories"],
                    "raw": pred["raw"],
                    "correct": is_correct,
                })

        accuracy = correct / len(questions) * 100 if questions else 0
        print(f"    Accuracy: {correct}/{len(questions)} = {accuracy:.1f}%")

        output_results[test_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(questions),
            "details": results,
        }
        total_questions += len(questions)
        total_correct += correct

    overall = total_correct / total_questions * 100 if total_questions else 0
    print(f"\n📊 Overall Accuracy: {total_correct}/{total_questions} = {overall:.1f}%")

    # 保存结果
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {args.output_file}")

    # 打印错误样例
    for test_name, data in output_results.items():
        wrong = [r for r in data["details"] if not r["correct"]]
        if wrong:
            print(f"\n❌ {test_name} - Wrong samples (showing first 5):")
            for r in wrong[:5]:
                print(f"    Q: {r['question'][:80]}...")
                print(f"       Expected: {r['expected']}, Got: {r['predicted']} ({r['raw'][:60]})")


if __name__ == "__main__":
    main()
