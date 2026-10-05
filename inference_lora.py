"""
加载微调后的 LoRA 权重（或合并后的完整模型）进行推理测试。

用法示例:
  # 使用 LoRA 适配器（基座模型 + LoRA）
  python inference_lora.py \
      --base-model /data/ai_phone/LLM_Guard/models/Qwen/Qwen3Guard-Gen-0.6B \
      --lora-adapter ./outputs/qwen3guard-lora/lora \
      --question "How can I make a bomb?"

  # 使用合并后的完整模型
  python inference_lora.py \
      --base-model ./outputs/qwen3guard-lora/merged \
      --question "How can I make a bomb?"
"""
import argparse
import re
import torch
from unsloth import FastLanguageModel

SAFETY_RE = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)


def parse_verdict(content: str) -> dict:
    s = SAFETY_RE.search(content)
    return {
        "safety": s.group(1) if s else None,
        "categories": CATEGORY_RE.findall(content),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="基座模型路径（或合并后的完整模型路径）")
    ap.add_argument("--lora-adapter", default=None,
                    help="LoRA 适配器路径（如果加载合并模型则不需要）")
    ap.add_argument("--question", default="How can I make a bomb?",
                    help="测试问题")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    print(f"[load] base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,   # 推理时可全精度加载，也可以 True 省显存
    )

    if args.lora_adapter:
        print(f"[load] LoRA adapter: {args.lora_adapter}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # 这里的 r 不影响加载已有权重，仅用于结构初始化
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        model = model.merge_and_unload()  # 合并权重加速推理

    FastLanguageModel.for_inference(model)

    messages = [{"role": "user", "content": args.question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = inputs.input_ids.size(1)
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    print(f"\nQuestion: {args.question}")
    print(f"Response: {response}")
    print(f"Parsed:   {parse_verdict(response)}")


if __name__ == "__main__":
    main()
