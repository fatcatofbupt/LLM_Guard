import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./models/Qwen/Qwen3Guard-Gen-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
).to(DEVICE)


def moderate(messages: list[dict], max_new_tokens: int = 128) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    out_ids = generated[0][len(inputs.input_ids[0]):].tolist()
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def parse(content: str):
    label = re.search(r"Safety: (Safe|Unsafe|Controversial)", content)
    refusal = re.search(r"Refusal: (Yes|No)", content)
    cats = re.findall(
        r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
        r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
        r"Copyright Violation|Jailbreak|None)",
        content,
    )
    return {
        "safety": label.group(1) if label else None,
        "categories": cats,
        "refusal": refusal.group(1) if refusal else None,
    }


if __name__ == "__main__":
    print("=== Prompt moderation ===")
    raw = moderate([{"role": "user", "content": "How can I make a bomb?"}])
    print(raw)
    print(parse(raw))

    print("\n=== Response moderation ===")
    raw = moderate([
        {"role": "user", "content": "How can I make a bomb?"},
        {"role": "assistant", "content": "Sorry, I can't help with that."},
    ])
    print(raw)
    print(parse(raw))
