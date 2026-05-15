"""
Run Qwen3Guard-Gen-0.6B on questions in docs/test_questions.json and write
each item back out with the model's safety verdict attached.

Output schema per item = original fields + {
    "answer":     raw model output (e.g. "Safety: Unsafe\\nCategories: Violent"),
    "safety":     "Safe" | "Unsafe" | "Controversial" | None,
    "categories": list[str],
}

Features:
  - Batched generation
  - Incremental save (resume on crash / re-run)
  - --limit N  to smoke-test (N items per top-level category)
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto").to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False)
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = enc.input_ids.size(1)
    return [
        tok.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        for i in range(out.size(0))
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--input", default="./docs/test_questions.json")
    ap.add_argument("--output", default="./docs/test_questions_answered_qwen3guard.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per top-level category (useful for smoke tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"[setup] device={device}, batch_size={args.batch_size}, limit={args.limit}")

    with open(args.input) as f:
        data = json.load(f)

    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            done = json.load(f)
        print(f"[resume] loaded existing output: {out_path}")
    else:
        done = {k: [] for k in data}

    tok, model = load_model(args.model, device)

    for cat, items in data.items():
        if args.limit is not None:
            items = items[: args.limit]
        already = len(done.get(cat, []))
        if already >= len(items):
            print(f"[{cat}] already complete ({already}/{len(items)}), skipping")
            continue

        remaining = items[already:]
        print(f"[{cat}] {already}/{len(items)} done, processing {len(remaining)} more")

        for start in tqdm(range(0, len(remaining), args.batch_size), desc=cat):
            batch = remaining[start : start + args.batch_size]
            raws = generate_batch(
                tok, model,
                prompts=[it["question"] for it in batch],
                max_new_tokens=args.max_new_tokens,
            )
            for it, raw in zip(batch, raws):
                rec = dict(it)
                rec["answer"] = raw
                rec.update(parse_verdict(raw))
                done.setdefault(cat, []).append(rec)

            with open(out_path, "w") as f:
                json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
