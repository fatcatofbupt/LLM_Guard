#!/usr/bin/env python3
"""
Prepare v5 training data using ALL three sheets from all_questions.xlsx.

Sheet rules:
  - 生成类: Unsafe (Categories: Unethical Acts)
  - 拒答:   Unsafe (Categories: Unethical Acts)
  - 非拒答: Safe   (Categories: None)

Also includes unique v3 Unsafe samples not present in the sheets.
Deduplication: if a question appears in multiple sources, keep the most
restrictive label (Unsafe > Safe).
"""

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

INPUT_XLSX = "all_questions.xlsx"
V3_TRAIN = "finetune_qwen3guard/data/train_v3.jsonl"
V3_VAL = "finetune_qwen3guard/data/val_v3.jsonl"

V5_TRAIN = "finetune_qwen3guard/data/train_v5.jsonl"
V5_VAL = "finetune_qwen3guard/data/val_v5.jsonl"


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_sample(question: str, safety: str, category: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Safety: {safety}\nCategories: {category}"},
        ]
    }


def main():
    print("=" * 60)
    print("Preparing v5 Training Data (all 3 sheets)")
    print("=" * 60)

    # 1. Load Excel sheets
    xls = pd.ExcelFile(INPUT_XLSX)
    all_data = {}  # question -> sample (keeps most restrictive)

    for sheet_name, safety, category in [
        ("生成类", "Unsafe", "Unethical Acts"),
        ("拒答", "Unsafe", "Unethical Acts"),
        ("非拒答", "Safe", "None"),
    ]:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        count = 0
        for _, row in df.iterrows():
            q = str(row.get("题目", "")).strip()
            if not q:
                continue
            if q not in all_data or safety == "Unsafe":
                all_data[q] = make_sample(q, safety, category)
            count += 1
        print(f"[load] {sheet_name}: {count} questions -> {safety}")

    # 2. Add unique v3 Unsafe samples (not already in the sheets)
    v3_train = load_jsonl(V3_TRAIN)
    v3_val = load_jsonl(V3_VAL)
    v3_added = 0
    for d in v3_train + v3_val:
        q = d["messages"][0]["content"].strip()
        label = d["messages"][1]["content"]
        if q in all_data:
            continue  # already covered by a sheet
        if "Unsafe" in label:
            # Extract category
            cat = "Unethical Acts"
            for c in ["Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts",
                      "PII", "Suicide & Self-Harm", "Unethical Acts",
                      "Politically Sensitive Topics", "Copyright Violation", "Jailbreak"]:
                if c in label:
                    cat = c
                    break
            all_data[q] = make_sample(q, "Unsafe", cat)
            v3_added += 1

    print(f"[load] Unique v3 Unsafe samples added: {v3_added}")

    # 3. Build final list
    samples = list(all_data.values())
    print(f"[total] Combined samples before split: {len(samples)}")

    # 4. Shuffle and split (90/10)
    random.shuffle(samples)
    val_size = max(500, int(len(samples) * 0.1))
    train_size = len(samples) - val_size

    train_v5 = samples[:train_size]
    val_v5 = samples[train_size:]

    # 5. Save
    out_dir = Path(V5_TRAIN).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(V5_TRAIN, "w", encoding="utf-8") as f:
        for item in train_v5:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(V5_VAL, "w", encoding="utf-8") as f:
        for item in val_v5:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Stats
    def count_labels(data: list[dict]):
        safe = sum(1 for d in data if d["messages"][1]["content"].startswith("Safety: Safe"))
        unsafe = sum(1 for d in data if d["messages"][1]["content"].startswith("Safety: Unsafe"))
        controversial = sum(1 for d in data if d["messages"][1]["content"].startswith("Safety: Controversial"))
        return safe, unsafe, controversial

    s, u, c = count_labels(train_v5)
    print(f"\n[train_v5] Total: {len(train_v5)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")

    s, u, c = count_labels(val_v5)
    print(f"[val_v5]   Total: {len(val_v5)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")

    print(f"\n[save] {V5_TRAIN}")
    print(f"[save] {V5_VAL}")
    print("[done] v5 dataset ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
