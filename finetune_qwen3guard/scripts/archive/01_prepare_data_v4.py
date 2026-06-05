#!/usr/bin/env python3
"""
Prepare v4 training data by adding all non-refusal (Safe) questions
to the existing v3 dataset.

Non-refusal questions come from all_questions.xlsx sheet "非拒答".
They are labeled Safety: Safe, Categories: None.
"""

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
INPUT_XLSX = "all_questions.xlsx"
SHEET_NAME = "非拒答"

V3_TRAIN = "finetune_qwen3guard/data/train_v3.jsonl"
V3_VAL = "finetune_qwen3guard/data/val_v3.jsonl"

V4_TRAIN = "finetune_qwen3guard/data/train_v4.jsonl"
V4_VAL = "finetune_qwen3guard/data/val_v4.jsonl"

NUM_NONREFUSAL_VAL = 500

# ------------------------------------------------------------------
# Load v3 data
# ------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    print("=" * 60)
    print("Preparing v4 Training Data")
    print("=" * 60)

    # 1. Load v3 train / val
    v3_train = load_jsonl(V3_TRAIN)
    v3_val = load_jsonl(V3_VAL)
    print(f"[load] v3 train: {len(v3_train)}, v3 val: {len(v3_val)}")

    # 2. Load non-refusal questions from Excel
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    nonrefusal_questions = []
    for _, row in df.iterrows():
        q = str(row.get("题目", "")).strip()
        if q:
            nonrefusal_questions.append(q)

    print(f"[load] Non-refusal questions from '{SHEET_NAME}': {len(nonrefusal_questions)}")

    # 3. Exclude questions that already exist in v3 (avoid duplicate / contradictory labels)
    v3_questions = set()
    for d in v3_train + v3_val:
        v3_questions.add(d["messages"][0]["content"].strip())

    unique_nonrefusal = [q for q in nonrefusal_questions if q not in v3_questions]
    excluded = len(nonrefusal_questions) - len(unique_nonrefusal)
    print(f"[dedup] Excluded {excluded} questions already in v3 train/val")
    print(f"[dedup] Unique non-refusal questions: {len(unique_nonrefusal)}")

    # 4. Build Safe samples
    random.shuffle(unique_nonrefusal)
    val_questions = unique_nonrefusal[:NUM_NONREFUSAL_VAL]
    train_questions = unique_nonrefusal[NUM_NONREFUSAL_VAL:]

    nonrefusal_train = [
        {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "Safety: Safe\nCategories: None"},
            ]
        }
        for q in train_questions
    ]
    nonrefusal_val = [
        {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "Safety: Safe\nCategories: None"},
            ]
        }
        for q in val_questions
    ]

    print(f"[build] Non-refusal train: {len(nonrefusal_train)}, val: {len(nonrefusal_val)}")

    # 4. Combine
    train_v4 = v3_train + nonrefusal_train
    val_v4 = v3_val + nonrefusal_val
    random.shuffle(train_v4)
    random.shuffle(val_v4)

    # 5. Save
    out_dir = Path(V4_TRAIN).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(V4_TRAIN, "w", encoding="utf-8") as f:
        for item in train_v4:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(V4_VAL, "w", encoding="utf-8") as f:
        for item in val_v4:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Stats
    def count_labels(data: list[dict]):
        safe = sum(1 for d in data if "Safe" in d["messages"][1]["content"])
        unsafe = sum(1 for d in data if "Unsafe" in d["messages"][1]["content"])
        controversial = sum(1 for d in data if "Controversial" in d["messages"][1]["content"])
        return safe, unsafe, controversial

    s, u, c = count_labels(train_v4)
    print(f"\n[train_v4] Total: {len(train_v4)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")

    s, u, c = count_labels(val_v4)
    print(f"[val_v4]   Total: {len(val_v4)}, Safe: {s}, Unsafe: {u}, Controversial: {c}")

    print(f"\n[save] {V4_TRAIN}")
    print(f"[save] {V4_VAL}")
    print("[done] v4 dataset ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
