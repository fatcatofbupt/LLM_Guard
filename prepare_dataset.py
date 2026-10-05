"""
将已有标注数据转换为 Unsloth 训练所需的 sharegpt / chatml 格式 JSON。

输入: docs/test_questions_answered_qwen3guard.json
输出: data/train.json   (sharegpt 格式，可直接喂给 unsloth)

sharegpt 格式示例:
[
  {
    "conversations": [
      {"from": "human", "value": "用户问题..."},
      {"from": "gpt",   "value": "Safety: ...\nCategories: ..."}
    ]
  },
  ...
]
"""
import argparse
import json
from pathlib import Path


def convert(input_path: str, output_path: str, val_ratio: float = 0.05):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for category, items in raw.items():
        for it in items:
            question = it.get("question", "")
            answer = it.get("answer", "")
            if not question or not answer:
                continue
            records.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
                # 保留原始元信息，方便溯源
                "source_category": category,
                "primary_category": it.get("primary_category", ""),
                "secondary_category": it.get("secondary_category", ""),
            })

    # 简单随机拆分 train / val
    import random
    random.seed(42)
    random.shuffle(records)

    split_idx = int(len(records) * (1 - val_ratio))
    train = records[:split_idx]
    val = records[split_idx:]

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    val_path = out_dir / "val.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"[done] train={len(train)}  val={len(val)}")
    print(f"       train -> {output_path}")
    print(f"       val   -> {val_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./docs/test_questions_answered_qwen3guard.json")
    ap.add_argument("--output", default="./data/train.json")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()
    convert(args.input, args.output, args.val_ratio)


if __name__ == "__main__":
    main()
