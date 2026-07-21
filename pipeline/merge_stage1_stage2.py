#!/usr/bin/env python3
"""
Merge Stage 1 (safety classification) and Stage 2 (model answers) into a final result.

Logic:
  - Unsafe rows (from stage1 safety_label): 模型的答案 = refusal message
  - Safe rows (from stage1 safety_label): 模型的答案 = corresponding Stage 2 answer
  - Any Safe rows with empty Stage 2 answers are listed for the user.

Usage:
    conda run -n guard_train python3 pipeline/merge_stage1_stage2.py
"""

import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
STAGE1_FILE = "questions/stage1_v5_1_raw.xlsx"
STAGE2_FILE = "data/reference/附件5 天津测试题_合并去重_已回答.xlsx"
OUTPUT_FILE = "data/raw/附件5_天津测试题_最终合并结果.xlsx"

REFUSAL_MESSAGE = "根据政策和法律规定，我无法回答你的问题。"

SHEETS = ["生成类", "拒答", "非拒答"]


def merge_sheet(sheet_name: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Merge one sheet from Stage 1 and Stage 2.
    Returns (merged_df, list_of_safe_empty_rows).
    """
    print(f"\n[merge] Processing sheet: '{sheet_name}' ...")

    df1 = pd.read_excel(STAGE1_FILE, sheet_name=sheet_name)
    df2 = pd.read_excel(STAGE2_FILE, sheet_name=sheet_name)

    total = len(df1)
    print(f"  Stage1 rows: {len(df1)}, Stage2 rows: {len(df2)}")

    # Build a lookup from Stage2: question -> answer
    stage2_lookup = {}
    for _, row in df2.iterrows():
        q = str(row.get("题目", "")).strip()
        if q:
            stage2_lookup[q] = str(row.get("模型的答案", "")).strip()

    # Result containers
    merged_answers = []
    safe_empty_rows = []

    unsafe_count = 0
    safe_with_answer = 0
    safe_empty_count = 0

    for idx, row in df1.iterrows():
        question = str(row.get("题目", "")).strip()
        verdict = str(row.get("safety_label", "")).strip()

        if verdict == "Unsafe":
            merged_answers.append(REFUSAL_MESSAGE)
            unsafe_count += 1
        elif verdict == "Safe":
            stage2_answer = stage2_lookup.get(question, "")
            if stage2_answer and stage2_answer.lower() != "nan":
                merged_answers.append(stage2_answer)
                safe_with_answer += 1
            else:
                merged_answers.append("")
                safe_empty_count += 1
                safe_empty_rows.append({
                    "sheet": sheet_name,
                    "index": idx,
                    "question": question[:100],
                })
        else:
            # Unknown or empty verdict
            merged_answers.append("")

    # Build output DataFrame based on Stage 2 structure + stage1 metadata
    out_df = df2.copy()

    # Ensure required columns exist
    if "模型的答案" not in out_df.columns:
        out_df["模型的答案"] = ""
    if "safety_label" not in out_df.columns:
        out_df["safety_label"] = ""
    if "categories" not in out_df.columns:
        out_df["categories"] = ""

    # Fill merged answers
    out_df["模型的答案"] = merged_answers

    # Add safety_label and categories from Stage 1
    out_df["safety_label"] = df1["safety_label"].values
    if "categories" in df1.columns:
        out_df["categories"] = df1["categories"].values

    print(f"  Unsafe -> refusal: {unsafe_count}")
    print(f"  Safe   -> stage2 answer: {safe_with_answer}")
    print(f"  Safe   -> empty (listed): {safe_empty_count}")

    return out_df, safe_empty_rows


def main():
    print("=" * 60)
    print("Merge Stage 1 + Stage 2 into Final Result")
    print(f"Stage1: {STAGE1_FILE}")
    print(f"Stage2: {STAGE2_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    all_safe_empty = []
    sheet_results = {}

    for sheet in SHEETS:
        merged_df, safe_empty = merge_sheet(sheet)
        sheet_results[sheet] = merged_df
        all_safe_empty.extend(safe_empty)

    # Write output
    print(f"\n[write] Saving to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet, df in sheet_results.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"  Sheet '{sheet}' -> {len(df)} rows")

    # Report safe-empty rows
    if all_safe_empty:
        print(f"\n{'=' * 60}")
        print(f"Safe rows with EMPTY Stage 2 answers ({len(all_safe_empty)} total):")
        print(f"{'=' * 60}")
        for item in all_safe_empty:
            print(f"  [{item['sheet']:4s}] idx={item['index']:5d} | {item['question']}")
    else:
        print(f"\n{'=' * 60}")
        print("No Safe rows with empty Stage 2 answers found.")
        print(f"{'=' * 60}")

    print(f"\nFinished! Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
