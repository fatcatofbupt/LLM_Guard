#!/usr/bin/env python3
"""
Stage 2 for 0721 test questions.

Reads Stage 1 safety results for `questions/0721-附件5_测试题.xlsx`,
copies existing 非拒答 answers from `data/reference/附件5 天津测试题_合并去重_已回答.xlsx`,
and fills refusal phrases for blocked 生成类 / 拒答 rows.

Does NOT call the backend API.

Usage:
    conda run -n py311 python3 pipeline/process_0721_stage2.py
"""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
STAGE1_OUTPUT = os.environ.get("STAGE1_OUTPUT", "data/interim/.batch_stage1_results_0721.pkl")
INPUT_FILE = os.environ.get("INPUT_FILE", "questions/0721-附件5_测试题.xlsx")
OLD_ANSWERED_FILE = os.environ.get(
    "OLD_ANSWERED_FILE", "data/reference/附件5 天津测试题_合并去重_已回答.xlsx"
)
OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE", "questions/0721-附件5_测试题_已回答.xlsx"
)

GENERATION_REFUSAL = "对不起，我还没有学会回答这个问题。如果你有其他问题，我非常乐意为你提供帮助"
REJECTION_REFUSAL = "根据政策和法律规定，我无法回答你的问题。"

# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)


def load_old_answers(path: str) -> dict[str, dict[str, str]]:
    """Build {sheet_name: {question_text: answer}} from the old answered file."""
    logger.info(f"[load] Loading old answers from {path} ...")
    xls = pd.ExcelFile(path)
    lookup: dict[str, dict[str, str]] = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "题目" not in df.columns or "模型的答案" not in df.columns:
            logger.warning(f"[{sheet_name}] Missing required columns, skipping.")
            continue
        sheet_lookup = {}
        for _, row in df.iterrows():
            q = str(row.get("题目", "")).strip()
            a = row.get("模型的答案", "")
            if q and pd.notna(a):
                sheet_lookup[q] = str(a)
        lookup[sheet_name] = sheet_lookup
        logger.info(f"[{sheet_name}] Loaded {len(sheet_lookup)} answered questions.")
    return lookup


def is_blocked(result: dict) -> bool:
    """Return True if the safety verdict is Unsafe or Controversial."""
    safety = result.get("safety")
    return safety in ("Unsafe", "Controversial")


def main():
    logger.info("=" * 60)
    logger.info("Stage 2: Build 0721 answered file")
    logger.info(f"Stage1 data: {STAGE1_OUTPUT}")
    logger.info(f"Input:       {INPUT_FILE}")
    logger.info(f"Old answers: {OLD_ANSWERED_FILE}")
    logger.info(f"Output:      {OUTPUT_FILE}")
    logger.info("=" * 60)

    if not os.path.exists(STAGE1_OUTPUT):
        logger.error(f"Stage 1 results not found: {STAGE1_OUTPUT}")
        logger.error("Please run batch_stage1_safety.py first.")
        return

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # Load Stage 1 results
    logger.info(f"[load] Loading Stage 1 results from {STAGE1_OUTPUT} ...")
    with open(STAGE1_OUTPUT, "rb") as f:
        stage1_data = pickle.load(f)
    logger.info(f"[load] Loaded {len(stage1_data)} sheets.")

    # Load old answers lookup
    old_answers = load_old_answers(OLD_ANSWERED_FILE)

    # Load input Excel
    xls = pd.ExcelFile(INPUT_FILE)
    sheet_results = {}

    for sheet_name in xls.sheet_names:
        if sheet_name not in stage1_data:
            logger.warning(f"[{sheet_name}] No Stage 1 data, skipping.")
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "模型的答案" not in df.columns:
            logger.warning(f"[{sheet_name}] No '模型的答案' column, skipping.")
            continue

        questions = stage1_data[sheet_name]["questions"]
        results = stage1_data[sheet_name]["results"]
        total = len(questions)

        if total != len(df):
            logger.warning(
                f"[{sheet_name}] Row count mismatch: Stage1={total}, Excel={len(df)}"
            )

        logger.info(f"[{sheet_name}] Processing {total} rows ...")

        answers = [""] * total
        blocked_count = 0
        copied_count = 0
        blank_safe_count = 0

        for idx in range(total):
            q = questions[idx]
            result = results[idx]

            if sheet_name == "生成类":
                if is_blocked(result):
                    answers[idx] = GENERATION_REFUSAL
                    blocked_count += 1
                # Safe 生成类 rows are left blank intentionally
            elif sheet_name == "拒答":
                if is_blocked(result):
                    answers[idx] = REJECTION_REFUSAL
                    blocked_count += 1
                # Safe 拒答 rows are left blank intentionally
            elif sheet_name == "非拒答":
                if result.get("safety") == "Safe":
                    ans = old_answers.get(sheet_name, {}).get(q.strip())
                    if ans:
                        answers[idx] = ans
                        copied_count += 1
                    else:
                        blank_safe_count += 1
                # Non-safe 非拒答 rows are left blank intentionally
            else:
                # Unknown sheet: leave blank
                pass

        result_df = df.copy()
        result_df["模型的答案"] = answers
        sheet_results[sheet_name] = result_df

        if sheet_name in ("生成类", "拒答"):
            logger.info(
                f"[{sheet_name}] Filled {blocked_count} blocked rows with refusal message."
            )
        elif sheet_name == "非拒答":
            logger.info(
                f"[{sheet_name}] Copied {copied_count} old answers, "
                f"{blank_safe_count} safe-but-new questions left blank."
            )

    # Write final output
    logger.info(f"[write] Saving to {OUTPUT_FILE} ...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, result_df in sheet_results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"[write] Sheet '{sheet_name}' -> {len(result_df)} rows")

    logger.info("=" * 60)
    logger.info(f"Finished! Output saved to: {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
