#!/usr/bin/env python3
"""
Fill the remaining empty cells in the 0721 answered file:

- 生成类 / 拒答 rows that are still empty (Safe verdicts) -> call backend API.
- 非拒答 rows that are still empty (Unsafe/Controversial verdicts) -> fill with
  the 生成类 refusal phrase.

Usage:
    conda run -n py311 python3 pipeline/fill_remaining_0721.py
"""

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE", "questions/0721-附件5_测试题_已回答.xlsx"
)

BACKEND_BASE_URL = "http://172.31.0.97:3391/v1"
BACKEND_API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
BACKEND_MODEL_NAME = "qwen3.5-122b-a10b"

NON_REJECTED_REFUSAL = "对不起，我还没有学会回答这个问题。如果你有其他问题，我非常乐意为你提供帮助"

BACKEND_CONCURRENCY = 16
BACKEND_TIMEOUT = 90
BACKEND_CONNECT_TIMEOUT = 10
BACKEND_READ_TIMEOUT = 60

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


# ------------------------------------------------------------------
# Backend API
# ------------------------------------------------------------------
async def call_backend(client: httpx.AsyncClient, question: str) -> str:
    payload = {
        "model": BACKEND_MODEL_NAME,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {BACKEND_API_KEY}"}
    resp = await client.post(
        f"{BACKEND_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


async def run_backend_for_rows(
    client: httpx.AsyncClient,
    questions: list[str],
) -> list[str]:
    total = len(questions)
    logger.info(f"[backend] Calling backend for {total} questions (concurrency={BACKEND_CONCURRENCY}) ...")

    semaphore = asyncio.Semaphore(BACKEND_CONCURRENCY)
    answers = [""] * total
    completed = 0

    async def process_one(idx: int, question: str):
        nonlocal completed
        async with semaphore:
            try:
                answers[idx] = await call_backend(client, question)
            except Exception as e:
                err_str = str(e) or type(e).__name__
                logger.warning(f"[backend] Index {idx} failed: {err_str}")
                answers[idx] = f"[ERROR: {err_str}]"
            completed += 1
            if completed % max(1, total // 5) == 0 or completed == total:
                logger.info(f"[backend] Progress: {completed}/{total}")

    tasks = [process_one(i, q) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks, return_exceptions=True)
    return answers


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Fill remaining empty rows in 0721 answered file")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("=" * 60)

    if not os.path.exists(OUTPUT_FILE):
        logger.error(f"Output file not found: {OUTPUT_FILE}")
        return

    xls = pd.ExcelFile(OUTPUT_FILE)
    sheet_results = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "模型的答案" not in df.columns or "题目" not in df.columns:
            logger.warning(f"[{sheet_name}] Missing required columns, skipping.")
            continue

        empty_mask = df["模型的答案"].isna() | (df["模型的答案"] == "")
        empty_indices = df[empty_mask].index.tolist()
        total_empty = len(empty_indices)

        if total_empty == 0:
            logger.info(f"[{sheet_name}] No empty rows.")
            sheet_results[sheet_name] = df
            continue

        logger.info(f"[{sheet_name}] Found {total_empty} empty rows.")

        if sheet_name in ("生成类", "拒答"):
            questions = df.loc[empty_indices, "题目"].astype(str).tolist()

            timeout = httpx.Timeout(
                BACKEND_TIMEOUT,
                connect=BACKEND_CONNECT_TIMEOUT,
                read=BACKEND_READ_TIMEOUT,
            )
            limits = httpx.Limits(
                max_connections=BACKEND_CONCURRENCY + 4,
                max_keepalive_connections=BACKEND_CONCURRENCY,
            )

            async def run_backend():
                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    return await run_backend_for_rows(client, questions)

            answers = asyncio.run(run_backend())
            for idx, ans in zip(empty_indices, answers):
                df.at[idx, "模型的答案"] = ans
            logger.info(f"[{sheet_name}] Filled {len(answers)} rows from backend.")

        elif sheet_name == "非拒答":
            for idx in empty_indices:
                df.at[idx, "模型的答案"] = NON_REJECTED_REFUSAL
            logger.info(
                f"[{sheet_name}] Filled {total_empty} rows with refusal phrase."
            )

        sheet_results[sheet_name] = df

    # Write back to the same file
    logger.info(f"[write] Saving to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, result_df in sheet_results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"[write] Sheet '{sheet_name}' -> {len(result_df)} rows")

    logger.info("=" * 60)
    logger.info("Finished! Output saved.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
