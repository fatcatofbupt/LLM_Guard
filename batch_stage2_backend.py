#!/usr/bin/env python3
"""
Stage 2: Backend API calls only.

Reads Stage 1 safety results and the original Excel file.
For Safe questions: calls backend qwen3.5-122b-a10b API.
For Unsafe/Controversial: fills refusal message directly.

Produces the final all_questiions_finished.xlsx.

Usage (only after Stage 1 completes):
    conda run -n py311 python3 batch_stage2_backend.py
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
STAGE1_OUTPUT = os.environ.get("STAGE1_OUTPUT", "data/.batch_stage1_results.pkl")
INPUT_FILE = os.environ.get("INPUT_FILE", "all_questions.xlsx")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "all_questiions_finished.xlsx")

BACKEND_BASE_URL = "http://172.31.0.97:3391/v1"
BACKEND_API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
BACKEND_MODEL_NAME = "qwen3.5-122b-a10b"

REFUSAL_MESSAGE = "根据政策和法律规定，我无法回答你的问题。"

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


async def run_backend_on_safe(
    client: httpx.AsyncClient,
    questions: list[str],
    indices: list[int],
) -> dict[int, str]:
    total = len(questions)
    logger.info(f"[backend] Calling backend for {total} safe questions (concurrency={BACKEND_CONCURRENCY}) ...")

    semaphore = asyncio.Semaphore(BACKEND_CONCURRENCY)
    answers = {}
    completed = 0

    async def process_one(idx: int, question: str):
        nonlocal completed
        async with semaphore:
            try:
                content = await call_backend(client, question)
                answers[idx] = content
            except Exception as e:
                err_str = str(e) or type(e).__name__
                logger.warning(f"[backend] Index {idx} failed: {err_str}")
                answers[idx] = f"[ERROR: {err_str}]"
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                logger.info(f"[backend] Progress: {completed}/{total} ({completed / total * 100:.1f}%)")

    tasks = [process_one(idx, q) for idx, q in zip(indices, questions)]
    await asyncio.gather(*tasks, return_exceptions=True)
    return answers


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Stage 2: Backend API Calls")
    logger.info(f"Stage1 data: {STAGE1_OUTPUT}")
    logger.info(f"Input:       {INPUT_FILE}")
    logger.info(f"Output:      {OUTPUT_FILE}")
    logger.info("=" * 60)

    import pickle

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

    # Load original Excel
    xls = pd.ExcelFile(INPUT_FILE)

    # Build timeout + connection limits
    timeout = httpx.Timeout(
        BACKEND_TIMEOUT,
        connect=BACKEND_CONNECT_TIMEOUT,
        read=BACKEND_READ_TIMEOUT,
    )
    limits = httpx.Limits(
        max_connections=BACKEND_CONCURRENCY + 4,
        max_keepalive_connections=BACKEND_CONCURRENCY,
    )

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

        logger.info(f"[{sheet_name}] Processing {total} rows ...")

        # Split safe / unsafe
        safe_indices = []
        safe_questions = []
        unsafe_indices = []

        for idx, result in enumerate(results):
            if result["safety"] == "Safe":
                safe_indices.append(idx)
                safe_questions.append(questions[idx])
            else:
                unsafe_indices.append(idx)

        answers = [""] * total

        # Fill unsafe answers
        for idx in unsafe_indices:
            answers[idx] = REFUSAL_MESSAGE
        logger.info(f"[{sheet_name}] Filled {len(unsafe_indices)} unsafe answers with refusal message.")

        # Stage 2: Backend calls for safe questions
        if safe_questions:
            async def run_backend():
                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    return await run_backend_on_safe(client, safe_questions, safe_indices)

            safe_answers = asyncio.run(run_backend())
            for idx, ans in safe_answers.items():
                answers[idx] = ans
            logger.info(f"[{sheet_name}] Filled {len(safe_answers)} safe answers from backend.")
        else:
            logger.info(f"[{sheet_name}] No safe questions, skipping backend.")

        result_df = df.copy()
        result_df["模型的答案"] = answers
        sheet_results[sheet_name] = result_df

    # Write final output
    logger.info(f"[write] Saving to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, result_df in sheet_results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"[write] Sheet '{sheet_name}' -> {len(result_df)} rows")

    logger.info("=" * 60)
    logger.info(f"Finished! Output saved to: {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
