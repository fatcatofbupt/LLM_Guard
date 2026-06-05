#!/usr/bin/env python3
"""
Two-stage batch processing of all_questions.xlsx (standalone script).

Stage 1: Load Qwen3Guard safety model locally, run all questions through it.
         Unsafe/Controversial → refusal message (skip Stage 2).
         Safe → proceed to Stage 2.

Stage 2: Call backend qwen3.5-122b-a10b API for safe questions only.

Output: all_questiions_finished.xlsx (same structure as input).

Usage:
    conda run -n py311 python3 batch_process_two_stage.py
"""

import asyncio
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SAFETY_MODEL_PATH = "./finetune_qwen3guard/output/lora_v3/merged_model"
DEVICE = "cuda:0"  # Safety model GPU

BACKEND_BASE_URL = "http://172.31.0.97:3391/v1"
BACKEND_API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
BACKEND_MODEL_NAME = "qwen3.5-122b-a10b"

INPUT_FILE = os.environ.get("INPUT_FILE", "all_questions.xlsx")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "all_questiions_finished.xlsx")
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", ".batch_checkpoint.pkl")

REFUSAL_MESSAGE = "根据政策和法律规定，我无法回答你的问题。"

SAFETY_BATCH_SIZE = 32       # Questions per safety inference batch
MAX_NEW_TOKENS = 128

BACKEND_CONCURRENCY = 16     # Max parallel backend calls
BACKEND_TIMEOUT = 90         # Seconds per backend request (total)
BACKEND_CONNECT_TIMEOUT = 10
BACKEND_READ_TIMEOUT = 60

# Regex for parsing guard output (same as safety_service.py)
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)

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
# Checkpoint helpers
# ------------------------------------------------------------------
def save_checkpoint(checkpoint: dict, path: str = CHECKPOINT_FILE):
    """Save progress to disk after each sheet."""
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    logger.info(f"[checkpoint] Saved to {path}")


def load_checkpoint(path: str = CHECKPOINT_FILE) -> dict | None:
    """Load checkpoint if exists."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            cp = pickle.load(f)
        logger.info(f"[checkpoint] Loaded from {path} — sheets done: {list(cp.keys())}")
        return cp
    return None


def clear_checkpoint(path: str = CHECKPOINT_FILE):
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"[checkpoint] Cleared {path}")


# ------------------------------------------------------------------
# Stage 1: Safety Model
# ------------------------------------------------------------------
def load_safety_model():
    """Load fine-tuned guard model."""
    logger.info(f"[safety] Loading model from {SAFETY_MODEL_PATH} onto {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(SAFETY_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        SAFETY_MODEL_PATH,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.eval()
    logger.info("[safety] Model loaded.")
    return tok, model


def batch_moderate(texts: list[str], tok, model) -> list[dict]:
    """
    Synchronous batch inference.
    Returns list of parsed verdict dicts, same order as input texts.
    """
    messages = [[{"role": "user", "content": t}] for t in texts]
    chat_texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    enc = tok(chat_texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    prompt_len = enc.input_ids.size(1)
    results = []
    for i in range(generated.size(0)):
        out_ids = generated[i][prompt_len:].tolist()
        raw = tok.decode(out_ids, skip_special_tokens=True).strip()

        s = SAFETY_RE.search(raw)
        cats = CATEGORY_RE.findall(raw)
        verdict = s.group(1) if s else None

        results.append({
            "raw": raw,
            "safety": verdict,
            "categories": cats,
        })
    return results


# ------------------------------------------------------------------
# Stage 2: Backend API
# ------------------------------------------------------------------
async def call_backend(client: httpx.AsyncClient, question: str) -> str:
    """Call backend qwen3.5-122b-a10b model. Returns assistant content."""
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


# ------------------------------------------------------------------
# Sheet Processing
# ------------------------------------------------------------------
def run_safety_on_questions(questions: list[str], tok, model) -> list[dict]:
    """
    Run safety model on all questions in batches.
    Returns list of result dicts (same order as questions).
    """
    total = len(questions)
    results = []
    logger.info(f"[safety] Running inference on {total} questions (batch_size={SAFETY_BATCH_SIZE}) ...")

    for i in range(0, total, SAFETY_BATCH_SIZE):
        batch = questions[i:i + SAFETY_BATCH_SIZE]
        batch_results = batch_moderate(batch, tok, model)
        results.extend(batch_results)

        done = min(i + SAFETY_BATCH_SIZE, total)
        if done % max(1, total // 10) < SAFETY_BATCH_SIZE or done == total:
            logger.info(f"[safety] Progress: {done}/{total} ({done / total * 100:.1f}%)")

    # Log summary
    safe_count = sum(1 for r in results if r["safety"] == "Safe")
    unsafe_count = sum(1 for r in results if r["safety"] in ("Unsafe", "Controversial"))
    unknown_count = total - safe_count - unsafe_count
    logger.info(f"[safety] Summary: Safe={safe_count}, Unsafe/Controversial={unsafe_count}, Unknown={unknown_count}")

    return results


async def run_backend_on_safe(
    client: httpx.AsyncClient,
    questions: list[str],
    indices: list[int],
) -> dict[int, str]:
    """
    Call backend API for safe questions concurrently.
    Returns dict mapping original index -> answer.
    """
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


def process_sheet(df: pd.DataFrame, sheet_name: str, tok, model) -> pd.DataFrame:
    """
    Process one sheet with two-stage logic.
    Returns new DataFrame with updated '模型的答案' column.
    """
    if "题目" not in df.columns:
        logger.warning(f"[{sheet_name}] '题目' column not found, skipping.")
        return df
    if "模型的答案" not in df.columns:
        logger.warning(f"[{sheet_name}] '模型的答案' column not found, skipping.")
        return df

    total = len(df)
    logger.info(f"[{sheet_name}] Processing {total} rows ...")

    # Extract questions
    questions = []
    for idx, row in df.iterrows():
        q = str(row.get("题目", "")).strip()
        questions.append(q)

    # Stage 1: Safety inference
    safety_results = run_safety_on_questions(questions, tok, model)

    # Split into safe and unsafe
    safe_indices = []
    safe_questions = []
    unsafe_indices = []

    for idx, result in enumerate(safety_results):
        if result["safety"] == "Safe":
            safe_indices.append(idx)
            safe_questions.append(questions[idx])
        else:
            unsafe_indices.append(idx)

    # Prepare answers array
    answers = [""] * total

    # Fill unsafe answers directly
    for idx in unsafe_indices:
        answers[idx] = REFUSAL_MESSAGE
    logger.info(f"[{sheet_name}] Filled {len(unsafe_indices)} unsafe answers with refusal message.")

    # Stage 2: Backend calls for safe questions
    if safe_questions:
        # Build timeout with explicit connect/read limits to prevent deadlocks
        timeout = httpx.Timeout(
            BACKEND_TIMEOUT,
            connect=BACKEND_CONNECT_TIMEOUT,
            read=BACKEND_READ_TIMEOUT,
        )
        # Limit connection pool to prevent resource exhaustion
        limits = httpx.Limits(
            max_connections=BACKEND_CONCURRENCY + 4,
            max_keepalive_connections=BACKEND_CONCURRENCY,
        )

        async def run_backend():
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                return await run_backend_on_safe(client, safe_questions, safe_indices)

        safe_answers = asyncio.run(run_backend())
        for idx, ans in safe_answers.items():
            answers[idx] = ans
        logger.info(f"[{sheet_name}] Filled {len(safe_answers)} safe answers from backend.")

    # Build result DataFrame
    result_df = df.copy()
    result_df["模型的答案"] = answers
    return result_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Two-Stage Batch Processing")
    logger.info(f"Input:  {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # Check input file
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {}

    # Load safety model (Stage 1)
    tok, model = load_safety_model()

    # Read Excel
    xls = pd.ExcelFile(INPUT_FILE)
    logger.info(f"Found sheets: {xls.sheet_names}")

    # Process each sheet
    sheet_results = {}
    for sheet_name in xls.sheet_names:
        if sheet_name in checkpoint:
            logger.info(f"[{sheet_name}] Skipping — already in checkpoint.")
            sheet_results[sheet_name] = checkpoint[sheet_name]
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)
        result_df = process_sheet(df, sheet_name, tok, model)
        sheet_results[sheet_name] = result_df

        # Save checkpoint after each sheet
        checkpoint[sheet_name] = result_df
        save_checkpoint(checkpoint)

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()
    logger.info("[cleanup] Safety model unloaded from GPU.")

    # Write output
    logger.info(f"[write] Saving to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name, result_df in sheet_results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"[write] Sheet '{sheet_name}' -> {len(result_df)} rows")

    # Clear checkpoint on success
    clear_checkpoint()

    logger.info("=" * 60)
    logger.info(f"Finished! Output saved to: {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
