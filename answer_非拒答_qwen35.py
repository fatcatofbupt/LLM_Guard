#!/usr/bin/env python3
"""
Answer all questions in the '非拒答' sheet of 附件5 天津测试题_合并去重.xlsx
using the qwen3.5-122b-a10b model via OpenAI-compatible API.

SEQUENTIAL mode: one request at a time, no concurrency.
This ensures the model is not overloaded while production jobs are running.

Supports checkpoint/resume. Saves results to a new Excel file.

Usage:
    conda run -n guard_train python3 answer_非拒答_qwen35.py
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_FILE = "data/附件5 天津测试题_合并去重.xlsx"
OUTPUT_FILE = "data/附件5 天津测试题_合并去重_已回答.xlsx"
SHEET_NAME = "非拒答"

API_BASE = "http://172.31.0.13:33890/v1"
API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
MODEL_NAME = "qwen3.5-122b-a10b"

MAX_TOKENS = 2048
REQUEST_TIMEOUT = 120          # seconds per API call
MAX_RETRIES = 3
RETRY_DELAY = 5                # seconds between retries

CHECKPOINT_FILE = ".answer_非拒答_checkpoint.json"
CHECKPOINT_INTERVAL = 50       # save progress every N rows

# ------------------------------------------------------------------
# Simple logger
# ------------------------------------------------------------------
def log(level: str, msg: str):
    timestamp = time.strftime("%H:%M:%S")
    color = {"INFO": "32", "WARN": "33", "ERROR": "31"}.get(level, "0")
    print(f"\033[{color}m{timestamp} | {level: <6} | {msg}\033[0m")


# ------------------------------------------------------------------
# API caller (stdlib only)
# ------------------------------------------------------------------
def call_api(question: str) -> str:
    """Call the remote model API and return the assistant's content."""
    url = f"{API_BASE}/chat/completions"
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, method="POST", data=payload)
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Connection", "close")

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if not choices:
                return "[ERROR: empty choices]"
            content = choices[0].get("message", {}).get("content")
            if content is None:
                reasoning = choices[0].get("message", {}).get("reasoning")
                if reasoning:
                    return reasoning
                return "[ERROR: content is null]"
            return content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            last_error = f"HTTP {e.code}: {body}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return f"[ERROR: {last_error}]"


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------
def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(answers: list, last_index: int):
    data = {
        "answers": answers,
        "last_index": last_index,
        "total": len(answers),
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    log("INFO", "=" * 60)
    log("INFO", "Answer 非拒答 sheet via qwen3.5-122b-a10b API")
    log("INFO", f"Input:       {INPUT_FILE}")
    log("INFO", f"Output:      {OUTPUT_FILE}")
    log("INFO", f"API:         {API_BASE}")
    log("INFO", "Mode:        SEQUENTIAL (1 request at a time)")
    log("INFO", "=" * 60)

    # 1. Read Excel
    if not os.path.exists(INPUT_FILE):
        log("ERROR", f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    log("INFO", f"Reading sheet '{SHEET_NAME}' from {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    total = len(df)
    log("INFO", f"Total rows: {total}")

    if "题目" not in df.columns:
        log("ERROR", "Column '题目' not found in the sheet.")
        sys.exit(1)

    # Ensure output column exists
    if "模型的答案" not in df.columns:
        df["模型的答案"] = ""

    # 2. Load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        log("INFO", f"Checkpoint found: last_index={checkpoint.get('last_index', -1)}")
        answers = checkpoint.get("answers", [""] * total)
        while len(answers) < total:
            answers.append("")
    else:
        answers = [""] * total

    start_idx = checkpoint.get("last_index", -1) + 1
    if start_idx >= total:
        log("INFO", "All rows already processed according to checkpoint.")
    else:
        log("INFO", f"Resuming from row {start_idx} ...")

    # 3. Process SEQUENTIALLY — one request at a time
    try:
        for idx in range(start_idx, total):
            question = str(df.at[idx, "题目"]).strip()
            if not question:
                answers[idx] = ""
                continue

            log("INFO", f"[{idx + 1}/{total}] Q: {question[:60]}...")
            t0 = time.time()
            answer = call_api(question)
            elapsed = time.time() - t0
            answers[idx] = answer

            log("INFO", f"[{idx + 1}/{total}] Done in {elapsed:.1f}s, answer length: {len(answer)} chars")

            # Save checkpoint periodically
            if (idx + 1) % CHECKPOINT_INTERVAL == 0 or idx == total - 1:
                save_checkpoint(answers, idx)
                log("INFO", f"Checkpoint saved at row {idx + 1}")

    except KeyboardInterrupt:
        log("WARN", "Interrupted by user. Saving checkpoint ...")
        last_done = start_idx - 1
        for i in range(start_idx, total):
            if answers[i] != "" or str(df.at[i, "题目"]).strip() == "":
                last_done = i
            else:
                break
        save_checkpoint(answers, last_done)
        log("INFO", f"Checkpoint saved at row {last_done + 1}. Resume by running the script again.")
        sys.exit(0)

    # 4. Write output Excel
    log("INFO", f"Writing results to {OUTPUT_FILE} ...")
    df["模型的答案"] = answers

    with pd.ExcelFile(INPUT_FILE) as xls:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            for sheet in xls.sheet_names:
                if sheet == SHEET_NAME:
                    df.to_excel(writer, sheet_name=sheet, index=False)
                else:
                    other_df = pd.read_excel(xls, sheet_name=sheet)
                    other_df.to_excel(writer, sheet_name=sheet, index=False)
                log("INFO", f"  Sheet '{sheet}' written.")

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    log("INFO", "=" * 60)
    log("INFO", f"Finished! Output saved to: {OUTPUT_FILE}")
    log("INFO", "=" * 60)


if __name__ == "__main__":
    main()
