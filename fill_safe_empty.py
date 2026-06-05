#!/usr/bin/env python3
"""
Answer the 20 Safe-but-empty questions directly via the backend LLM API.
Update the final merged Excel in-place.

Usage:
    conda run -n guard_train python3 fill_safe_empty.py
"""

import json
import os
import time
import urllib.request

import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FINAL_FILE = "data/附件5_天津测试题_最终合并结果.xlsx"

API_BASE = "http://172.31.0.13:33890/v1"
API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
MODEL_NAME = "qwen3.5-122b-a10b"

MAX_TOKENS = 2048
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 5


# ------------------------------------------------------------------
# API caller (same as stage2 script)
# ------------------------------------------------------------------
def call_api(question: str) -> str:
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
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Fill Safe-but-empty rows via backend LLM")
    print(f"File: {FINAL_FILE}")
    print("=" * 60)

    # Read all sheets
    xls = pd.ExcelFile(FINAL_FILE)
    sheet_results = {}
    filled_count = 0

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "safety_label" not in df.columns or "模型的答案" not in df.columns:
            sheet_results[sheet_name] = df
            continue

        # Find Safe rows with empty answers
        safe_mask = df["safety_label"].astype(str).str.strip() == "Safe"
        empty_mask = df["模型的答案"].isna() | (df["模型的答案"].astype(str).str.strip() == "")
        target_mask = safe_mask & empty_mask
        target_indices = df[target_mask].index.tolist()

        print(f"\n[{sheet_name}] Found {len(target_indices)} Safe-but-empty rows")

        for idx in target_indices:
            question = str(df.at[idx, "题目"]).strip()
            if not question:
                continue

            print(f"  [{idx}] Q: {question[:60]}...")
            t0 = time.time()
            answer = call_api(question)
            elapsed = time.time() - t0
            print(f"  [{idx}] Done in {elapsed:.1f}s, len={len(answer)} chars")

            df.at[idx, "模型的答案"] = answer
            filled_count += 1

        sheet_results[sheet_name] = df

    # Write back
    print(f"\n[write] Saving {filled_count} new answers to {FINAL_FILE} ...")
    with pd.ExcelWriter(FINAL_FILE, engine="openpyxl") as writer:
        for sheet_name, df in sheet_results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet '{sheet_name}' -> {len(df)} rows")

    print(f"\nFinished! Filled {filled_count} answers.")
    print(f"Output: {FINAL_FILE}")


if __name__ == "__main__":
    main()
