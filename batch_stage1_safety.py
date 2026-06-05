#!/usr/bin/env python3
"""
Stage 1: Safety inference only.

Loads the fine-tuned Qwen3Guard model and runs ALL questions from
all_questions.xlsx through it. Saves intermediate results to disk.

Does NOT call the backend API.

Usage:
    conda run -n py311 python3 batch_stage1_safety.py
"""

import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SAFETY_MODEL_PATH = "./finetune_qwen3guard/output/lora_v5_1/merged_model"
DEVICE = "cuda:0"

INPUT_FILE = os.environ.get("INPUT_FILE", "all_questions.xlsx")
STAGE1_OUTPUT = os.environ.get("STAGE1_OUTPUT", "data/.batch_stage1_results.pkl")

SAFETY_BATCH_SIZE = 32
MAX_NEW_TOKENS = 128

# Regex for parsing guard output
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
# Model loading & inference
# ------------------------------------------------------------------
def load_safety_model():
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


def run_safety_on_questions(questions: list[str], tok, model) -> list[dict]:
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

    safe_count = sum(1 for r in results if r["safety"] == "Safe")
    unsafe_count = sum(1 for r in results if r["safety"] in ("Unsafe", "Controversial"))
    unknown_count = total - safe_count - unsafe_count
    logger.info(f"[safety] Summary: Safe={safe_count}, Unsafe/Controversial={unsafe_count}, Unknown={unknown_count}")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Stage 1: Safety Inference")
    logger.info(f"Model:  {SAFETY_MODEL_PATH}")
    logger.info(f"Input:  {INPUT_FILE}")
    logger.info(f"Output: {STAGE1_OUTPUT}")
    logger.info("=" * 60)

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    tok, model = load_safety_model()

    xls = pd.ExcelFile(INPUT_FILE)
    logger.info(f"Found sheets: {xls.sheet_names}")

    all_results = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if "题目" not in df.columns:
            logger.warning(f"[{sheet_name}] No '题目' column, skipping.")
            continue

        questions = [str(row.get("题目", "")).strip() for _, row in df.iterrows()]
        results = run_safety_on_questions(questions, tok, model)

        all_results[sheet_name] = {
            "questions": questions,
            "results": results,
        }

        safe = sum(1 for r in results if r["safety"] == "Safe")
        unsafe = sum(1 for r in results if r["safety"] in ("Unsafe", "Controversial"))
        logger.info(f"[{sheet_name}] Done — Safe={safe}, Unsafe/Controversial={unsafe}")

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()
    logger.info("[cleanup] Safety model unloaded from GPU.")

    # Save intermediate results
    with open(STAGE1_OUTPUT, "wb") as f:
        pickle.dump(all_results, f)
    logger.info(f"[save] Stage 1 results saved to: {STAGE1_OUTPUT}")

    # Summary
    logger.info("=" * 60)
    total_safe = 0
    total_unsafe = 0
    for sheet_name, data in all_results.items():
        s = sum(1 for r in data["results"] if r["safety"] == "Safe")
        u = sum(1 for r in data["results"] if r["safety"] in ("Unsafe", "Controversial"))
        total_safe += s
        total_unsafe += u
        logger.info(f"  {sheet_name:10s} | Safe: {s:5d} | Unsafe: {u:5d}")
    logger.info(f"  {'TOTAL':10s} | Safe: {total_safe:5d} | Unsafe: {total_unsafe:5d}")
    logger.info("=" * 60)
    logger.info("Stage 1 complete. Run batch_stage2_backend.py after approval.")


if __name__ == "__main__":
    main()
