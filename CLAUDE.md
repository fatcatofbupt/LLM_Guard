# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A small research workspace for **LLM safety moderation**. It runs the **Qwen3Guard-Gen-0.6B** model locally to classify prompts and responses, against a Chinese-language safety evaluation corpus stored in `docs/`.

There is no build system, no test framework, and no package manifest — everything runs as ad-hoc Python scripts.

## Run

```bash
conda run -n py311 python3 scripts/demos/hello_qwen3guard.py
```

This is the main demo entry point. It loads the model from the **local path** `models/Qwen/Qwen3Guard-Gen-0.6B` (not from Hugging Face Hub — the weights are vendored under `models/`, ~1.5 GB safetensors). Device selection is automatic: `cuda` → `mps` → `cpu`.

Requires `transformers>=4.51.0` and `torch`. Use `bash scripts/install_deps.sh` or `pip install -r requirements.txt`.

## Architecture

### `scripts/demos/hello_qwen3guard.py`

Two-function pipeline:

- `moderate(messages, max_new_tokens=128)` — applies the chat template, generates, decodes only the newly generated tokens. The **same function** handles both prompt moderation (one `user` message) and response moderation (a `user` + `assistant` pair). The model's output format differs accordingly.
- `parse(content)` — regex-extracts three fields from the generated text:
  - `safety`: `Safe | Unsafe | Controversial`
  - `categories`: list drawn from a fixed enum of 9 categories + `Jailbreak` (input only) + `None`
  - `refusal`: `Yes | No` (only present for response moderation)

The category enum and severity scheme are defined by the model and documented in `models/Qwen/Qwen3Guard-Gen-0.6B/README.md` — refer to it before changing the regex in `parse()`.

### `docs/` — evaluation corpus

The source data is two Excel files in Chinese. They've been converted to JSON with **English field names** (see `aaa-附件5：评估测试题.json` and `aaa-附件5：评估测试题_已回答_OtherModel.json`):

- Top-level dict keyed by 7 test categories: `appendix_a_content_evaluation_manual_sampling`, `appendix_a_content_evaluation_keyword_sampling`, `intellectual_property_trade_secrets_test`, `discriminatory_content_test`, `accuracy`, `reliability`, `should_refuse`.
- Each value is a list of records with fields: `question`, `primary_category`, `secondary_category`, and optionally `answer` (only in the `_已回答_OtherModel` variant, which carries answers from a different model for comparison).
- ~20,910 records total per file. Chinese values are preserved as-is; only keys were translated.

When regenerating these JSONs from xlsx, keep the field-name mapping consistent with what already exists.

### `models/Qwen/Qwen3Guard-Gen-0.6B/`

Local copy of the model. The model is generative — safety classification is framed as instruction-following, so the output is plain text that must be regex-parsed. Do not assume a classifier head.
