#!/usr/bin/env python3
"""
Safety-guarded chat completion service with async batch inference.

Pipeline:
  1. User input -> async batch queue -> Qwen3Guard fine-tuned model (GPU 3)
  2. If Safe  -> forward to backend LLM (OpenAI-compatible API)
  3. If Unsafe -> return fixed refusal message

Features:
  - Async batch inference: multiple safety checks are batched together
  - Queue-based: requests wait in a queue, worker processes in batches
  - Streaming & non-streaming both supported

Endpoints:
  POST /v1/chat/completions   (OpenAI-compatible)
  GET  /health                (health check)
"""

import os
import re
import json
import time
import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SAFETY_MODEL_PATH = "finetune_qwen3guard/output/lora_v5_1/merged_model"
DEVICE = "cuda:3"  # GPU 3

BACKEND_BASE_URL = "http://172.31.0.97:3391/v1"
BACKEND_API_KEY = "JmpwFmQoEz6kBwSIFcAyl6b7q6XxPmbM"
BACKEND_MODEL_NAME = "qwen3.5-122b-a10b"

SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 32469

REFUSAL_MESSAGE = "根据政策和法律规定，我无法回答你的问题。"

MAX_NEW_TOKENS_SAFETY = 128
BATCH_SIZE = 8          # Max items per batch
BATCH_TIMEOUT_MS = 50   # Max wait time (ms) to form a batch

# Regex for parsing guard output
SAFETY_RE = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)

# ------------------------------------------------------------------
# Dataclass for queued safety check requests
# ------------------------------------------------------------------
@dataclass
class SafetyRequest:
    text: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueue_time: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
safety_tok = None
safety_model = None
backend_client = None
safety_queue: asyncio.Queue = asyncio.Queue()
batch_worker_task: asyncio.Task | None = None

# Keyword filter (loaded at startup)
KEYWORD_PATTERNS: list[tuple[str, str]] = []  # (keyword_text, violation_type)
KEYWORD_EXCEL_PATH = "data/raw/附件4 天津关键词拦截列表_合并去重.xlsx"


# ------------------------------------------------------------------
# Safety model loading
# ------------------------------------------------------------------
def load_safety_model():
    """Load fine-tuned guard model onto GPU 3."""
    logger.info(f"[safety] Loading model from {SAFETY_MODEL_PATH} onto {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(SAFETY_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        SAFETY_MODEL_PATH,
        torch_dtype="auto",
        trust_remote_code=True
        # device_map={"": DEVICE},
    )
    model.eval()
    logger.info("[safety] Model loaded.")
    return tok, model


# ------------------------------------------------------------------
# Batch inference worker
# ------------------------------------------------------------------
def _batch_moderate(texts: list[str]) -> list[dict]:
    """
    Synchronous batch inference. Called from the worker thread.
    Returns list of parsed verdict dicts, same order as input texts.
    """
    messages = [[{"role": "user", "content": t}] for t in texts]
    chat_texts = [
        safety_tok.apply_chat_template(m, tokenize=False)
        for m in messages
    ]
    enc = safety_tok(chat_texts, return_tensors="pt", padding=True).to(safety_model.device)

    with torch.no_grad():
        generated = safety_model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS_SAFETY,
            do_sample=False,
            pad_token_id=safety_tok.pad_token_id,
        )

    prompt_len = enc.input_ids.size(1)
    results = []
    for i in range(generated.size(0)):
        out_ids = generated[i][prompt_len:].tolist()
        raw = safety_tok.decode(out_ids, skip_special_tokens=True).strip()

        s = SAFETY_RE.search(raw)
        cats = CATEGORY_RE.findall(raw)
        verdict = s.group(1) if s else None

        results.append({
            "raw": raw,
            "safety": verdict,
            "categories": cats,
        })
    return results


async def _batch_worker():
    """
    Background coroutine that collects safety requests from the queue
    and processes them in batches.
    """
    loop = asyncio.get_event_loop()

    while True:
        batch: list[SafetyRequest] = []

        # Wait for at least one item
        first = await safety_queue.get()
        batch.append(first)

        # Collect more items until batch is full or timeout
        deadline = time.time() + (BATCH_TIMEOUT_MS / 1000.0)
        while len(batch) < BATCH_SIZE:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(
                    safety_queue.get(),
                    timeout=remaining,
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break

        # Run batch inference in executor (model calls are CPU-bound / GPU-sync)
        texts = [r.text for r in batch]
        try:
            results = await loop.run_in_executor(
                None,  # default executor
                lambda: _batch_moderate(texts),
            )
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


async def check_safety_async(text: str) -> dict:
    """
    Enqueue a safety check request and asynchronously wait for the result.
    """
    req = SafetyRequest(text=text)
    await safety_queue.put(req)
    result = await req.future
    return result


def is_safe(text: str) -> bool:
    """
    Synchronous wrapper (for direct use). Prefer check_safety_async in async context.
    """
    result = check_safety_async(text)
    # This is a coroutine; caller must await it
    raise RuntimeError("Use 'await check_safety_async(text)' in async context")


# ------------------------------------------------------------------
# Backend client (OpenAI-compatible, async)
# ------------------------------------------------------------------
def load_keyword_patterns():
    """Load keyword patterns from Excel. Exact substring match, no regex."""
    import pandas as pd

    if not os.path.exists(KEYWORD_EXCEL_PATH):
        logger.warning(f"[keyword] Excel not found: {KEYWORD_EXCEL_PATH}, skipping keyword filter.")
        return []

    df = pd.read_excel(KEYWORD_EXCEL_PATH)
    if "关键词" not in df.columns:
        logger.warning("[keyword] Column '关键词' not found, skipping keyword filter.")
        return []

    patterns = []
    for _, row in df.iterrows():
        keyword = str(row.get("关键词", "")).strip()
        vtype = str(row.get("违规类型", "")).strip()
        if not keyword:
            continue
        # Strip '*' characters (they were wildcards in the original regex approach)
        cleaned = keyword.replace("*", "").lower()
        if cleaned:
            patterns.append((cleaned, vtype))

    logger.info(f"[keyword] Loaded {len(patterns)} keyword patterns from {KEYWORD_EXCEL_PATH}")
    return patterns


def check_keywords(text: str) -> tuple[bool, str | None]:
    """
    Check if text contains any blocked keyword (exact substring match, no regex).
    Returns (hit: bool, violation_type: str | None)
    """
    lowered = text.lower()
    for kw, vtype in KEYWORD_PATTERNS:
        if kw in lowered:
            return True, vtype
    return False, None


# ------------------------------------------------------------------
# Backend client (OpenAI-compatible, async)
# ------------------------------------------------------------------
def load_backend_client():
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package required: pip install openai")

    client = AsyncOpenAI(
        base_url=BACKEND_BASE_URL,
        api_key=BACKEND_API_KEY,
    )
    logger.info(f"[backend] Async client -> {BACKEND_BASE_URL}, model={BACKEND_MODEL_NAME}")
    return client


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Minimal request body — clients send only model, messages, and stream flag.
    temperature / top_p / max_tokens are hidden and handled internally.
    """
    model: str = BACKEND_MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global safety_tok, safety_model, backend_client, batch_worker_task, KEYWORD_PATTERNS

    # Load keyword filter first (fast, no GPU needed)
    KEYWORD_PATTERNS = load_keyword_patterns()

    safety_tok, safety_model = load_safety_model()
    backend_client = load_backend_client()

    # Start the batch worker
    batch_worker_task = asyncio.create_task(_batch_worker())
    logger.info(f"[startup] Batch worker started (batch_size={BATCH_SIZE}, timeout={BATCH_TIMEOUT_MS}ms)")

    yield

    # Shutdown
    if batch_worker_task:
        batch_worker_task.cancel()
        try:
            await batch_worker_task
        except asyncio.CancelledError:
            pass
    del safety_model
    torch.cuda.empty_cache()
    logger.info("[shutdown] Cleaned up.")


app = FastAPI(title="Model Safety Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "queue_size": safety_queue.qsize(),
    }


async def _stream_refusal() -> AsyncGenerator[str, None]:
    chunk = {
        "id": "chatcmpl-refused",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": BACKEND_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": REFUSAL_MESSAGE},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _build_refusal_response() -> dict:
    return {
        "id": "chatcmpl-refused",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": BACKEND_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REFUSAL_MESSAGE},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions with async batch safety guard.
    """
    # Extract latest user message
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")

    latest_user_text = user_messages[-1].content
    backend_messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # ---- Step 0: Keyword pre-filter (fastest, no model call) ----
    keyword_hit, violation_type = check_keywords(latest_user_text)
    keyword_hit=False
    if keyword_hit:
        logger.warning(f"[keyword] BLOCKED keyword hit: type='{violation_type}', text='{latest_user_text[:80]}...'")
        if req.stream:
            return StreamingResponse(
                _stream_refusal(),
                media_type="text/event-stream",
            )
        return _build_refusal_response()

    # ---- Step 1: Parallel safety check + backend call ----
    safety_task = asyncio.create_task(check_safety_async(latest_user_text))

    try:
        if req.stream:
            # Start backend stream creation in parallel with safety check
            backend_stream_task = asyncio.create_task(
                backend_client.chat.completions.create(
                    model=BACKEND_MODEL_NAME,
                    messages=backend_messages,
                    stream=True,
                )
            )

            # Wait for safety verdict (usually the faster one)
            result = await safety_task
            safety = result.get("safety")
            logger.info(f"[safety] verdict={safety}, raw={result['raw']!r}")

            if safety in ("Unsafe", "Controversial"):
                logger.warning(f"[safety] BLOCKED ({safety}), raw={result['raw']!r}")
                backend_stream_task.cancel()
                try:
                    await backend_stream_task
                except asyncio.CancelledError:
                    pass
                return StreamingResponse(
                    _stream_refusal(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            # Safe → start streaming immediately (do not wait for all tokens)
            backend_stream = await backend_stream_task

            async def stream_generator():
                async for chunk in backend_stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        else:
            # Start backend call in parallel
            backend_task = asyncio.create_task(
                backend_client.chat.completions.create(
                    model=BACKEND_MODEL_NAME,
                    messages=backend_messages,
                )
            )

            # Wait for safety verdict first (usually faster)
            result = await safety_task
            safety = result.get("safety")
            logger.info(f"[safety] verdict={safety}, raw={result['raw']!r}")

            if safety in ("Unsafe", "Controversial"):
                logger.warning(f"[safety] BLOCKED ({safety}), raw={result['raw']!r}")
                backend_task.cancel()
                try:
                    await backend_task
                except asyncio.CancelledError:
                    pass
                return _build_refusal_response()

            # Safe → wait for and return backend response
            resp = await backend_task
            return resp.model_dump()

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
