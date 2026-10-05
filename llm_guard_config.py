"""LLM_Guard 全局配置读取。

后端模型配置统一放在项目根目录 config.toml（模板见 config.example.toml）。
config.toml 已 gitignore，可存放 API Key。

用法:
    from llm_guard_config import get_backend_config
    cfg = get_backend_config()
    # cfg = {"base_url": ..., "api_key": ..., "model_name": ...}
"""
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = _ROOT / "config.toml"
EXAMPLE_PATH = _ROOT / "config.example.toml"


def get_backend_config() -> dict:
    """读取 config.toml 的 [backend] 配置，缺失或未填写时直接退出。

    返回 {"base_url", "api_key", "model_name", "enable_thinking"}。
    enable_thinking 默认 False（关闭思考模式，直接回答）。
    """
    if not CONFIG_PATH.exists():
        sys.exit(
            f"[config] 未找到 {CONFIG_PATH.name}\n"
            f"[config] 请执行: cp {EXAMPLE_PATH.name} {CONFIG_PATH.name}，"
            "填入后端模型配置后再运行"
        )

    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    backend = cfg.get("backend", {})
    base_url = str(backend.get("base_url", "")).strip().rstrip("/")
    api_key = str(backend.get("api_key", "")).strip()
    model_name = str(backend.get("model_name", "")).strip()
    enable_thinking = bool(backend.get("enable_thinking", False))

    missing = [
        name
        for name, value in (("base_url", base_url), ("api_key", api_key), ("model_name", model_name))
        if not value
    ]
    if missing:
        sys.exit(
            f"[config] {CONFIG_PATH.name} 的 [backend] 缺少: {', '.join(missing)}\n"
            "[config] 请填入新的后端模型地址、API Key 和模型名"
        )

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name,
        "enable_thinking": enable_thinking,
    }
