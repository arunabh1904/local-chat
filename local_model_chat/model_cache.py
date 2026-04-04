from __future__ import annotations

import os
from pathlib import Path


MODEL_CACHE_ENV_VAR = "LOCAL_CHAT_MODEL_CACHE_DIR"


def default_model_cache_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "models"


def resolve_model_cache_dir(model_cache_dir: str | None = None) -> Path:
    raw_value = model_cache_dir or os.environ.get(MODEL_CACHE_ENV_VAR)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return default_model_cache_dir().resolve()


def configure_model_cache(model_cache_dir: str | None = None) -> Path:
    cache_root = resolve_model_cache_dir(model_cache_dir)
    hf_home = cache_root / "huggingface"
    hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"

    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    os.environ[MODEL_CACHE_ENV_VAR] = str(cache_root)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    return cache_root
