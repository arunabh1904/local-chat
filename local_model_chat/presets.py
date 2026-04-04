from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer directly and clearly."
DEFAULT_PRESET_ID = "gemma4-26b-a4b-mlx"
GEMMA_BENCHMARK_URL = (
    "https://arunabh1904.github.io/blog/2026/04/04/"
    "running-gemma-4-locally-on-a-64-gb-macbook-pro.html"
)
QWEN_BENCHMARK_URL = (
    "https://arunabh1904.github.io/blog/2026/04/04/"
    "running-qwen-3-5-and-qwen-3-locally-on-a-64-gb-macbook-pro.html"
)


@dataclass(frozen=True)
class Preset:
    id: str
    family: str
    label: str
    runtime: str
    loader_kind: str
    hf_repo: str
    benchmark_url: str
    hf_file: str | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_label(self) -> str:
        return "MLX" if self.runtime == "mlx" else "llama.cpp"

    @property
    def supports_images(self) -> bool:
        return self.loader_kind == "mlx_vlm"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "label": self.label,
            "runtime": self.runtime,
            "runtime_label": self.runtime_label,
            "benchmark_url": self.benchmark_url,
            "supports_images": self.supports_images,
        }


def _preset(
    *,
    preset_id: str,
    family: str,
    label: str,
    runtime: str,
    loader_kind: str,
    hf_repo: str,
    benchmark_url: str,
    hf_file: str | None = None,
    disable_thinking: bool = True,
) -> Preset:
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    return Preset(
        id=preset_id,
        family=family,
        label=label,
        runtime=runtime,
        loader_kind=loader_kind,
        hf_repo=hf_repo,
        hf_file=hf_file,
        benchmark_url=benchmark_url,
        chat_template_kwargs=kwargs,
    )


PRESETS: tuple[Preset, ...] = (
    _preset(
        preset_id="gemma4-e2b-mlx",
        family="Gemma 4",
        label="Gemma 4 E2B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/gemma-4-e2b-it-4bit",
        benchmark_url=GEMMA_BENCHMARK_URL,
    ),
    _preset(
        preset_id="gemma4-e4b-mlx",
        family="Gemma 4",
        label="Gemma 4 E4B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/gemma-4-e4b-it-4bit",
        benchmark_url=GEMMA_BENCHMARK_URL,
    ),
    _preset(
        preset_id="gemma4-26b-a4b-mlx",
        family="Gemma 4",
        label="Gemma 4 26B A4B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/gemma-4-26b-a4b-it-4bit",
        benchmark_url=GEMMA_BENCHMARK_URL,
    ),
    _preset(
        preset_id="gemma4-31b-mlx",
        family="Gemma 4",
        label="Gemma 4 31B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/gemma-4-31b-it-4bit",
        benchmark_url=GEMMA_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen35-4b-mlx",
        family="Qwen 3.5",
        label="Qwen 3.5 4B",
        runtime="mlx",
        loader_kind="mlx_vlm",
        hf_repo="mlx-community/Qwen3.5-4B-MLX-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen35-9b-mlx",
        family="Qwen 3.5",
        label="Qwen 3.5 9B",
        runtime="mlx",
        loader_kind="mlx_vlm",
        hf_repo="mlx-community/Qwen3.5-9B-MLX-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen35-27b-mlx",
        family="Qwen 3.5",
        label="Qwen 3.5 27B",
        runtime="mlx",
        loader_kind="mlx_vlm",
        hf_repo="mlx-community/Qwen3.5-27B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen35-35b-a3b-mlx",
        family="Qwen 3.5",
        label="Qwen 3.5 35B A3B",
        runtime="mlx",
        loader_kind="mlx_vlm",
        hf_repo="mlx-community/Qwen3.5-35B-A3B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen3-4b-mlx",
        family="Qwen 3",
        label="Qwen 3 4B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/Qwen3-4B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen3-14b-mlx",
        family="Qwen 3",
        label="Qwen 3 14B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/Qwen3-14B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen3-30b-a3b-mlx",
        family="Qwen 3",
        label="Qwen 3 30B A3B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/Qwen3-30B-A3B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
    _preset(
        preset_id="qwen3-32b-mlx",
        family="Qwen 3",
        label="Qwen 3 32B",
        runtime="mlx",
        loader_kind="mlx_lm",
        hf_repo="mlx-community/Qwen3-32B-4bit",
        benchmark_url=QWEN_BENCHMARK_URL,
    ),
)

PRESET_BY_ID = {preset.id: preset for preset in PRESETS}

LEGACY_MODEL_MAP = {
    ("mlx", "e2b"): "gemma4-e2b-mlx",
    ("mlx", "e4b"): "gemma4-e4b-mlx",
    ("mlx", "26b"): "gemma4-26b-a4b-mlx",
    ("mlx", "31b"): "gemma4-31b-mlx",
    ("llama", "e2b"): "gemma4-e2b-mlx",
    ("llama", "e4b"): "gemma4-e4b-mlx",
    ("llama", "26b"): "gemma4-26b-a4b-mlx",
    ("llama", "31b"): "gemma4-31b-mlx",
}


def all_presets() -> list[Preset]:
    return list(PRESETS)


def get_preset(preset_id: str) -> Preset:
    try:
        return PRESET_BY_ID[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown preset `{preset_id}`") from exc


def resolve_initial_preset_id(
    preset_id: str | None,
    runtime: str | None,
    legacy_model: str | None,
) -> str:
    if preset_id:
        if preset_id in PRESET_BY_ID:
            return preset_id
        raise ValueError(f"Unknown preset `{preset_id}`")

    if runtime and legacy_model:
        mapped = LEGACY_MODEL_MAP.get((runtime, legacy_model))
        if mapped:
            return mapped
        raise ValueError(
            f"Legacy combination `--runtime {runtime} --model {legacy_model}` is unsupported"
        )

    return DEFAULT_PRESET_ID


def runtime_model_help() -> str:
    return ", ".join(sorted({model for _, model in LEGACY_MODEL_MAP}))
