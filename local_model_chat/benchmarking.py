from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .backends import BackendFactorySettings, CompletionResult, create_backend
from .model_cache import configure_model_cache
from .presets import PRESET_BY_ID, get_preset


DEFAULT_OUTPUT_DIR = "benchmarks/results"

DEFAULT_BENCHMARK_PRESET_IDS = [
    "qwen35-4b-mlx",
    "qwen35-9b-mlx",
    "qwen35-27b-mlx",
    "qwen35-35b-a3b-mlx",
    "qwen3-4b-mlx",
    "qwen3-14b-mlx",
    "qwen3-30b-a3b-mlx",
    "qwen3-32b-mlx",
    "qwen35-9b-llama",
    "qwen35-27b-llama",
    "qwen35-35b-a3b-llama",
    "qwen3-14b-llama",
    "qwen3-30b-a3b-llama",
    "qwen3-32b-llama",
]

EXCLUDED_MODELS = [
    "Qwen3.5-122B-A10B",
    "Qwen3.5-397B-A17B",
    "Qwen3-235B-A22B",
]

SUITES = {
    "short": {"target_prompt_tokens": 512, "max_tokens": 192},
    "long": {"target_prompt_tokens": 8192, "max_tokens": 96},
}


@dataclass(frozen=True)
class BenchmarkRun:
    preset_id: str
    family: str
    label: str
    runtime: str
    artifact: str
    suite: str
    target_prompt_tokens: int
    max_tokens: int
    status: str
    reply_preview: str | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    ttft_ms: float | None = None
    decode_tokens_per_s: float | None = None
    average_tokens_per_s: float | None = None
    prompt_tokens_per_s: float | None = None
    peak_memory_gb: float | None = None
    error: str | None = None


def run_key(run: BenchmarkRun) -> tuple[str, str]:
    return (run.preset_id, run.suite)


def _machine_summary() -> dict[str, str]:
    def run_text(command: list[str]) -> str:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return completed.stdout.strip()

    return {
        "date": date.today().isoformat(),
        "platform": platform.platform(),
        "machine": run_text(["uname", "-a"]),
        "cpu": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": run_text(["sysctl", "-n", "hw.memsize"]),
    }


def build_prompt(target_prompt_tokens: int) -> list[dict[str, str]]:
    block = (
        "Local inference on Apple Silicon works best when the prompt is deterministic, "
        "boring, and easy to score. The benchmark should measure prompt processing and "
        "decode speed rather than extra reasoning behavior, so the task below is plain. "
        "The assistant should read background notes and then answer with short numbered "
        "lines containing concrete facts."
    )
    repeats = max(8, target_prompt_tokens // 64)
    background = "\n".join(f"{index + 1}. {block}" for index in range(repeats))
    task = (
        "Read the background notes and respond with exactly twelve numbered lines. "
        "Each line should contain one short factual observation. Return only the list.\n\n"
        f"Background notes:\n{background}"
    )
    return [{"role": "user", "content": task}]


def run_single_benchmark(
    preset_id: str,
    suite_name: str,
    settings: BackendFactorySettings,
) -> BenchmarkRun:
    preset = get_preset(preset_id)
    suite = SUITES[suite_name]
    backend = create_backend(preset, settings)
    try:
        result: CompletionResult
        messages = build_prompt(suite["target_prompt_tokens"])
        if preset.runtime == "llama":
            result = backend.benchmark(messages, max_tokens=suite["max_tokens"])  # type: ignore[attr-defined]
        else:
            result = backend.generate(messages, max_tokens=suite["max_tokens"])
        metrics = result.metrics
        return BenchmarkRun(
            preset_id=preset.id,
            family=preset.family,
            label=preset.label,
            runtime=preset.runtime,
            artifact=preset.hf_repo if preset.hf_file is None else f"{preset.hf_repo}:{preset.hf_file}",
            suite=suite_name,
            target_prompt_tokens=suite["target_prompt_tokens"],
            max_tokens=suite["max_tokens"],
            status="ok",
            reply_preview=result.text[:120],
            prompt_tokens=metrics.prompt_tokens,
            generation_tokens=metrics.generation_tokens,
            ttft_ms=metrics.ttft_ms,
            decode_tokens_per_s=metrics.decode_tokens_per_s,
            average_tokens_per_s=metrics.average_tokens_per_s,
            prompt_tokens_per_s=metrics.prompt_tokens_per_s,
            peak_memory_gb=metrics.peak_memory_gb,
        )
    except Exception as exc:  # noqa: BLE001
        return BenchmarkRun(
            preset_id=preset.id,
            family=preset.family,
            label=preset.label,
            runtime=preset.runtime,
            artifact=preset.hf_repo if preset.hf_file is None else f"{preset.hf_repo}:{preset.hf_file}",
            suite=suite_name,
            target_prompt_tokens=suite["target_prompt_tokens"],
            max_tokens=suite["max_tokens"],
            status="error",
            error=str(exc),
        )
    finally:
        backend.close()


def render_markdown(runs: list[BenchmarkRun]) -> str:
    lines = [
        f"# Qwen local benchmark results ({date.today().isoformat()})",
        "",
        "This file is generated by `benchmark_local_models.py`.",
        "",
        "Excluded official models:",
        "",
        *[f"- `{model}`" for model in EXCLUDED_MODELS],
        "",
        "Official note:",
        "As of this benchmark date, the official local/open comparison covers Qwen 3.5 and Qwen 3. No official Qwen 3.6 local model target was found.",
        "",
    ]
    for suite_name in ("short", "long"):
        lines.extend(
            [
                f"## {suite_name.title()} suite",
                "",
                "| Preset | Runtime | Status | TTFT ms | Decode tok/s | Avg tok/s | Prompt tok | Gen tok | Notes |",
                "| ------ | ------- | ------ | ------- | ------------ | --------- | ---------- | ------- | ----- |",
            ]
        )
        for run in [item for item in runs if item.suite == suite_name]:
            notes = (run.error or run.reply_preview or "").replace("\n", " ").strip()
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{run.label}`",
                        run.runtime,
                        run.status,
                        f"{run.ttft_ms:.0f}" if run.ttft_ms is not None else "-",
                        (
                            f"{run.decode_tokens_per_s:.2f}"
                            if run.decode_tokens_per_s is not None
                            else "-"
                        ),
                        (
                            f"{run.average_tokens_per_s:.2f}"
                            if run.average_tokens_per_s is not None
                            else "-"
                        ),
                        str(run.prompt_tokens) if run.prompt_tokens is not None else "-",
                        str(run.generation_tokens) if run.generation_tokens is not None else "-",
                        notes.replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, stamp: str, runs: list[BenchmarkRun]) -> tuple[Path, Path]:
    json_path = output_dir / f"{stamp}-qwen-local-benchmarks.json"
    md_path = output_dir / f"{stamp}-qwen-local-benchmarks.md"
    payload = {
        "meta": {
            **_machine_summary(),
            "excluded_models": EXCLUDED_MODELS,
            "note": (
                "As of this benchmark date, the official local/open comparison covers "
                "Qwen 3.5 and Qwen 3 because no official Qwen 3.6 local model target was found."
            ),
        },
        "runs": [asdict(run) for run in runs],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(runs), encoding="utf-8")
    return json_path, md_path


def load_existing_runs(output_dir: Path, stamp: str) -> list[BenchmarkRun]:
    json_path = output_dir / f"{stamp}-qwen-local-benchmarks.json"
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return [BenchmarkRun(**item) for item in payload.get("runs", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen presets.")
    parser.add_argument(
        "--preset",
        action="append",
        dest="preset_ids",
        help="Specific preset id to benchmark. Repeat for multiple values.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITES),
        dest="suites",
        help="Specific suite to run. Repeat for multiple values.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llama-url", default=None)
    parser.add_argument("--no-auto-start-llama", action="store_true")
    parser.add_argument("--llama-port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Cache root for downloaded model artifacts. Defaults to a sibling `models/` directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_preset_ids = args.preset_ids or DEFAULT_BENCHMARK_PRESET_IDS
    selected_suites = args.suites or ["short", "long"]

    for preset_id in selected_preset_ids:
        if preset_id not in PRESET_BY_ID:
            raise SystemExit(f"Unknown preset `{preset_id}`")

    model_cache_dir = configure_model_cache(args.model_cache_dir)
    settings = BackendFactorySettings(
        llama_url=args.llama_url,
        auto_start_llama=not args.no_auto_start_llama,
        ctx_size=args.ctx_size,
        llama_port=args.llama_port,
        model_cache_dir=str(model_cache_dir),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    runs = load_existing_runs(output_dir, stamp)
    run_index = {run_key(run): run for run in runs}
    json_path: Path | None = None
    md_path: Path | None = None
    for preset_id in selected_preset_ids:
        for suite_name in selected_suites:
            print(f"[benchmark] {preset_id} / {suite_name}", flush=True)
            run = run_single_benchmark(preset_id, suite_name, settings)
            run_index[run_key(run)] = run
            runs = sorted(run_index.values(), key=lambda item: (item.family, item.label, item.runtime, item.suite))
            json_path, md_path = write_outputs(output_dir, stamp, runs)

    if json_path is None or md_path is None:
        json_path, md_path = write_outputs(output_dir, stamp, runs)

    print(f"[benchmark] model cache: {model_cache_dir}", flush=True)
    print(f"[benchmark] wrote {json_path}", flush=True)
    print(f"[benchmark] wrote {md_path}", flush=True)
    return 0
