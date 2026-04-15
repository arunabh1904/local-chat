from __future__ import annotations

import base64
import binascii
import gc
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter

from .presets import DEFAULT_VISION_PRESET_ID, Preset, SYSTEM_PROMPT, get_preset


REQUEST_TIMEOUT = 60 * 60
LLAMA_INTERNAL_HOST = "127.0.0.1"
LLAMA_INTERNAL_PORT = 18080
SUMMARY_MAX_TOKENS = 256
SUMMARY_SYSTEM_PROMPT = (
    "You compress a conversation so another local model can continue it. "
    "Preserve user goals, constraints, important facts, unresolved questions, "
    "and partial work. Keep it concise and factual."
)

# Xet-backed downloads were unreliable for some official Qwen MLX repos on this setup.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass(frozen=True)
class BackendFactorySettings:
    system_prompt: str = SYSTEM_PROMPT
    llama_url: str | None = None
    auto_start_llama: bool = True
    ctx_size: int = 16384
    llama_port: int = LLAMA_INTERNAL_PORT
    model_cache_dir: str | None = None
    vision_preset_id: str | None = DEFAULT_VISION_PRESET_ID


@dataclass(frozen=True)
class CompletionMetrics:
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    decode_tokens_per_s: float | None = None
    average_tokens_per_s: float | None = None
    prompt_tokens_per_s: float | None = None
    peak_memory_gb: float | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    metrics: CompletionMetrics


@dataclass(frozen=True)
class ConversationImage:
    name: str
    media_type: str
    data_url: str


def log(message: str) -> None:
    print(f"[local-chat] {message}", flush=True)


def make_session() -> requests.Session:
    session = requests.Session()
    session.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
    return session


def wait_for_http(url: str, attempts: int = 180, delay_s: float = 1.0) -> None:
    session = make_session()
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = session.get(url, timeout=5)
            if response.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(delay_s)
    raise RuntimeError(f"Timed out waiting for {url}") from last_error


def cleanup_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        proc.wait(timeout=20)


def _serialize_messages(
    messages: list[dict[str, str]],
    conversation_image: ConversationImage | None = None,
) -> str:
    if not messages:
        transcript = "(no recent messages)"
    else:
        transcript = "\n\n".join(
        f"{item['role'].upper()}:\n{item['content'].strip()}" for item in messages
    )
    if conversation_image is None:
        return transcript
    return (
        f"[Attached image: {conversation_image.name or 'unnamed image'}]\n\n"
        f"{transcript}"
    )


def _messages_with_optional_image(
    messages: list[dict[str, str]],
    conversation_image: ConversationImage | None,
) -> list[dict[str, Any]]:
    rendered_messages: list[dict[str, Any]] = [dict(item) for item in messages]
    if conversation_image is None:
        return rendered_messages

    for index in range(len(rendered_messages) - 1, -1, -1):
        item = rendered_messages[index]
        if item["role"] != "user":
            continue
        item["content"] = [
            {"type": "image"},
            {"type": "text", "text": item["content"]},
        ]
        return rendered_messages

    rendered_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the attached image briefly."},
            ],
        }
    )
    return rendered_messages


def build_chat_messages(
    system_prompt: str,
    conversation_summary: str,
    messages: list[dict[str, str]],
    conversation_image: ConversationImage | None = None,
) -> list[dict[str, Any]]:
    system_parts = [system_prompt.strip()]
    if conversation_summary.strip():
        system_parts.append(
            "Carry forward this prior conversation summary when responding:\n"
            f"{conversation_summary.strip()}"
        )
    if conversation_image is not None:
        system_parts.append(
            "The current turn includes one attached image. Use it alongside the text."
        )
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        *_messages_with_optional_image(messages, conversation_image),
    ]


def build_summary_messages(
    conversation_summary: str,
    messages: list[dict[str, str]],
    conversation_image: ConversationImage | None = None,
) -> list[dict[str, Any]]:
    prior_summary = conversation_summary.strip() or "(none)"
    transcript = _serialize_messages(messages, conversation_image)
    prompt = (
        "Write a concise carry-forward summary for a model switch.\n\n"
        f"Existing summary:\n{prior_summary}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        "Return a short paragraph. Mention the user's goal, constraints, "
        "important factual context, and the most recent unfinished thread."
    )
    user_message: dict[str, Any]
    if conversation_image is None:
        user_message = {"role": "user", "content": prompt}
    else:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        user_message,
    ]


def _image_suffix(conversation_image: ConversationImage) -> str:
    media_type = conversation_image.media_type.lower()
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    raise ValueError(f"Unsupported image media type `{conversation_image.media_type}`")


def _decode_image_bytes(conversation_image: ConversationImage) -> bytes:
    prefix = f"data:{conversation_image.media_type};base64,"
    if not conversation_image.data_url.startswith(prefix):
        raise ValueError("Image payload must be a base64 data URL")
    payload = conversation_image.data_url.removeprefix(prefix)
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Image payload is not valid base64") from exc


@contextmanager
def materialize_conversation_image(
    conversation_image: ConversationImage | None,
) -> Any:
    if conversation_image is None:
        yield None
        return

    image_bytes = _decode_image_bytes(conversation_image)
    suffix = _image_suffix(conversation_image)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="local-chat-image-",
            suffix=suffix,
            delete=False,
        ) as handle:
            handle.write(image_bytes)
            temp_path = handle.name
        yield temp_path
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _chat_template_target(processor_or_tokenizer: Any) -> Any:
    if hasattr(processor_or_tokenizer, "apply_chat_template"):
        return processor_or_tokenizer
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    raise RuntimeError("This model does not expose a chat template")


def _render_prompt(
    processor_or_tokenizer: Any,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any],
) -> str:
    target = _chat_template_target(processor_or_tokenizer)
    return target.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


class Backend:
    runtime_name: str
    display_model: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        conversation_image: ConversationImage | None = None,
    ) -> CompletionResult:
        raise NotImplementedError

    def close(self) -> None:
        return


class MLXLMBackend(Backend):
    runtime_name = "MLX"

    def __init__(self, preset: Preset):
        try:
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "MLX text models require `mlx-lm`. Run `./setup.sh` first."
            ) from exc

        log(f"Loading MLX text model {preset.hf_repo}")
        self.model, self.tokenizer = load(preset.hf_repo)
        self.stream_generate = stream_generate
        self.sampler = make_sampler(temp=0.0, top_k=1, top_p=1.0)
        self.display_model = preset.label
        self.chat_template_kwargs = preset.chat_template_kwargs
        self.lock = threading.Lock()

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        conversation_image: ConversationImage | None = None,
    ) -> CompletionResult:
        if conversation_image is not None:
            raise RuntimeError(
                f"{self.display_model} is text-only. Switch to a Qwen 3.5 preset to use image chat."
            )
        rendered_prompt = _render_prompt(
            self.tokenizer,
            messages,
            self.chat_template_kwargs,
        )
        pieces: list[str] = []
        first_token_at: float | None = None
        last_chunk: Any | None = None
        started = time.perf_counter()
        with self.lock:
            for chunk in self.stream_generate(
                self.model,
                self.tokenizer,
                rendered_prompt,
                max_tokens=max_tokens,
                sampler=self.sampler,
            ):
                last_chunk = chunk
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(chunk.text)
        finished = time.perf_counter()
        text = "".join(pieces).strip()
        generation_tokens = getattr(last_chunk, "generation_tokens", None)
        if generation_tokens is None:
            generation_tokens = len(self.tokenizer.encode(text))
        metrics = CompletionMetrics(
            ttft_ms=((first_token_at - started) * 1000) if first_token_at else None,
            prompt_tokens=getattr(last_chunk, "prompt_tokens", None),
            generation_tokens=generation_tokens,
            decode_tokens_per_s=getattr(last_chunk, "generation_tps", None),
            average_tokens_per_s=(
                generation_tokens / max(finished - started, 1e-9)
                if generation_tokens is not None
                else None
            ),
            prompt_tokens_per_s=getattr(last_chunk, "prompt_tps", None),
            peak_memory_gb=getattr(last_chunk, "peak_memory", None),
        )
        return CompletionResult(text=text, metrics=metrics)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()


class MLXVLMBackend(Backend):
    runtime_name = "MLX"

    def __init__(self, preset: Preset):
        try:
            from mlx_vlm import load
            from mlx_vlm.generate import stream_generate
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "MLX VLM models require `mlx-vlm`. Run `./setup.sh` first."
            ) from exc

        log(f"Loading MLX VLM model {preset.hf_repo}")
        self.model, self.processor = load(preset.hf_repo)
        self.stream_generate = stream_generate
        self.display_model = preset.label
        self.chat_template_kwargs = preset.chat_template_kwargs
        self.lock = threading.Lock()

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        conversation_image: ConversationImage | None = None,
    ) -> CompletionResult:
        rendered_prompt = _render_prompt(
            self.processor,
            messages,
            self.chat_template_kwargs,
        )
        pieces: list[str] = []
        first_token_at: float | None = None
        last_chunk: Any | None = None
        started = time.perf_counter()
        with materialize_conversation_image(conversation_image) as image_path:
            with self.lock:
                for chunk in self.stream_generate(
                    self.model,
                    self.processor,
                    prompt=rendered_prompt,
                    image=image_path,
                    audio=None,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                ):
                    last_chunk = chunk
                    if chunk.text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(chunk.text)
        finished = time.perf_counter()
        text = "".join(pieces).strip()
        generation_tokens = getattr(last_chunk, "generation_tokens", None)
        metrics = CompletionMetrics(
            ttft_ms=((first_token_at - started) * 1000) if first_token_at else None,
            prompt_tokens=getattr(last_chunk, "prompt_tokens", None),
            generation_tokens=generation_tokens,
            decode_tokens_per_s=getattr(last_chunk, "generation_tps", None),
            average_tokens_per_s=(
                generation_tokens / max(finished - started, 1e-9)
                if generation_tokens is not None
                else None
            ),
            prompt_tokens_per_s=getattr(last_chunk, "prompt_tps", None),
            peak_memory_gb=getattr(last_chunk, "peak_memory", None),
        )
        return CompletionResult(text=text, metrics=metrics)

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()


class LlamaCppBackend(Backend):
    runtime_name = "llama.cpp"

    def __init__(self, preset: Preset, settings: BackendFactorySettings):
        self.display_model = preset.label
        self.preset = preset
        self.llama_url = settings.llama_url or (
            f"http://{LLAMA_INTERNAL_HOST}:{settings.llama_port}"
        )
        self.session = make_session()
        self.proc: subprocess.Popen[Any] | None = None

        if settings.auto_start_llama:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "Auto-starting llama.cpp requires `huggingface_hub`. Run `./setup.sh` first."
                ) from exc

            if not preset.hf_file:
                raise RuntimeError(f"Preset `{preset.id}` does not define a GGUF file")
            log(f"Downloading GGUF for {preset.label} from {preset.hf_repo}")
            model_path = hf_hub_download(repo_id=preset.hf_repo, filename=preset.hf_file)
            log_path = Path("/private/tmp") / f"local-model-chat-{preset.id}.log"
            log_file = log_path.open("w")
            command = [
                "llama-server",
                "--model",
                model_path,
                "--no-mmproj",
                "--host",
                LLAMA_INTERNAL_HOST,
                "--port",
                str(settings.llama_port),
                "--ctx-size",
                str(settings.ctx_size),
                "--flash-attn",
                "on",
            ]
            if preset.chat_template_kwargs:
                command.extend(
                    [
                        "--chat-template-kwargs",
                        json.dumps(preset.chat_template_kwargs),
                    ]
                )
            log(f"Starting llama-server on {self.llama_url}")
            self.proc = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_for_http(f"{self.llama_url}/health")
            except Exception:
                cleanup_process(self.proc)
                raise
        else:
            log(f"Using existing llama.cpp server at {self.llama_url}")
            wait_for_http(f"{self.llama_url}/health")

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.preset.id,
            "messages": messages,
            "stream": stream,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 123,
            "max_tokens": max_tokens,
        }
        if self.preset.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.preset.chat_template_kwargs
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _extract_text_fragment(self, delta: Any) -> str:
        if isinstance(delta, str):
            return delta
        if isinstance(delta, list):
            pieces: list[str] = []
            for item in delta:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            return "".join(pieces)
        return ""

    def _nonstream_request(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> tuple[str, dict[str, Any], float]:
        started = time.perf_counter()
        response = self.session.post(
            f"{self.llama_url}/v1/chat/completions",
            json=self._build_payload(messages, max_tokens, stream=False),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        finished = time.perf_counter()
        data = response.json()
        text = self._extract_text_fragment(data["choices"][0]["message"]["content"]).strip()
        return text, data.get("usage", {}), finished - started

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        conversation_image: ConversationImage | None = None,
    ) -> CompletionResult:
        if conversation_image is not None:
            raise RuntimeError(
                f"{self.display_model} is text-only. Switch to a Qwen 3.5 preset to use image chat."
            )
        text, usage, total_time = self._nonstream_request(messages, max_tokens)
        generation_tokens = usage.get("completion_tokens")
        metrics = CompletionMetrics(
            generation_tokens=generation_tokens,
            prompt_tokens=usage.get("prompt_tokens"),
            average_tokens_per_s=(
                generation_tokens / max(total_time, 1e-9)
                if generation_tokens is not None
                else None
            ),
        )
        return CompletionResult(text=text, metrics=metrics)

    def benchmark(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> CompletionResult:
        pieces: list[str] = []
        usage: dict[str, Any] | None = None
        first_token_at: float | None = None
        started = time.perf_counter()
        with self.session.post(
            f"{self.llama_url}/v1/chat/completions",
            json=self._build_payload(messages, max_tokens, stream=True),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                payload = raw_line.removeprefix("data: ").strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                usage = event.get("usage") or usage
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                text = self._extract_text_fragment(content)
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(text)
        finished = time.perf_counter()
        text = "".join(pieces).strip()
        if usage is None:
            _, usage, _ = self._nonstream_request(messages, max_tokens)
        generation_tokens = usage.get("completion_tokens")
        decode_time = (finished - first_token_at) if first_token_at is not None else None
        metrics = CompletionMetrics(
            ttft_ms=((first_token_at - started) * 1000) if first_token_at else None,
            prompt_tokens=usage.get("prompt_tokens"),
            generation_tokens=generation_tokens,
            decode_tokens_per_s=(
                generation_tokens / max(decode_time, 1e-9)
                if generation_tokens is not None and decode_time
                else None
            ),
            average_tokens_per_s=(
                generation_tokens / max(finished - started, 1e-9)
                if generation_tokens is not None
                else None
            ),
        )
        return CompletionResult(text=text, metrics=metrics)

    def close(self) -> None:
        cleanup_process(self.proc)


def create_backend(preset: Preset, settings: BackendFactorySettings) -> Backend:
    if preset.loader_kind == "mlx_lm":
        return MLXLMBackend(preset)
    if preset.loader_kind == "mlx_vlm":
        return MLXVLMBackend(preset)
    if preset.loader_kind == "llama_cpp":
        return LlamaCppBackend(preset, settings)
    raise RuntimeError(f"Unsupported loader kind `{preset.loader_kind}`")


class BackendManager:
    def __init__(
        self,
        initial_preset: Preset,
        settings: BackendFactorySettings,
        backend_factory: Callable[[Preset, BackendFactorySettings], Backend] = create_backend,
        autoload: bool = True,
    ):
        self.settings = settings
        self.backend_factory = backend_factory
        self.lock = threading.RLock()
        self.state = "loading"
        self.error: str | None = None
        self.current_preset = initial_preset
        self.backend: Backend | None = None
        self.cache_notice = ""
        self.loader_thread: threading.Thread | None = None
        if autoload:
            self._load_locked(initial_preset)
        else:
            self._begin_load_locked(initial_preset)

    def _hub_cache_dir(self) -> Path | None:
        if not self.settings.model_cache_dir:
            return None
        return Path(self.settings.model_cache_dir) / "huggingface" / "hub"

    def _repo_cache_dir(self, preset: Preset) -> Path | None:
        hub_cache_dir = self._hub_cache_dir()
        if hub_cache_dir is None:
            return None
        return hub_cache_dir / f"models--{preset.hf_repo.replace('/', '--')}"

    def _has_repo_cache(self, preset: Preset) -> bool:
        repo_cache_dir = self._repo_cache_dir(preset)
        return bool(repo_cache_dir and repo_cache_dir.exists() and any(repo_cache_dir.iterdir()))

    def _cache_notice_for_preset(self, preset: Preset) -> str:
        hub_cache_dir = self._hub_cache_dir()
        if hub_cache_dir is None:
            return f"Loading {preset.label}. Model artifacts will use the configured Hugging Face cache."
        if self._has_repo_cache(preset):
            return f"Loading {preset.label} from local cache at {hub_cache_dir}."
        return (
            f"Downloading {preset.label} from Hugging Face into {hub_cache_dir}, "
            "then loading it. The first run can take a while."
        )

    def _begin_load_locked(self, preset: Preset) -> None:
        self.state = "loading"
        self.error = None
        self.current_preset = preset
        self.backend = None
        self.cache_notice = self._cache_notice_for_preset(preset)

    def _finish_load_locked(self, preset: Preset, backend: Backend) -> None:
        self.current_preset = preset
        self.backend = backend
        self.state = "ready"
        self.error = None
        self.cache_notice = ""

    def _fail_load_locked(self, preset: Preset, exc: Exception) -> None:
        self.current_preset = preset
        self.backend = None
        self.state = "error"
        self.error = str(exc)
        self.cache_notice = f"Failed to load {preset.label}: {exc}"

    def _load_locked(self, preset: Preset) -> None:
        self._begin_load_locked(preset)
        try:
            backend = self.backend_factory(preset, self.settings)
        except Exception as exc:
            self._fail_load_locked(preset, exc)
            raise
        self._finish_load_locked(preset, backend)

    def _load_in_background(self, preset: Preset) -> None:
        try:
            backend = self.backend_factory(preset, self.settings)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._fail_load_locked(preset, exc)
            return

        with self.lock:
            if self.current_preset.id != preset.id or self.state != "loading":
                backend.close()
                return
            self._finish_load_locked(preset, backend)

    def start_loading(self) -> None:
        with self.lock:
            if self.backend is not None:
                return
            if self.loader_thread is not None and self.loader_thread.is_alive():
                return
            preset = self.current_preset
            self._begin_load_locked(preset)
            self.loader_thread = threading.Thread(
                target=self._load_in_background,
                args=(preset,),
                daemon=True,
            )
            self.loader_thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            vision_preset = (
                get_preset(self.settings.vision_preset_id)
                if self.settings.vision_preset_id
                else None
            )
            return {
                "state": self.state,
                "error": self.error,
                "current_preset": self.current_preset.to_public_dict(),
                "benchmark_url": self.current_preset.benchmark_url,
                "model_cache_dir": self.settings.model_cache_dir,
                "cache_notice": self.cache_notice,
                "image_chat_available": bool(vision_preset),
                "vision_preset": vision_preset.to_public_dict() if vision_preset else None,
            }

    def _require_backend(self) -> Backend:
        if self.state != "ready" or self.backend is None:
            raise RuntimeError("The backend is not ready yet")
        return self.backend

    def chat(
        self,
        messages: list[dict[str, str]],
        conversation_summary: str,
        max_tokens: int,
        conversation_image: ConversationImage | None = None,
    ) -> CompletionResult:
        with self.lock:
            if conversation_image is not None and not self.current_preset.supports_images:
                if not self.settings.vision_preset_id:
                    raise RuntimeError(
                        f"{self.current_preset.label} is text-only and no vision preset is configured."
                    )
                vision_preset = get_preset(self.settings.vision_preset_id)
                if not vision_preset.supports_images:
                    raise RuntimeError(
                        f"Configured vision preset `{vision_preset.id}` does not support images."
                    )
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
                    gc.collect()
                log(
                    f"Image attached while {self.current_preset.label} is active; "
                    f"switching to vision preset {vision_preset.label}"
                )
                self._load_locked(vision_preset)
            backend = self._require_backend()
            full_messages = build_chat_messages(
                self.settings.system_prompt,
                conversation_summary,
                messages,
                conversation_image,
            )
            return backend.generate(
                full_messages,
                max_tokens=max_tokens,
                conversation_image=conversation_image,
            )

    def benchmark(
        self,
        preset_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> CompletionResult:
        preset = get_preset(preset_id)
        backend = self.backend_factory(preset, self.settings)
        try:
            if isinstance(backend, LlamaCppBackend):
                return backend.benchmark(messages, max_tokens=max_tokens)
            return backend.generate(messages, max_tokens=max_tokens)
        finally:
            backend.close()

    def switch_preset(
        self,
        preset_id: str,
        messages: list[dict[str, str]],
        conversation_summary: str,
        conversation_image: ConversationImage | None = None,
    ) -> str:
        target_preset = get_preset(preset_id)
        with self.lock:
            backend = self._require_backend()
            if target_preset.id == self.current_preset.id:
                return conversation_summary.strip()

            summary_image = conversation_image if self.current_preset.supports_images else None
            if messages or conversation_summary.strip():
                summary_result = backend.generate(
                    build_summary_messages(
                        conversation_summary,
                        messages,
                        summary_image,
                    ),
                    max_tokens=SUMMARY_MAX_TOKENS,
                    conversation_image=summary_image,
                )
                carried_summary = summary_result.text.strip()
            elif summary_image is not None:
                summary_result = backend.generate(
                    build_summary_messages("", [], summary_image),
                    max_tokens=SUMMARY_MAX_TOKENS,
                    conversation_image=summary_image,
                )
                carried_summary = summary_result.text.strip()
            elif conversation_image is not None:
                carried_summary = (
                    f"The user attached image `{conversation_image.name}`, but it was cleared "
                    "while switching away from a text-only preset before the model analyzed it."
                )
            else:
                carried_summary = ""

            backend.close()
            self.backend = None
            gc.collect()
            try:
                self._load_locked(target_preset)
            except Exception as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            return carried_summary

    def close(self) -> None:
        with self.lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
