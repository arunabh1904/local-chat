#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import signal
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter


SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer directly and clearly."
DEFAULT_MAX_TOKENS = 512
REQUEST_TIMEOUT = 60 * 60
LLAMA_INTERNAL_HOST = "127.0.0.1"
LLAMA_INTERNAL_PORT = 18080
BENCHMARK_URL = "https://arunabh1904.github.io/blog/2026/04/04/running-gemma-4-locally-on-a-64-gb-macbook-pro.html"


HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#f4f8ff" />
    <title>Gemma Local Chat</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500&family=Orbitron:wght@500;700;800&family=Rajdhani:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap"
    />
    <style>
      :root {
        color-scheme: light;
        --font-body: "Rajdhani", sans-serif;
        --font-heading: "Orbitron", sans-serif;
        --font-reading: "IBM Plex Sans", sans-serif;
        --font-mono: "Roboto Mono", monospace;
        --bg: #f4f8ff;
        --bg-secondary: #e5eefc;
        --text: #10213f;
        --muted: #4d628f;
        --accent: #356dff;
        --accent-soft: rgba(53, 109, 255, 0.1);
        --accent-strong: rgba(53, 109, 255, 0.18);
        --panel: rgba(245, 249, 255, 0.84);
        --panel-strong: rgba(248, 251, 255, 0.96);
        --panel-elevated: linear-gradient(145deg, rgba(248, 251, 255, 0.94), rgba(228, 238, 255, 0.9));
        --line: rgba(53, 109, 255, 0.16);
        --line-strong: rgba(53, 109, 255, 0.32);
        --shadow: 0 0 0 1px rgba(53, 109, 255, 0.05), 0 18px 44px rgba(53, 85, 150, 0.12);
        --assistant: linear-gradient(145deg, rgba(255, 255, 255, 0.98), rgba(242, 247, 255, 0.92));
        --user: linear-gradient(145deg, rgba(53, 109, 255, 0.98), rgba(87, 135, 255, 0.94));
        --system: rgba(247, 250, 255, 0.76);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        min-height: 100vh;
        background:
          radial-gradient(circle at top, rgba(77, 132, 255, 0.1), transparent 34%),
          linear-gradient(180deg, rgba(231, 240, 255, 0.56), transparent 22%),
          linear-gradient(145deg, var(--bg), var(--bg-secondary));
        color: var(--text);
        font-family: var(--font-body);
      }

      .shell {
        width: min(980px, calc(100vw - 32px));
        margin: 28px auto;
        background: var(--panel-elevated);
        border: 1px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        overflow: hidden;
        position: relative;
        z-index: 1;
      }

      .hero {
        display: grid;
        gap: 14px;
        padding: 26px 24px 18px;
        border-bottom: 1px solid var(--line);
        background:
          linear-gradient(145deg, rgba(248, 251, 255, 0.96), rgba(235, 242, 255, 0.92)),
          var(--panel);
      }

      .eyebrow {
        margin: 0;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 12px;
        font-weight: 700;
      }

      .hero-row {
        display: grid;
        gap: 12px;
      }

      .hero h1 {
        margin: 0;
        font-size: clamp(30px, 4vw, 40px);
        letter-spacing: -0.01em;
        line-height: 1;
        font-family: var(--font-heading);
      }

      .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.6;
        font-family: var(--font-reading);
        max-width: 62ch;
      }

      .hero-link,
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(248, 251, 255, 0.94);
        border: 1px solid var(--line);
        border-radius: 999px;
        font-size: 13px;
        color: var(--muted);
      }

      .hero-link {
        text-decoration: none;
        transition:
          border-color 0.16s ease,
          color 0.16s ease,
          transform 0.16s ease,
          background-color 0.16s ease;
      }

      .hero-link:hover {
        color: var(--text);
        border-color: var(--line-strong);
        background: rgba(239, 245, 255, 0.98);
        transform: translateY(-1px);
      }

      .hero-link strong {
        color: var(--accent);
        font-weight: 700;
      }

      .pill strong {
        color: var(--accent);
        font-weight: 600;
      }

      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .chat {
        display: grid;
        grid-template-rows: minmax(360px, 1fr) auto;
        min-height: 72vh;
      }

      .messages {
        padding: 22px;
        display: grid;
        gap: 14px;
        align-content: start;
        overflow-y: auto;
        background: linear-gradient(180deg, rgba(248, 251, 255, 0.46), rgba(245, 249, 255, 0.2));
      }

      .message {
        max-width: min(82ch, 90%);
        padding: 14px 16px;
        border-radius: 20px;
        border: 1px solid var(--line);
        white-space: pre-wrap;
        line-height: 1.55;
        font-size: 15px;
        box-shadow: 0 10px 28px rgba(53, 85, 150, 0.08);
        font-family: var(--font-reading);
      }

      .message.user {
        margin-left: auto;
        background: var(--user);
        border-color: rgba(53, 109, 255, 0.34);
        color: #fdfefe;
        border-bottom-right-radius: 6px;
      }

      .message.assistant {
        background: var(--assistant);
        border-bottom-left-radius: 6px;
        color: var(--text);
      }

      .message.system {
        background: var(--system);
        border-style: dashed;
        color: var(--muted);
      }

      .composer {
        padding: 18px 22px 22px;
        border-top: 1px solid var(--line);
        background: var(--panel-strong);
      }

      .controls {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
        flex-wrap: wrap;
      }

      label {
        display: grid;
        gap: 6px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 700;
      }

      input[type="number"] {
        width: 120px;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 10px 12px;
        font: inherit;
        background: white;
        color: var(--text);
        box-shadow: inset 0 0 0 1px rgba(53, 109, 255, 0.04);
      }

      textarea {
        width: 100%;
        min-height: 92px;
        resize: vertical;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px 16px;
        font: inherit;
        line-height: 1.5;
        background: white;
        color: var(--text);
        font-family: var(--font-reading);
        box-shadow: inset 0 0 0 1px rgba(53, 109, 255, 0.04);
      }

      textarea::placeholder {
        color: rgba(77, 98, 143, 0.88);
      }

      .actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 12px;
        flex-wrap: wrap;
      }

      .hint {
        color: var(--muted);
        font-size: 13px;
        font-family: var(--font-reading);
      }

      button {
        appearance: none;
        border: none;
        border-radius: 999px;
        padding: 12px 18px;
        font: inherit;
        cursor: pointer;
        transition: transform 140ms ease, opacity 140ms ease, background 140ms ease;
      }

      button:hover { transform: translateY(-1px); }
      button:disabled { opacity: 0.55; cursor: wait; transform: none; }

      .primary {
        background: linear-gradient(145deg, rgba(53, 109, 255, 0.96), rgba(78, 125, 255, 0.94));
        color: white;
        box-shadow: 0 0 0 1px rgba(53, 109, 255, 0.12), 0 12px 28px rgba(53, 85, 150, 0.16);
      }

      .secondary {
        background: rgba(248, 251, 255, 0.94);
        color: var(--accent);
        border: 1px solid var(--line);
      }

      a {
        color: inherit;
      }

      input,
      textarea,
      button {
        outline: none;
      }

      input:focus,
      textarea:focus,
      button:focus-visible {
        border-color: var(--line-strong);
        box-shadow: 0 0 0 1px rgba(53, 109, 255, 0.16), 0 0 0 6px rgba(53, 109, 255, 0.05);
      }

      @media (max-width: 720px) {
        .shell { width: calc(100vw - 16px); margin: 8px auto; }
        .hero, .messages, .composer { padding-left: 16px; padding-right: 16px; }
        .message { max-width: 100%; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="hero">
        <p class="eyebrow">Local UI</p>
        <div class="hero-row">
          <div>
            <h1>Gemma Local Chat</h1>
            <p>
              A small local browser chat app for Gemma 4. It keeps conversation state in the
              browser and sends each turn to your local backend.
            </p>
          </div>
          <a class="hero-link" href="__BENCHMARK_URL__" target="_blank" rel="noreferrer">
            <strong>Benchmark Note</strong>
            <span>Read the Gemma 4 benchmark post</span>
          </a>
        </div>
        <div class="meta">
          <div class="pill"><strong>Runtime</strong> <span id="runtime">Loading...</span></div>
          <div class="pill"><strong>Model</strong> <span id="model">Loading...</span></div>
          <div class="pill"><strong>Status</strong> <span id="status">Starting up</span></div>
        </div>
      </section>

      <section class="chat">
        <div id="messages" class="messages"></div>

        <div class="composer">
          <div class="controls">
            <label>
              Max Tokens
              <input id="maxTokens" type="number" min="32" max="4096" step="32" value="__DEFAULT_MAX_TOKENS__" />
            </label>
          </div>

          <textarea
            id="prompt"
            placeholder="Ask Gemma something. Shift+Enter adds a newline."
          ></textarea>

          <div class="actions">
            <div class="hint">
              Enter sends. Shift+Enter adds a newline. Clear resets local chat history.
            </div>
            <div style="display:flex; gap:10px;">
              <button id="clearButton" class="secondary" type="button">Clear</button>
              <button id="sendButton" class="primary" type="button">Send</button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <script>
      const messagesEl = document.getElementById("messages");
      const promptEl = document.getElementById("prompt");
      const sendButton = document.getElementById("sendButton");
      const clearButton = document.getElementById("clearButton");
      const maxTokensEl = document.getElementById("maxTokens");
      const statusEl = document.getElementById("status");
      const runtimeEl = document.getElementById("runtime");
      const modelEl = document.getElementById("model");

      const chat = [];
      let busy = false;

      function setBusy(nextBusy, label = "Ready") {
        busy = nextBusy;
        sendButton.disabled = nextBusy;
        clearButton.disabled = nextBusy;
        promptEl.disabled = nextBusy;
        statusEl.textContent = nextBusy ? label : "Ready";
      }

      function renderMessage(role, content) {
        const bubble = document.createElement("div");
        bubble.className = `message ${role}`;
        bubble.textContent = content;
        messagesEl.appendChild(bubble);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return bubble;
      }

      function resetChat() {
        chat.length = 0;
        messagesEl.innerHTML = "";
        renderMessage(
          "system",
          "Conversation cleared. Your backend stays loaded, so the next reply should still be warm."
        );
      }

      async function loadInfo() {
        const response = await fetch("/api/info");
        const data = await response.json();
        runtimeEl.textContent = data.runtime;
        modelEl.textContent = data.model;
        statusEl.textContent = "Ready";
        renderMessage(
          "system",
          `Connected to ${data.runtime} with ${data.model}. The server keeps a fixed system prompt and disables Gemma thinking mode where supported.`
        );
      }

      async function sendPrompt() {
        if (busy) return;
        const text = promptEl.value.trim();
        if (!text) return;

        renderMessage("user", text);
        chat.push({ role: "user", content: text });
        promptEl.value = "";

        const pending = renderMessage("assistant", "Thinking...");
        setBusy(true, "Generating");

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: chat,
              max_tokens: Number(maxTokensEl.value) || 512,
            }),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || "Request failed");
          }
          pending.textContent = data.reply;
          chat.push({ role: "assistant", content: data.reply });
        } catch (error) {
          pending.textContent = `Error: ${error.message}`;
          pending.classList.add("system");
        } finally {
          setBusy(false);
          promptEl.focus();
        }
      }

      sendButton.addEventListener("click", sendPrompt);
      clearButton.addEventListener("click", resetChat);
      promptEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendPrompt();
        }
      });

      loadInfo().catch((error) => {
        renderMessage("system", `Startup error: ${error.message}`);
        statusEl.textContent = "Error";
      });
    </script>
  </body>
</html>
"""


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    mlx_repo: str
    llama_repo: str
    llama_file: str


MODELS: dict[str, ModelSpec] = {
    "e2b": ModelSpec(
        slug="e2b",
        label="Gemma 4 E2B",
        mlx_repo="mlx-community/gemma-4-e2b-it-4bit",
        llama_repo="ggml-org/gemma-4-E2B-it-GGUF",
        llama_file="gemma-4-e2b-it-Q8_0.gguf",
    ),
    "e4b": ModelSpec(
        slug="e4b",
        label="Gemma 4 E4B",
        mlx_repo="mlx-community/gemma-4-e4b-it-4bit",
        llama_repo="ggml-org/gemma-4-E4B-it-GGUF",
        llama_file="gemma-4-e4b-it-Q4_K_M.gguf",
    ),
    "26b": ModelSpec(
        slug="26b",
        label="Gemma 4 26B A4B",
        mlx_repo="mlx-community/gemma-4-26b-a4b-it-4bit",
        llama_repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
        llama_file="gemma-4-26B-A4B-it-Q4_K_M.gguf",
    ),
    "31b": ModelSpec(
        slug="31b",
        label="Gemma 4 31B",
        mlx_repo="mlx-community/gemma-4-31b-it-4bit",
        llama_repo="ggml-org/gemma-4-31B-it-GGUF",
        llama_file="gemma-4-31B-it-Q4_K_M.gguf",
    ),
}


def log(message: str) -> None:
    print(f"[gemma-chat] {message}", flush=True)


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


class Backend:
    runtime_name: str
    display_model: str

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return


class MLXBackend(Backend):
    runtime_name = "MLX"

    def __init__(self, spec: ModelSpec, system_prompt: str):
        try:
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - error path
            raise RuntimeError(
                "MLX runtime requires `mlx-lm`. Run `./setup.sh` first."
            ) from exc

        log(f"Loading MLX model {spec.mlx_repo}")
        self.model, self.tokenizer = load(spec.mlx_repo)
        self.stream_generate = stream_generate
        self.sampler = make_sampler(temp=0.0, top_k=1, top_p=1.0)
        self.display_model = spec.label
        self.system_prompt = system_prompt
        self.lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        rendered_prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}, *messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pieces: list[str] = []
        with self.lock:
            for chunk in self.stream_generate(
                self.model,
                self.tokenizer,
                rendered_prompt,
                max_tokens=max_tokens,
                sampler=self.sampler,
            ):
                if chunk.text:
                    pieces.append(chunk.text)
        return "".join(pieces).strip()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()


class LlamaCppBackend(Backend):
    runtime_name = "llama.cpp"

    def __init__(
        self,
        spec: ModelSpec,
        system_prompt: str,
        llama_url: str | None,
        auto_start: bool,
        ctx_size: int,
        llama_port: int,
    ):
        self.display_model = spec.label
        self.system_prompt = system_prompt
        self.llama_url = llama_url or f"http://{LLAMA_INTERNAL_HOST}:{llama_port}"
        self.session = make_session()
        self.proc: subprocess.Popen[Any] | None = None

        if auto_start:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - error path
                raise RuntimeError(
                    "Auto-starting llama.cpp requires `huggingface_hub`. Run `./setup.sh` first."
                ) from exc

            log(f"Downloading GGUF for {spec.label} from {spec.llama_repo}")
            model_path = hf_hub_download(repo_id=spec.llama_repo, filename=spec.llama_file)
            log_path = Path("/private/tmp") / f"gemma-local-chat-{spec.slug}.log"
            log_file = log_path.open("w")
            log(f"Starting llama-server on {self.llama_url}")
            self.proc = subprocess.Popen(
                [
                    "llama-server",
                    "--model",
                    model_path,
                    "--no-mmproj",
                    "--chat-template-kwargs",
                    '{"enable_thinking": false}',
                    "--host",
                    LLAMA_INTERNAL_HOST,
                    "--port",
                    str(llama_port),
                    "--ctx-size",
                    str(ctx_size),
                    "--flash-attn",
                    "on",
                ],
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

    def chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        payload = {
            "model": "gemma4",
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "stream": False,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 123,
            "max_tokens": max_tokens,
        }
        response = self.session.post(
            f"{self.llama_url}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        cleanup_process(self.proc)


class AppServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], backend: Backend, default_max_tokens: int):
        super().__init__(server_address, ChatHandler)
        self.backend = backend
        self.default_max_tokens = default_max_tokens


class ChatHandler(BaseHTTPRequestHandler):
    server: AppServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            page = (
                HTML.replace("__DEFAULT_MAX_TOKENS__", str(self.server.default_max_tokens))
                .replace("__BENCHMARK_URL__", BENCHMARK_URL)
            )
            self._send_html(page)
            return
        if self.path == "/api/info":
            self._send_json(
                {
                    "runtime": self.server.backend.runtime_name,
                    "model": self.server.backend.display_model,
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_json({"error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens", self.server.default_max_tokens))

        if not isinstance(messages, list):
            self._send_json({"error": "`messages` must be a list"}, status=400)
            return

        clean_messages: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                self._send_json({"error": "Each message must be an object"}, status=400)
                return
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                self._send_json(
                    {"error": "Each message needs a `role` of `user` or `assistant` and string `content`"},
                    status=400,
                )
                return
            clean_messages.append({"role": role, "content": content})

        try:
            reply = self.server.backend.chat(clean_messages, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json({"reply": reply})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a small local chat app for Gemma 4.")
    parser.add_argument("--runtime", choices=["mlx", "llama"], default="mlx")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="26b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--llama-url", default=None, help="Existing llama-server URL, for example http://127.0.0.1:8080")
    parser.add_argument("--no-auto-start-llama", action="store_true", help="Use an existing llama-server instead of starting one")
    parser.add_argument("--llama-port", type=int, default=LLAMA_INTERNAL_PORT)
    parser.add_argument("--ctx-size", type=int, default=16384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = MODELS[args.model]

    if args.runtime == "mlx":
        backend: Backend = MLXBackend(spec, system_prompt=args.system_prompt)
    else:
        backend = LlamaCppBackend(
            spec,
            system_prompt=args.system_prompt,
            llama_url=args.llama_url,
            auto_start=not args.no_auto_start_llama,
            ctx_size=args.ctx_size,
            llama_port=args.llama_port,
        )

    server = AppServer((args.host, args.port), backend, default_max_tokens=args.max_tokens)
    url = f"http://{args.host}:{args.port}"
    log(f"Serving chat app at {url}")
    log(f"Runtime: {backend.runtime_name}")
    log(f"Model: {backend.display_model}")

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
    finally:
        server.server_close()
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
