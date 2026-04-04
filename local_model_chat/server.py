from __future__ import annotations

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends import BackendManager, BackendFactorySettings
from .model_cache import configure_model_cache
from .presets import (
    DEFAULT_PRESET_ID,
    all_presets,
    get_preset,
    resolve_initial_preset_id,
    runtime_model_help,
)


DEFAULT_MAX_TOKENS = 512


HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#f4f8ff" />
    <title>Local Chat</title>
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
        width: min(1040px, calc(100vw - 32px));
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
        max-width: 66ch;
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

      .hero-link strong,
      .pill strong {
        color: var(--accent);
        font-weight: 700;
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
        display: grid;
        gap: 12px;
        margin-bottom: 12px;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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

      input[type="number"],
      select,
      textarea {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 12px 14px;
        font: inherit;
        background: white;
        color: var(--text);
        box-shadow: inset 0 0 0 1px rgba(53, 109, 255, 0.04);
      }

      input[type="number"] {
        border-radius: 14px;
      }

      textarea {
        min-height: 92px;
        resize: vertical;
        line-height: 1.5;
        font-family: var(--font-reading);
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

      a { color: inherit; }

      input,
      select,
      textarea,
      button {
        outline: none;
      }

      input:focus,
      select:focus,
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
            <h1>Local Chat</h1>
            <p>
              A tiny browser chat app for local Apple Silicon inference. Pick a preset,
              reload the active local weights on demand, and keep continuity through a
              short carry-forward summary when you switch models.
            </p>
          </div>
          <a id="benchmarkLink" class="hero-link" href="#" target="_blank" rel="noreferrer">
            <strong>Benchmark Note</strong>
            <span id="benchmarkLabel">Read the local model benchmark post</span>
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
              Active Preset
              <select id="presetSelect"></select>
            </label>
            <label>
              Max Tokens
              <input id="maxTokens" type="number" min="32" max="4096" step="32" value="__DEFAULT_MAX_TOKENS__" />
            </label>
          </div>

          <textarea
            id="prompt"
            placeholder="Ask your local model something. Shift+Enter adds a newline."
          ></textarea>

          <div class="actions">
            <div class="hint">
              Enter sends. Shift+Enter adds a newline. Switching presets clears the visible
              transcript and carries a short summary into the next turn.
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
      const STORAGE_KEYS = {
        chat: "local-chat:chat",
        summary: "local-chat:summary",
        preset: "local-chat:preset",
      };

      const messagesEl = document.getElementById("messages");
      const promptEl = document.getElementById("prompt");
      const sendButton = document.getElementById("sendButton");
      const clearButton = document.getElementById("clearButton");
      const maxTokensEl = document.getElementById("maxTokens");
      const statusEl = document.getElementById("status");
      const runtimeEl = document.getElementById("runtime");
      const modelEl = document.getElementById("model");
      const presetSelectEl = document.getElementById("presetSelect");
      const benchmarkLinkEl = document.getElementById("benchmarkLink");
      const benchmarkLabelEl = document.getElementById("benchmarkLabel");

      let chat = [];
      let conversationSummary = "";
      let currentPresetId = "";
      let busy = false;
      let presetsLoaded = false;

      function readStoredJson(key, fallback) {
        const raw = sessionStorage.getItem(key);
        if (!raw) return fallback;
        try {
          return JSON.parse(raw);
        } catch (_) {
          return fallback;
        }
      }

      function persistState() {
        sessionStorage.setItem(STORAGE_KEYS.chat, JSON.stringify(chat));
        sessionStorage.setItem(STORAGE_KEYS.summary, conversationSummary);
        if (currentPresetId) {
          sessionStorage.setItem(STORAGE_KEYS.preset, currentPresetId);
        }
      }

      function setBusy(nextBusy, label = "Ready") {
        busy = nextBusy;
        sendButton.disabled = nextBusy;
        clearButton.disabled = nextBusy;
        promptEl.disabled = nextBusy;
        presetSelectEl.disabled = nextBusy;
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

      function resetVisibleChat(message) {
        chat = [];
        messagesEl.innerHTML = "";
        persistState();
        if (message) {
          renderMessage("system", message);
        }
      }

      function clearChat() {
        conversationSummary = "";
        resetVisibleChat(
          "Conversation and carry-forward summary cleared. The current backend stays loaded."
        );
      }

      function restoreMessages() {
        messagesEl.innerHTML = "";
        for (const item of chat) {
          renderMessage(item.role, item.content);
        }
        if (!chat.length) {
          renderMessage(
            "system",
            "Connected. Pick a preset, send a prompt, and switch models whenever you want."
          );
        }
      }

      function populatePresets(presets) {
        if (presetsLoaded) return;
        const groups = new Map();
        for (const preset of presets) {
          if (!groups.has(preset.family)) {
            groups.set(preset.family, []);
          }
          groups.get(preset.family).push(preset);
        }

        presetSelectEl.innerHTML = "";
        for (const [family, items] of groups.entries()) {
          const optgroup = document.createElement("optgroup");
          optgroup.label = family;
          for (const preset of items) {
            const option = document.createElement("option");
            option.value = preset.id;
            option.textContent = `${preset.label} / ${preset.runtime_label}`;
            optgroup.appendChild(option);
          }
          presetSelectEl.appendChild(optgroup);
        }
        presetsLoaded = true;
      }

      function syncInfo(info) {
        populatePresets(info.presets);
        runtimeEl.textContent = info.current_preset.runtime_label;
        modelEl.textContent = info.current_preset.label;
        statusEl.textContent = info.state === "ready" ? "Ready" : info.state;
        currentPresetId = info.current_preset.id;
        presetSelectEl.value = currentPresetId;
        benchmarkLinkEl.href = info.benchmark_url;
        benchmarkLabelEl.textContent = `Read the ${info.current_preset.family} benchmark post`;
        persistState();
      }

      async function loadInfo() {
        const response = await fetch("/api/info");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to load app info");
        }

        const storedPreset = sessionStorage.getItem(STORAGE_KEYS.preset);
        chat = readStoredJson(STORAGE_KEYS.chat, []);
        conversationSummary = sessionStorage.getItem(STORAGE_KEYS.summary) || "";

        if (storedPreset && storedPreset !== data.current_preset.id) {
          chat = [];
          conversationSummary = "";
        }

        syncInfo(data);
        restoreMessages();
      }

      async function sendPrompt() {
        if (busy) return;
        const text = promptEl.value.trim();
        if (!text) return;

        renderMessage("user", text);
        chat.push({ role: "user", content: text });
        persistState();
        promptEl.value = "";

        const pending = renderMessage("assistant", "Thinking...");
        setBusy(true, "Generating");

        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: chat,
              conversation_summary: conversationSummary,
              max_tokens: Number(maxTokensEl.value) || 512,
            }),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || "Request failed");
          }
          pending.textContent = data.reply;
          chat.push({ role: "assistant", content: data.reply });
          persistState();
        } catch (error) {
          pending.textContent = `Error: ${error.message}`;
          pending.classList.add("system");
        } finally {
          setBusy(false);
          promptEl.focus();
        }
      }

      async function switchPreset(nextPresetId) {
        if (busy || !nextPresetId || nextPresetId === currentPresetId) {
          presetSelectEl.value = currentPresetId;
          return;
        }

        const previousPresetId = currentPresetId;
        setBusy(true, "Reloading");
        try {
          const response = await fetch("/api/switch-preset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              preset_id: nextPresetId,
              messages: chat,
              conversation_summary: conversationSummary,
            }),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.error || "Preset switch failed");
          }

          conversationSummary = data.conversation_summary || "";
          chat = [];
          syncInfo(data);
          messagesEl.innerHTML = "";
          const summaryText = conversationSummary
            ? `Carry-forward summary:\\n${conversationSummary}`
            : "No summary was needed because there was no conversation context yet.";
          renderMessage(
            "system",
            `Switched presets. Visible transcript cleared.\\n\\n${summaryText}`
          );
          persistState();
        } catch (error) {
          currentPresetId = previousPresetId;
          presetSelectEl.value = previousPresetId;
          renderMessage("system", `Preset switch failed: ${error.message}`);
        } finally {
          setBusy(false);
        }
      }

      sendButton.addEventListener("click", sendPrompt);
      clearButton.addEventListener("click", clearChat);
      presetSelectEl.addEventListener("change", (event) => {
        switchPreset(event.target.value);
      });
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


class AppServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: BackendManager,
        default_max_tokens: int,
    ):
        super().__init__(server_address, ChatHandler)
        self.manager = manager
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

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON") from exc

    def _validate_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("`messages` must be a list")

        clean_messages: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("Each message must be an object")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError(
                    "Each message needs a `role` of `user` or `assistant` and string `content`"
                )
            clean_messages.append({"role": role, "content": content})
        return clean_messages

    def _validate_summary(self, payload: dict[str, Any]) -> str:
        value = payload.get("conversation_summary", "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("`conversation_summary` must be a string")
        return value

    def _info_payload(self) -> dict[str, Any]:
        snapshot = self.server.manager.snapshot()
        return {
            **snapshot,
            "presets": [preset.to_public_dict() for preset in all_presets()],
        }

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            page = HTML.replace("__DEFAULT_MAX_TOKENS__", str(self.server.default_max_tokens))
            self._send_html(page)
            return
        if self.path == "/api/info":
            self._send_json(self._info_payload())
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/chat", "/api/switch-preset"}:
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
            messages = self._validate_messages(payload)
            conversation_summary = self._validate_summary(payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/chat":
            max_tokens = int(payload.get("max_tokens", self.server.default_max_tokens))
            try:
                result = self.server.manager.chat(
                    messages=messages,
                    conversation_summary=conversation_summary,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)
                return
            self._send_json({"reply": result.text})
            return

        preset_id = payload.get("preset_id")
        if not isinstance(preset_id, str) or not preset_id:
            self._send_json({"error": "`preset_id` must be a non-empty string"}, status=400)
            return

        try:
            carried_summary = self.server.manager.switch_preset(
                preset_id=preset_id,
                messages=messages,
                conversation_summary=conversation_summary,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return

        snapshot = self._info_payload()
        snapshot["conversation_summary"] = carried_summary
        self._send_json(snapshot)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Local Chat with switchable local model presets."
    )
    parser.add_argument("--preset", default=None, help="Initial preset id")
    parser.add_argument(
        "--runtime",
        choices=["mlx", "llama"],
        default=None,
        help="Legacy Gemma-only runtime alias",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Legacy Gemma-only model alias ({runtime_model_help()})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--llama-url",
        default=None,
        help="Existing llama-server URL, for example http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--no-auto-start-llama",
        action="store_true",
        help="Use an existing llama-server instead of starting one",
    )
    parser.add_argument("--llama-port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Cache root for downloaded model artifacts. Defaults to a sibling `models/` directory.",
    )
    parser.add_argument("--list-presets", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def main() -> int:
    args = parse_args()

    if args.list_presets:
        for preset in all_presets():
            print(f"{preset.id}\t{preset.family}\t{preset.runtime_label}\t{preset.label}")
        return 0

    preset_id = resolve_initial_preset_id(args.preset, args.runtime, args.model)
    initial_preset = get_preset(preset_id)
    model_cache_dir = configure_model_cache(args.model_cache_dir)
    settings = BackendFactorySettings(
        system_prompt=args.system_prompt or "You are a concise, helpful assistant. Answer directly and clearly.",
        llama_url=args.llama_url,
        auto_start_llama=not args.no_auto_start_llama,
        ctx_size=args.ctx_size,
        llama_port=args.llama_port,
        model_cache_dir=str(model_cache_dir),
    )
    manager = BackendManager(initial_preset=initial_preset, settings=settings)
    server = AppServer((args.host, args.port), manager, default_max_tokens=args.max_tokens)
    url = f"http://{args.host}:{args.port}"
    print(f"[local-chat] Serving chat app at {url}", flush=True)
    print(
        f"[local-chat] Initial preset: {initial_preset.label} ({initial_preset.runtime_label})",
        flush=True,
    )
    print(f"[local-chat] Model cache: {model_cache_dir}", flush=True)

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[local-chat] Shutting down", flush=True)
    finally:
        server.server_close()
        manager.close()
    return 0
