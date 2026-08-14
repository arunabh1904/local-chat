from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends import BackendManager, BackendFactorySettings, ConversationImage
from .model_cache import configure_model_cache
from .presets import (
    DEFAULT_PRESET_ID,
    DEFAULT_VISION_PRESET_ID,
    all_presets,
    get_preset,
    resolve_initial_preset_id,
)


# ``-1`` is the native "generate until EOS" value for both llama.cpp and the
# MLX generation path.  Keep the control available to API/CLI callers, but do
# not impose an arbitrary reply-length ceiling in the chat UI.
DEFAULT_MAX_TOKENS = -1


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

      .notice {
        display: none;
        padding: 18px 20px;
        border: 2px solid var(--line-strong);
        border-radius: 16px;
        background: rgba(248, 251, 255, 0.96);
        color: var(--text);
        font-size: 16px;
        font-weight: 700;
        line-height: 1.5;
        font-family: var(--font-reading);
        box-shadow: 0 16px 36px rgba(53, 85, 150, 0.14);
      }

      .notice.visible {
        display: block;
      }

      .notice.downloading {
        background: #fee2e2;
        border-color: #b91c1c;
        color: #7f1d1d;
        box-shadow: 0 18px 42px rgba(185, 28, 28, 0.2);
      }

      .notice.error {
        background: #fecaca;
        border-color: #991b1b;
        color: #7f1d1d;
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

      .dropzone {
        margin-top: 12px;
        padding: 16px;
        border: 1.5px dashed var(--line-strong);
        border-radius: 18px;
        background: rgba(247, 250, 255, 0.86);
        color: var(--muted);
        display: grid;
        gap: 10px;
        transition:
          border-color 0.16s ease,
          background-color 0.16s ease,
          transform 0.16s ease;
      }

      .dropzone.active {
        border-color: var(--accent);
        background: rgba(232, 241, 255, 0.92);
        transform: translateY(-1px);
      }

      .dropzone.disabled {
        opacity: 0.65;
      }

      .dropzone-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
      }

      .dropzone-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .dropzone-copy {
        font-family: var(--font-reading);
        font-size: 14px;
        line-height: 1.55;
      }

      .attachment-preview {
        display: none;
        gap: 14px;
        align-items: center;
        grid-template-columns: 112px minmax(0, 1fr);
      }

      .attachment-preview.visible {
        display: grid;
      }

      .attachment-preview img {
        width: 112px;
        height: 112px;
        object-fit: cover;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: white;
      }

      .attachment-meta {
        display: grid;
        gap: 8px;
        min-width: 0;
      }

      .attachment-name {
        color: var(--text);
        font-weight: 700;
        font-size: 15px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .attachment-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
        font-family: var(--font-reading);
      }

      .message-image-note {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(248, 251, 255, 0.88);
        border: 1px solid rgba(53, 109, 255, 0.18);
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
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
        .attachment-preview,
        .attachment-preview.visible { grid-template-columns: 1fr; }
        .attachment-preview img { width: 100%; height: auto; max-height: 240px; }
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
              A tiny browser chat app for local Apple Silicon inference. It starts on
              Muse Glimmer with full Metal offload, native vision, and DFlash speculative
              decoding, then keeps continuity through a short carry-forward summary when
              you switch presets.
            </p>
          </div>
          <a id="benchmarkLink" class="hero-link" href="#" target="_blank" rel="noreferrer">
            <strong>Benchmark Note</strong>
            <span id="benchmarkLabel">Read the local model benchmark post</span>
          </a>
        </div>
        <div class="controls top-controls">
          <label>
            <span class="control-label">Switch model</span>
            <select id="presetSelect" aria-label="Switch active model" title="Switch active model"></select>
          </label>
        </div>
        <div class="meta">
          <div class="pill"><strong>Runtime</strong> <span id="runtime">Loading...</span></div>
          <div class="pill"><strong>Model</strong> <span id="model">Loading...</span></div>
          <div class="pill"><strong>Status</strong> <span id="status">Starting up</span></div>
        </div>
        <div id="cacheNotice" class="notice"></div>
      </section>

      <section class="chat">
        <div id="messages" class="messages"></div>

        <div class="composer">
          <textarea
            id="prompt"
            placeholder="Ask your local model something. Shift+Enter adds a newline."
          ></textarea>

          <div id="imageDropzone" class="dropzone">
            <div class="dropzone-header">
              <div>
                <div class="dropzone-title">Image Attachment</div>
                <div id="dropzoneCopy" class="dropzone-copy">
                  Drag and drop one image here, or click to browse.
                </div>
              </div>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <button id="browseImageButton" class="secondary" type="button">Choose Image</button>
                <button id="removeImageButton" class="secondary" type="button">Remove Image</button>
              </div>
            </div>
            <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp" hidden />
            <div id="attachmentPreview" class="attachment-preview">
              <img id="attachmentImage" alt="Attached preview" />
              <div class="attachment-meta">
                <div id="attachmentName" class="attachment-name"></div>
                <div id="attachmentNote" class="attachment-note"></div>
              </div>
            </div>
          </div>

          <div class="actions">
            <div class="hint">
              Enter sends. Shift+Enter adds a newline. Switching presets clears the visible
              transcript, carries a short summary into the next turn, and clears any attached image.
            </div>
            <div style="display:flex; gap:10px;">
              <button id="attachButton" class="secondary attach-button" type="button" aria-label="Attach image">+</button>
              <button id="clearButton" class="secondary" type="button">Clear</button>
              <button id="sendButton" class="primary" type="button">Send</button>
            </div>
          </div>
          <div id="metrics" class="metrics" aria-live="polite"></div>
        </div>
      </section>
    </div>

    <script>
      const STORAGE_KEYS = {
        chat: "local-chat:chat",
        summary: "local-chat:summary",
        preset: "local-chat:preset",
      };
      const INFO_TIMEOUT_MS = 8000;
      const INFO_RETRY_ATTEMPTS = 4;
      const INFO_RETRY_DELAY_MS = 700;
      const CHAT_TIMEOUT_MS = 60 * 60 * 1000;
      const SUPPORTED_IMAGE_TYPES = new Map([
        ["image/jpeg", "JPEG"],
        ["image/png", "PNG"],
        ["image/webp", "WebP"],
      ]);
      const IMAGE_TYPE_BY_EXTENSION = new Map([
        ["jpg", "image/jpeg"],
        ["jpeg", "image/jpeg"],
        ["png", "image/png"],
        ["webp", "image/webp"],
      ]);

      const messagesEl = document.getElementById("messages");
      const composerEl = document.querySelector(".composer");
      const promptEl = document.getElementById("prompt");
      const sendButton = document.getElementById("sendButton");
      const attachButton = document.getElementById("attachButton");
      const clearButton = document.getElementById("clearButton");
      const statusEl = document.getElementById("status");
      const cacheNoticeEl = document.getElementById("cacheNotice");
      const runtimeEl = document.getElementById("runtime");
      const modelEl = document.getElementById("model");
      const presetSelectEl = document.getElementById("presetSelect");
      const benchmarkLinkEl = document.getElementById("benchmarkLink");
      const benchmarkLabelEl = document.getElementById("benchmarkLabel");
      const imageDropzoneEl = document.getElementById("imageDropzone");
      const dropzoneCopyEl = document.getElementById("dropzoneCopy");
      const browseImageButtonEl = document.getElementById("browseImageButton");
      const removeImageButtonEl = document.getElementById("removeImageButton");
      const imageInputEl = document.getElementById("imageInput");
      const attachmentPreviewEl = document.getElementById("attachmentPreview");
      const attachmentImageEl = document.getElementById("attachmentImage");
      const attachmentNameEl = document.getElementById("attachmentName");
      const attachmentNoteEl = document.getElementById("attachmentNote");
      const metricsEl = document.getElementById("metrics");

      let chat = [];
      let conversationSummary = "";
      let currentPresetId = "";
      let currentPresetSupportsImages = false;
      let imageChatAvailable = false;
      let visionPresetLabel = "the vision preset";
      let modelCacheDir = "";
      let activeImage = null;
      let busy = false;
      let presetsLoaded = false;
      let messagesRestored = false;

      new ResizeObserver(() => {
        messagesEl.style.paddingBottom = `${composerEl.offsetHeight + 34}px`;
      }).observe(composerEl);

      function delay(ms) {
        return new Promise((resolve) => window.setTimeout(resolve, ms));
      }

      async function fetchJson(url, options = {}, timeoutMs = CHAT_TIMEOUT_MS) {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
          const response = await fetch(url, { ...options, signal: controller.signal });
          let data = {};
          try {
            data = await response.json();
          } catch (_) {
            data = {};
          }
          if (!response.ok) {
            throw new Error(data.error || `Request failed with HTTP ${response.status}`);
          }
          return data;
        } catch (error) {
          if (error.name === "AbortError") {
            throw new Error("The local server did not respond before the request timed out");
          }
          throw error;
        } finally {
          window.clearTimeout(timeoutId);
        }
      }

      async function fetchNdjson(url, options, onEvent) {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
        try {
          const response = await fetch(url, { ...options, signal: controller.signal });
          if (!response.ok || !response.body) {
            throw new Error(`Request failed with HTTP ${response.status}`);
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffered = "";
          let finalEvent = null;
          while (true) {
            const { value, done } = await reader.read();
            buffered += decoder.decode(value || new Uint8Array(), { stream: !done });
            const lines = buffered.split("\\n");
            buffered = lines.pop() || "";
            for (const line of lines) {
              if (!line.trim()) continue;
              const event = JSON.parse(line);
              if (event.type === "error") throw new Error(event.error || "Generation failed");
              onEvent(event);
              if (event.type === "done") finalEvent = event;
            }
            if (done) break;
          }
          if (buffered.trim()) {
            const event = JSON.parse(buffered);
            if (event.type === "error") throw new Error(event.error || "Generation failed");
            onEvent(event);
            if (event.type === "done") finalEvent = event;
          }
          if (!finalEvent) throw new Error("The stream ended before generation completed");
          return finalEvent;
        } catch (error) {
          if (error.name === "AbortError") {
            throw new Error("The local generation timed out");
          }
          throw error;
        } finally {
          window.clearTimeout(timeoutId);
        }
      }

      function formatMetrics(metrics, browserTtftMs) {
        const parts = [];
        const ttft = metrics && metrics.ttft_ms != null ? metrics.ttft_ms : browserTtftMs;
        if (ttft != null) parts.push(`TTFT ${Math.round(ttft)} ms`);
        if (metrics && metrics.average_tokens_per_s != null) {
          parts.push(`${metrics.average_tokens_per_s.toFixed(1)} tok/s`);
        }
        if (metrics && metrics.generation_tokens != null) {
          parts.push(`${metrics.generation_tokens} tokens`);
        }
        return parts.join("  ·  ");
      }

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

      function updateAttachmentUI() {
        const imagesEnabled = imageChatAvailable && !busy;
        imageDropzoneEl.classList.toggle("disabled", !imageChatAvailable);
        browseImageButtonEl.disabled = !imagesEnabled;
        imageInputEl.disabled = !imagesEnabled;
        removeImageButtonEl.disabled = busy || !activeImage;
        attachButton.disabled = !imagesEnabled;

        if (activeImage) {
          attachmentPreviewEl.classList.add("visible");
          attachmentImageEl.src = activeImage.data_url;
          attachmentNameEl.textContent = activeImage.name;
          attachmentNoteEl.textContent = currentPresetSupportsImages
            ? "This image stays attached for upcoming turns until you clear it or switch presets."
            : `This image will use ${visionPresetLabel} because the active model is text-only.`;
        } else {
          attachmentPreviewEl.classList.remove("visible");
          attachmentImageEl.removeAttribute("src");
          attachmentNameEl.textContent = "";
          attachmentNoteEl.textContent = "";
        }

        if (imageChatAvailable) {
          dropzoneCopyEl.textContent = activeImage
            ? "The current conversation image is ready. Drop a new one to replace it."
            : currentPresetSupportsImages
              ? "Drag and drop one image here. The active model handles text and images."
              : `Drag and drop one image here. Text uses the active model; images use ${visionPresetLabel} when needed.`;
        } else {
          dropzoneCopyEl.textContent = "Image chat is unavailable because no vision preset is configured.";
        }
      }

      function imageUnavailableMessage() {
        if (busy) {
          return "Wait for the current local request to finish before attaching an image.";
        }
        if (!currentPresetId) {
          return "Presets are still loading. Try attaching the image again once the app is ready.";
        }
        return "Image chat is unavailable because no vision preset is configured.";
      }

      function showImageUnavailable() {
        renderMessage("system", imageUnavailableMessage());
      }

      function eventHasFiles(event) {
        const types = event.dataTransfer ? event.dataTransfer.types : [];
        return Array.from(types).includes("Files");
      }

      function modelCacheDescription() {
        return modelCacheDir ? `${modelCacheDir}/huggingface/hub` : "the configured model cache";
      }

      function setBusy(nextBusy, label = "Ready") {
        busy = nextBusy;
        sendButton.disabled = nextBusy;
        clearButton.disabled = nextBusy;
        promptEl.disabled = nextBusy;
        presetSelectEl.disabled = nextBusy;
        statusEl.textContent = nextBusy ? label : "Ready";
        updateAttachmentUI();
      }

      function renderMessage(role, content, options = {}) {
        const bubble = document.createElement("div");
        bubble.className = `message ${role}`;
        if (options.imageName) {
          const imageNote = document.createElement("div");
          imageNote.className = "message-image-note";
          imageNote.textContent = `Image: ${options.imageName}`;
          bubble.appendChild(imageNote);
        }
        const body = document.createElement("div");
        body.textContent = content;
        bubble.appendChild(body);
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
        activeImage = null;
        resetVisibleChat(
          "Conversation, image attachment, and carry-forward summary cleared. The current backend stays loaded."
        );
        updateAttachmentUI();
      }

      function restoreMessages() {
        messagesRestored = true;
        messagesEl.innerHTML = "";
        for (const item of chat) {
          renderMessage(item.role, item.content, { imageName: item.image_name || "" });
        }
        if (!chat.length) {
          renderMessage(
            "system",
            currentPresetSupportsImages
              ? "Connected. The active model handles text and images; drag in an image whenever you need vision."
              : "Connected. Text uses the active model, and image turns use the configured vision preset when needed."
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
            option.textContent = preset.supports_images
              ? `${preset.label} / ${preset.runtime_label} / Image chat`
              : `${preset.label} / ${preset.runtime_label}`;
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
        currentPresetSupportsImages = Boolean(info.current_preset.supports_images);
        imageChatAvailable = Boolean(info.image_chat_available);
        visionPresetLabel = info.vision_preset ? info.vision_preset.label : "the vision preset";
        modelCacheDir = info.model_cache_dir || "";
        presetSelectEl.value = currentPresetId;
        benchmarkLinkEl.href = info.benchmark_url;
        benchmarkLabelEl.textContent = `Read the ${info.current_preset.family} benchmark post`;
        const cacheNotice = info.cache_notice || "";
        const cacheNoticeKind = cacheNotice.toLowerCase();
        cacheNoticeEl.textContent = cacheNotice;
        cacheNoticeEl.classList.toggle("visible", Boolean(cacheNotice));
        cacheNoticeEl.classList.toggle("downloading", cacheNoticeKind.startsWith("downloading"));
        cacheNoticeEl.classList.toggle("error", cacheNoticeKind.startsWith("failed"));
        updateAttachmentUI();
        persistState();
      }

      function clearActiveImage({ announce = false } = {}) {
        if (!activeImage) return;
        const clearedName = activeImage.name;
        activeImage = null;
        imageInputEl.value = "";
        updateAttachmentUI();
        if (announce) {
          renderMessage("system", `Removed image attachment: ${clearedName}`);
        }
      }

      function fileToDataUrl(file) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => reject(new Error("Failed to read the selected image"));
          reader.readAsDataURL(file);
        });
      }

      function inferImageMediaType(file) {
        if (SUPPORTED_IMAGE_TYPES.has(file.type)) {
          return file.type;
        }
        const extension = (file.name || "").split(".").pop().toLowerCase();
        return IMAGE_TYPE_BY_EXTENSION.get(extension) || "";
      }

      function normalizeDataUrl(dataUrl, mediaType) {
        const prefix = `data:${mediaType};base64,`;
        if (dataUrl.startsWith(prefix)) {
          return dataUrl;
        }
        const separatorIndex = dataUrl.indexOf(",");
        if (separatorIndex === -1) {
          throw new Error("Failed to read the selected image as a base64 data URL");
        }
        return `${prefix}${dataUrl.slice(separatorIndex + 1)}`;
      }

      async function attachImageFile(file) {
        if (!file) return;
        const mediaType = inferImageMediaType(file);
        if (!mediaType) {
          throw new Error("Only JPEG, PNG, and WebP images are supported");
        }
        if (!imageChatAvailable) {
          throw new Error("Image chat is unavailable because no vision preset is configured.");
        }
        const dataUrl = await fileToDataUrl(file);
        activeImage = {
          name: file.name || "image",
          media_type: mediaType,
          data_url: normalizeDataUrl(dataUrl, mediaType),
        };
        updateAttachmentUI();
        const routeText = currentPresetSupportsImages
          ? "It will stay available for upcoming turns until you clear it or switch presets."
          : `Sending it will switch to ${visionPresetLabel}. If that model is not cached yet, it will download into ${modelCacheDescription()}.`;
        renderMessage(
          "system",
          `Attached image: ${activeImage.name}. ${routeText}`
        );
      }

      async function handleSelectedFiles(files) {
        if (!files || !files.length) return;
        const firstFile = Array.from(files).find((file) => Boolean(inferImageMediaType(file)));
        if (!firstFile) {
          throw new Error("Drop or choose a JPEG, PNG, or WebP image");
        }
        await attachImageFile(firstFile);
      }

      async function loadInfo() {
        if (!messagesRestored) {
          setBusy(true, "Connecting");
        }
        let data = null;
        let lastError = null;
        for (let attempt = 1; attempt <= INFO_RETRY_ATTEMPTS; attempt += 1) {
          try {
            statusEl.textContent = attempt === 1 ? "Connecting" : `Retrying ${attempt}/${INFO_RETRY_ATTEMPTS}`;
            data = await fetchJson("/api/info", {}, INFO_TIMEOUT_MS);
            break;
          } catch (error) {
            lastError = error;
            if (attempt < INFO_RETRY_ATTEMPTS) {
              await delay(INFO_RETRY_DELAY_MS);
            }
          }
        }
        if (!data) {
          setBusy(false);
          throw lastError || new Error("Failed to load app info");
        }

        const storedPreset = sessionStorage.getItem(STORAGE_KEYS.preset);
        chat = readStoredJson(STORAGE_KEYS.chat, []);
        conversationSummary = sessionStorage.getItem(STORAGE_KEYS.summary) || "";

        if (storedPreset && storedPreset !== data.current_preset.id) {
          chat = [];
          conversationSummary = "";
        }

        syncInfo(data);
        if (!messagesRestored) {
          restoreMessages();
        }
        if (data.state === "loading") {
          setBusy(true, "Loading model");
          window.setTimeout(() => {
            loadInfo().catch((error) => {
              setBusy(false);
              renderMessage("system", `Startup error: ${error.message}`);
              statusEl.textContent = "Error";
            });
          }, 1000);
          return;
        }
        setBusy(false);
        if (data.state === "error") {
          statusEl.textContent = "Error";
        }
      }

      async function sendPrompt() {
        if (busy) return;

        const rawText = promptEl.value.trim();
        const text = rawText || (activeImage ? "Describe the attached image briefly." : "");
        if (!text) return;

        renderMessage("user", text, { imageName: activeImage ? activeImage.name : "" });
        chat.push({
          role: "user",
          content: text,
          image_name: activeImage ? activeImage.name : undefined,
        });
        persistState();
        promptEl.value = "";

        const pending = renderMessage("assistant", "");
        const pendingBody = pending.lastElementChild;
        pending.classList.add("streaming");
        const routingToVision = Boolean(activeImage && !currentPresetSupportsImages);
        setBusy(true, routingToVision ? `Loading ${visionPresetLabel}` : "Generating");
        metricsEl.textContent = routingToVision ? `Loading ${visionPresetLabel}…` : "Waiting for first token…";

        try {
          const requestStarted = performance.now();
          let firstPaintAt = null;
          let streamedText = "";
          const data = await fetchNdjson("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: chat,
              conversation_summary: conversationSummary,
              conversation_image: activeImage,
            }),
          }, (event) => {
            if (event.type === "delta") {
              if (firstPaintAt === null) {
                firstPaintAt = performance.now();
                metricsEl.textContent = `TTFT ${Math.round(firstPaintAt - requestStarted)} ms  ·  streaming`;
              }
              streamedText += event.text || "";
              pendingBody.textContent = streamedText;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          });
          pending.classList.remove("streaming");
          pendingBody.textContent = data.reply || streamedText;
          if (data.info) {
            syncInfo(data.info);
          }
          metricsEl.textContent = formatMetrics(
            data.metrics || {},
            firstPaintAt === null ? null : firstPaintAt - requestStarted,
          );
          chat.push({ role: "assistant", content: data.reply || streamedText });
          persistState();
        } catch (error) {
          pending.classList.remove("streaming");
          pendingBody.textContent = `Error: ${error.message}`;
          pending.classList.add("system");
          metricsEl.textContent = "";
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
          const data = await fetchJson("/api/switch-preset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              preset_id: nextPresetId,
              messages: chat,
              conversation_summary: conversationSummary,
              conversation_image: activeImage,
            }),
          });

          conversationSummary = data.conversation_summary || "";
          const removedImageName = activeImage ? activeImage.name : "";
          chat = [];
          activeImage = null;
          imageInputEl.value = "";
          syncInfo(data);
          messagesEl.innerHTML = "";
          const summaryText = conversationSummary
            ? `Carry-forward summary:\\n${conversationSummary}`
            : "No summary was needed because there was no conversation context yet.";
          const imageText = removedImageName
            ? `\\n\\nAttached image cleared: ${removedImageName}`
            : "";
          renderMessage(
            "system",
            `Switched presets. Visible transcript cleared.\\n\\n${summaryText}${imageText}`
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
      browseImageButtonEl.addEventListener("click", () => {
        if (!busy && imageChatAvailable) {
          imageInputEl.click();
        }
      });
      attachButton.addEventListener("click", () => {
        if (!busy && imageChatAvailable) imageInputEl.click();
      });
      removeImageButtonEl.addEventListener("click", () => {
        clearActiveImage({ announce: true });
      });
      imageInputEl.addEventListener("change", async (event) => {
        try {
          await handleSelectedFiles(event.target.files);
        } catch (error) {
          renderMessage("system", `Image attach failed: ${error.message}`);
        } finally {
          imageInputEl.value = "";
        }
      });
      imageDropzoneEl.addEventListener("click", (event) => {
        if (event.target.closest("button")) {
          return;
        }
        if (!busy && imageChatAvailable) {
          imageInputEl.click();
        } else {
          showImageUnavailable();
        }
      });
      presetSelectEl.addEventListener("change", (event) => {
        switchPreset(event.target.value);
      });
      imageDropzoneEl.addEventListener("dragenter", (event) => {
        event.preventDefault();
        if (!busy && imageChatAvailable) {
          imageDropzoneEl.classList.add("active");
        }
      });
      imageDropzoneEl.addEventListener("dragover", (event) => {
        event.preventDefault();
        if (!busy && imageChatAvailable) {
          imageDropzoneEl.classList.add("active");
        }
      });
      imageDropzoneEl.addEventListener("dragleave", (event) => {
        if (!imageDropzoneEl.contains(event.relatedTarget)) {
          imageDropzoneEl.classList.remove("active");
        }
      });
      imageDropzoneEl.addEventListener("drop", async (event) => {
        event.preventDefault();
        imageDropzoneEl.classList.remove("active");
        if (busy || !imageChatAvailable) {
          showImageUnavailable();
          return;
        }
        try {
          await handleSelectedFiles(event.dataTransfer.files);
        } catch (error) {
          renderMessage("system", `Image attach failed: ${error.message}`);
        }
      });
      window.addEventListener("dragover", (event) => {
        if (eventHasFiles(event)) {
          event.preventDefault();
        }
      });
      window.addEventListener("drop", async (event) => {
        if (!eventHasFiles(event) || imageDropzoneEl.contains(event.target)) {
          return;
        }
        event.preventDefault();
        if (busy || !imageChatAvailable) {
          showImageUnavailable();
          return;
        }
        try {
          await handleSelectedFiles(event.dataTransfer.files);
        } catch (error) {
          renderMessage("system", `Image attach failed: ${error.message}`);
        }
      });
      promptEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendPrompt();
        }
      });

      loadInfo().catch((error) => {
        setBusy(false);
        renderMessage("system", `Startup error: ${error.message}`);
        statusEl.textContent = "Error";
      });
    </script>
  </body>
</html>
"""


MODERN_CSS = r"""
      :root {
        color-scheme: dark;
        --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        --font-heading: var(--font-body);
        --font-reading: var(--font-body);
        --bg: #0b0d12;
        --bg-secondary: #0b0d12;
        --text: #f5f7fb;
        --muted: #98a1b3;
        --accent: #8b5cf6;
        --accent-2: #22d3ee;
        --accent-soft: rgba(139, 92, 246, 0.13);
        --panel: rgba(17, 20, 28, 0.92);
        --panel-strong: rgba(20, 23, 32, 0.96);
        --line: rgba(255, 255, 255, 0.09);
        --line-strong: rgba(139, 92, 246, 0.52);
        --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
        --assistant: transparent;
        --user: #242936;
        --system: rgba(139, 92, 246, 0.08);
      }

      html, body { height: 100%; }
      body {
        min-height: 100%;
        overflow: hidden;
        background:
          radial-gradient(circle at 20% -10%, rgba(139, 92, 246, 0.18), transparent 35%),
          radial-gradient(circle at 90% 5%, rgba(34, 211, 238, 0.08), transparent 30%),
          #0b0d12;
        font-family: var(--font-body);
      }

      .shell {
        width: 100%;
        height: 100vh;
        margin: 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
      }

      .hero {
        min-height: 64px;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        border-bottom: 1px solid var(--line);
        background: rgba(11, 13, 18, 0.78);
        backdrop-filter: blur(22px);
        position: relative;
        z-index: 5;
      }

      .hero-row { display: flex; align-items: center; gap: 12px; }
      .eyebrow, .hero p, .hero-link { display: none; }
      .hero h1 {
        font-size: 16px;
        letter-spacing: -0.01em;
        font-weight: 650;
      }
      .hero h1::before {
        content: "";
        display: inline-block;
        width: 11px;
        height: 11px;
        margin-right: 10px;
        border-radius: 4px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 0 18px rgba(139, 92, 246, 0.55);
      }

      .meta { gap: 8px; margin-left: auto; }
      .meta .pill:nth-child(-n+2) { display: none; }
      .pill {
        min-height: 34px;
        padding: 7px 11px;
        background: rgba(255,255,255,.035);
        border-color: var(--line);
        color: var(--muted);
        font-size: 12px;
      }
      .pill strong { color: #c8cfdd; font-weight: 550; }
      .pill:last-child span::before {
        content: "";
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-right: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 10px rgba(52, 211, 153, .5);
      }

      .notice {
        position: absolute;
        top: 68px;
        left: 50%;
        width: min(620px, calc(100vw - 32px));
        transform: translateX(-50%);
        padding: 13px 16px 13px 44px;
        border-width: 1px;
        border-radius: 14px;
        background: rgba(28, 24, 45, .97);
        color: #ddd6fe;
        font-size: 13px;
        font-weight: 550;
        box-shadow: var(--shadow);
      }
      .notice::before {
        content: "";
        position: absolute;
        left: 17px;
        top: 16px;
        width: 12px;
        height: 12px;
        border: 2px solid rgba(196,181,253,.3);
        border-top-color: #c4b5fd;
        border-radius: 50%;
        animation: spin .75s linear infinite;
      }
      .notice.downloading, .notice.error {
        background: rgba(45, 25, 31, .97);
        border-color: rgba(248,113,113,.35);
        color: #fecaca;
      }
      @keyframes spin { to { transform: rotate(360deg); } }

      .chat {
        min-height: 0;
        height: 100%;
        grid-template-rows: minmax(0, 1fr) auto;
      }
      .messages {
        width: 100%;
        padding: 38px max(24px, calc((100vw - 820px) / 2)) 170px;
        gap: 28px;
        background: transparent;
        scroll-behavior: smooth;
      }
      .message {
        max-width: 100%;
        padding: 0;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        font-size: 15.5px;
        line-height: 1.72;
        letter-spacing: -.005em;
      }
      .message > div:last-child { max-width: 76ch; }
      .message.user {
        justify-self: end;
        width: fit-content;
        max-width: min(78%, 680px);
        padding: 11px 16px;
        border-radius: 20px 20px 6px 20px;
        background: var(--user);
        color: var(--text);
      }
      .message.assistant { color: #e9edf5; }
      .message.assistant::before {
        content: "Local";
        display: block;
        margin-bottom: 7px;
        color: #a78bfa;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
      }
      .message.assistant.streaming > div:last-child::after {
        content: "";
        display: inline-block;
        width: 7px;
        height: 16px;
        margin-left: 3px;
        vertical-align: -2px;
        border-radius: 2px;
        background: #a78bfa;
        animation: blink .8s steps(1) infinite;
      }
      @keyframes blink { 50% { opacity: .2; } }
      .message.system {
        justify-self: center;
        width: fit-content;
        max-width: 680px;
        padding: 9px 13px;
        border: 1px solid rgba(139,92,246,.16);
        border-radius: 12px;
        color: #aab2c2;
        font-size: 13px;
        text-align: center;
      }

      .composer {
        position: absolute;
        z-index: 4;
        left: 50%;
        bottom: 18px;
        width: min(780px, calc(100vw - 32px));
        transform: translateX(-50%);
        padding: 10px;
        border: 1px solid rgba(255,255,255,.11);
        border-radius: 24px;
        background: rgba(27, 30, 40, .93);
        box-shadow: 0 18px 55px rgba(0,0,0,.45), 0 0 0 1px rgba(139,92,246,.04);
        backdrop-filter: blur(22px);
      }
      .controls {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0 2px 6px;
      }
      .top-controls { margin: 0 auto 0 8px; }
      .controls.top-controls label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }
      .top-controls .control-label {
        display: inline;
        color: #98a1b3;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
      }
      .control-label { display: none; }
      .controls label { display: contents; }
      .controls label::first-line { font-size: 0; }
      .controls.top-controls label::first-line { font-size: 12px; }
      select, textarea {
        background: rgba(255,255,255,.045);
        border-color: transparent;
        color: var(--text);
        box-shadow: none;
      }
      select {
        width: auto;
        max-width: min(440px, 65vw);
        height: 34px;
        padding: 6px 32px 6px 10px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 600;
      }
      input[type="number"] {
        width: 82px;
        height: 34px;
        padding: 6px 9px;
        border-radius: 10px;
        font-size: 12px;
      }
      textarea {
        min-height: 56px;
        max-height: 190px;
        padding: 12px 54px 8px 12px;
        resize: none;
        font-size: 15px;
        line-height: 1.45;
      }
      textarea::placeholder { color: #747d90; }
      .actions { margin: 2px 2px 0; min-height: 34px; }
      .hint { display: none; }
      .actions > div:last-child {
        position: absolute;
        left: 14px;
        right: 14px;
        bottom: 14px;
        display: flex;
        align-items: center;
        pointer-events: none;
      }
      .actions > div:last-child button { pointer-events: auto; }
      button { padding: 8px 13px; font-size: 12px; font-weight: 650; }
      .primary {
        width: 36px;
        height: 36px;
        padding: 0;
        font-size: 0;
        background: linear-gradient(135deg, var(--accent), #6d4aff);
        box-shadow: 0 7px 18px rgba(109,74,255,.28);
      }
      .primary::after { content: "↑"; font-size: 21px; line-height: 1; }
      .secondary { background: rgba(255,255,255,.055); color: #c3cad7; border-color: var(--line); }
      #clearButton { margin-right: 0; }
      .attach-button {
        position: static;
        margin-right: auto;
        width: 34px;
        height: 34px;
        padding: 0;
        border-radius: 50%;
        font-size: 21px;
        line-height: 1;
      }
      .dropzone {
        display: none;
        margin: 5px 2px 7px;
        padding: 10px;
        border-radius: 14px;
        background: rgba(139,92,246,.06);
      }
      .dropzone:has(.attachment-preview.visible) { display: grid; }
      .dropzone-header { display: none; }
      .attachment-preview { grid-template-columns: 56px minmax(0,1fr); }
      .attachment-preview img { width: 56px; height: 56px; border-radius: 10px; }
      .message-image-note { margin: 0 0 7px; background: rgba(139,92,246,.1); color: #c4b5fd; }
      .metrics {
        min-height: 17px;
        margin: 3px 48px 0;
        text-align: center;
        color: #7f899c;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
      }
      input:focus, select:focus, textarea:focus, button:focus-visible {
        border-color: rgba(139,92,246,.45);
        box-shadow: 0 0 0 3px rgba(139,92,246,.1);
      }

      @media (max-width: 720px) {
        .shell { width: 100%; margin: 0; }
        .hero { min-height: 56px; padding: 8px 12px; }
        .meta .pill:nth-child(-n+2) { display: none; }
        .messages { padding: 26px 16px 170px; }
        .message.user { max-width: 88%; }
        .composer { bottom: 8px; width: calc(100vw - 16px); border-radius: 20px; }
        .hint { max-width: calc(100% - 105px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      }
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
    protocol_version = "HTTP/1.1"

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

    def _start_ndjson(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _send_ndjson_event(self, payload: dict[str, Any]) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.wfile.write(f"{len(body):X}\r\n".encode("ascii"))
        self.wfile.write(body)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _finish_ndjson(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

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

    def _validate_conversation_image(
        self,
        payload: dict[str, Any],
    ) -> ConversationImage | None:
        value = payload.get("conversation_image")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("`conversation_image` must be an object")

        name = value.get("name") or "image"
        media_type = value.get("media_type")
        data_url = value.get("data_url")
        if not isinstance(name, str):
            raise ValueError("`conversation_image.name` must be a string")
        if not isinstance(media_type, str) or not media_type.startswith("image/"):
            raise ValueError("`conversation_image.media_type` must be an image media type")
        if media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Only JPEG, PNG, and WebP images are supported")
        expected_prefix = f"data:{media_type};base64,"
        if not isinstance(data_url, str) or not data_url.startswith(expected_prefix):
            raise ValueError(
                "`conversation_image.data_url` must be a base64 data URL matching the media type"
            )
        return ConversationImage(name=name.strip() or "image", media_type=media_type, data_url=data_url)

    def _info_payload(self) -> dict[str, Any]:
        snapshot = self.server.manager.snapshot()
        return {
            **snapshot,
            "presets": [preset.to_public_dict() for preset in all_presets()],
        }

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            page = HTML.replace("</head>", f"<style>{MODERN_CSS}</style></head>")
            self._send_html(page)
            return
        if self.path == "/api/info":
            self._send_json(self._info_payload())
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/chat", "/api/chat/stream", "/api/switch-preset"}:
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
            messages = self._validate_messages(payload)
            conversation_summary = self._validate_summary(payload)
            conversation_image = self._validate_conversation_image(payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        if self.path == "/api/chat/stream":
            max_tokens = int(payload.get("max_tokens", self.server.default_max_tokens))
            self._start_ndjson()
            try:
                result = self.server.manager.stream_chat(
                    messages=messages,
                    conversation_summary=conversation_summary,
                    max_tokens=max_tokens,
                    conversation_image=conversation_image,
                    on_text=lambda text: self._send_ndjson_event(
                        {"type": "delta", "text": text}
                    ),
                )
                self._send_ndjson_event(
                    {
                        "type": "done",
                        "reply": result.text,
                        "metrics": asdict(result.metrics),
                        "info": self._info_payload(),
                    }
                )
                self._finish_ndjson()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                try:
                    self._send_ndjson_event({"type": "error", "error": str(exc)})
                    self._finish_ndjson()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        if self.path == "/api/chat":
            max_tokens = int(payload.get("max_tokens", self.server.default_max_tokens))
            try:
                result = self.server.manager.chat(
                    messages=messages,
                    conversation_summary=conversation_summary,
                    max_tokens=max_tokens,
                    conversation_image=conversation_image,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=500)
                return
            snapshot = self._info_payload()
            snapshot["reply"] = result.text
            self._send_json(snapshot)
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
                conversation_image=conversation_image,
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
        "--vision-preset",
        default=DEFAULT_VISION_PRESET_ID,
        help=(
            "Preset to use automatically for image turns when the active preset is text-only. "
            "Use an empty string to disable image routing."
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["mlx", "llama"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Default generated tokens per reply; -1 runs until the model stops (default: -1).",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Cache root for downloaded model artifacts. Defaults to `~/.cache`.",
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

    requested_preset = args.preset
    if not requested_preset and not args.runtime and not args.model:
        requested_preset = DEFAULT_PRESET_ID
    preset_id = resolve_initial_preset_id(requested_preset, args.runtime, args.model)
    initial_preset = get_preset(preset_id)
    vision_preset_id = args.vision_preset.strip() or None
    if vision_preset_id is not None and not get_preset(vision_preset_id).supports_images:
        raise ValueError(f"--vision-preset `{vision_preset_id}` must support images")
    model_cache_dir = configure_model_cache(args.model_cache_dir)
    settings = BackendFactorySettings(
        system_prompt=args.system_prompt or "You are a concise, helpful assistant. Answer directly and clearly.",
        ctx_size=args.ctx_size,
        model_cache_dir=str(model_cache_dir),
        vision_preset_id=vision_preset_id,
    )
    manager = BackendManager(initial_preset=initial_preset, settings=settings, autoload=False)
    server = AppServer((args.host, args.port), manager, default_max_tokens=args.max_tokens)
    url = f"http://{args.host}:{args.port}"
    print(f"[local-chat] Serving chat app at {url}", flush=True)
    print(
        f"[local-chat] Initial preset: {initial_preset.label} ({initial_preset.runtime_label})",
        flush=True,
    )
    print(f"[local-chat] Model cache: {model_cache_dir}", flush=True)
    if vision_preset_id:
        vision_preset = get_preset(vision_preset_id)
        print(
            f"[local-chat] Image turns auto-route to: {vision_preset.label} ({vision_preset.runtime_label})",
            flush=True,
        )
    print(f"[local-chat] {manager.snapshot()['cache_notice']}", flush=True)
    manager.start_loading()

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
