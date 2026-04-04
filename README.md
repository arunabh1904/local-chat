# Gemma Local Chat

A tiny local browser chat app for Gemma 4 on Apple Silicon.

The UI is intentionally light and minimal, with one small link back to the published Gemma 4 benchmark post.

It supports:

- `MLX` for the most Apple-native path
- `llama.cpp` for a lower-level local runtime

It does not try to be a full agent framework. It is a small, local, one-file chat app you can run and talk to from your browser.

## Quick Start

```bash
cd /Users/arunabhmishra/Code/gemma-local-chat
./setup.sh
./run.sh --runtime mlx --model 26b --open-browser
```

Then open [http://127.0.0.1:8099](http://127.0.0.1:8099).

`setup.sh` prefers Homebrew Python `3.12+` when available, because current `mlx` packages do not install cleanly on the macOS system Python `3.9`.
It also installs a known-good `mlx-lm` Git commit because the current PyPI release lags Gemma 4 support.

## Recommended Models

On a `64 GB` Apple Silicon machine:

- `26b` is the best everyday balance
- `31b` is the strongest model that still fits comfortably
- `e2b` is the fastest smoke-test option

## Usage

Start with MLX:

```bash
./run.sh --runtime mlx --model 26b --open-browser
```

Try the larger model:

```bash
./run.sh --runtime mlx --model 31b --open-browser
```

Use llama.cpp with auto-start:

```bash
brew install llama.cpp
./run.sh --runtime llama --model 26b --open-browser
```

Use an already running `llama-server`:

```bash
./run.sh \
  --runtime llama \
  --model 26b \
  --no-auto-start-llama \
  --llama-url http://127.0.0.1:8080 \
  --open-browser
```

## Notes

- The app keeps chat history in the browser session and sends the full conversation back on each turn.
- It uses a fixed system prompt and disables Gemma thinking mode where the backend supports that.
- This repo currently focuses on `MLX` and `llama.cpp`. I did not include `Ollama` because Gemma 4 was unstable on the tested `M5 Max` setup.
- The first run for any model may take a while because the weights have to download locally before the UI becomes responsive.

## Troubleshooting

- If `./setup.sh` fails on macOS system Python, install Homebrew Python and rerun. The script already prefers `python3.12+` when it is available.
- If `--runtime llama` fails immediately, make sure `llama-server` exists on your `PATH`:

```bash
brew install llama.cpp
```

- If `MLX` loads but a larger model feels sluggish, try `26b` first before jumping to `31b`.
- If you want the fastest smoke test, start with `e2b`.

## CLI Options

```bash
./run.sh --help
```

Useful flags:

- `--runtime mlx|llama`
- `--model e2b|e4b|26b|31b`
- `--port 8099`
- `--max-tokens 512`
- `--open-browser`
- `--system-prompt "You are a concise assistant."`
