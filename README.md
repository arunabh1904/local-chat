# Local Chat

A tiny local browser chat app for Apple Silicon with switchable local model presets.

This repo started as a Gemma-only chat app. It now supports ad hoc weight reloads from the UI, so you can choose a preset from a dropdown and swap the active locally served model without restarting the app yourself.

The GitHub repo is now `local-chat`, and the app itself is branded as `Local Chat`.

It currently supports:

- `Gemma 4` on `MLX`
- `Gemma 4` on `llama.cpp`
- `Qwen 3.5` on `MLX` via `mlx-vlm` in text-only mode
- `Qwen 3.5` on `llama.cpp`
- `Qwen 3` on `MLX`
- `Qwen 3` on `llama.cpp`

The UI stays intentionally small. Chat history lives in the browser session, and switching presets clears the visible transcript while carrying a short summary forward into the next model.

## Quick Start

```bash
cd /Users/arunabhmishra/Code/local-chat
./setup.sh
./run.sh --preset gemma4-26b-a4b-mlx --open-browser
```

Then open [http://127.0.0.1:8099](http://127.0.0.1:8099).

If you want the direct entrypoint instead of `run.sh`, use:

```bash
./.venv/bin/python local_chat.py --preset gemma4-26b-a4b-mlx
```

## What Changed

- One dropdown now lists supported local presets.
- Changing the dropdown triggers a backend reload and weight swap.
- The server exposes preset metadata through `/api/info`.
- Qwen benchmark automation now lives in `benchmark_local_models.py`.
- `--preset` is the main startup flag.
- The old Gemma entrypoint still works as a compatibility shim:

```bash
python gemma_local_chat.py --runtime mlx --model 26b --open-browser
```

## Example Presets

```bash
./run.sh --list-presets
```

Useful starting points:

- `gemma4-26b-a4b-mlx`
- `gemma4-31b-mlx`
- `qwen35-9b-mlx`
- `qwen35-35b-a3b-llama`
- `qwen3-14b-mlx`
- `qwen3-32b-llama`

## Benchmarking

The benchmark runner reuses the same preset registry and backend code as the app.

Run the default Qwen matrix:

```bash
./.venv/bin/python benchmark_local_models.py
```

Run a smaller slice:

```bash
./.venv/bin/python benchmark_local_models.py --preset qwen35-4b-mlx --preset qwen3-14b-llama --suite short
```

Results are written under `benchmarks/results/` as JSON and Markdown.

## Notes

- `setup.sh` installs `mlx-lm`, `mlx-vlm`, plus `torch` and `torchvision`, because Qwen 3.5 MLX presets still instantiate the upstream Qwen3-VL processor stack even in text-only mode.
- The app is text-only in this pass. Qwen 3.5 is served without images.
- The first load for any preset can take a while because weights download on demand.
- The UI links to a family-specific benchmark post for the active preset.
- If you already had a `.venv` under the old `gemma-local-chat` path, rerun `./setup.sh` once after pulling the rename so path-bound virtualenv scripts are refreshed.
- As of April 4, 2026, the official local/open Qwen benchmark scope in this repo covers `Qwen 3.5` and `Qwen 3`. No official open local `Qwen 3.6` target was found.

## CLI

```bash
./run.sh --help
```

Useful flags:

- `--preset <preset-id>`
- `--runtime mlx|llama`
- `--model e2b|e4b|26b|31b`
- `--port 8099`
- `--max-tokens 512`
- `--open-browser`
- `--system-prompt "You are a concise assistant."`
- `--no-auto-start-llama`
- `--llama-url http://127.0.0.1:8080`
