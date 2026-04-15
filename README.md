# Local Chat

A tiny local browser chat app for Apple Silicon with switchable local model presets.

This repo started as a Gemma-only chat app. It now supports ad hoc weight reloads from the UI, so you can choose a preset from a dropdown and swap the active locally served model without restarting the app yourself.

The GitHub repo is now `local-chat`, and the app itself is branded as `Local Chat`.

It currently supports:

- `Gemma 4` on `MLX`
- `Qwen 3.5` on `MLX` with image chat support
- `Qwen 3` on `MLX`

The UI stays intentionally small. Chat history lives in the browser session, and switching presets clears the visible transcript while carrying a short summary forward into the next model. Qwen 3.5 presets also let you drag and drop one image into the active conversation.

Model artifacts are cached on disk under `~/.cache` by default, so preset switches reuse the shared Hugging Face downloads you already have instead of fetching them again. The reload still has to happen in memory because only one large local model stays active at a time.

`./run.sh` starts the same app every time: Gemma 4 26B A4B is the default text model, and image turns automatically route to Qwen 3.5 9B, the largest cached MLX vision preset used by this repo. On a fresh machine, the browser shows whether the active model is being downloaded into the cache or loaded from the cache.

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

Override the cache location if you want:

```bash
./run.sh --model-cache-dir /Users/arunabhmishra/.cache --preset qwen3-14b-mlx
```

Override the automatic vision route if needed:

```bash
./run.sh --vision-preset qwen35-9b-mlx
```

## What Changed

- One dropdown now lists supported local presets.
- The app now uses one MLX-first serving path per model instead of duplicating MLX and `llama.cpp` presets.
- Changing the dropdown triggers a backend reload and weight swap.
- Qwen 3.5 presets now support drag-and-drop image chat in the browser UI.
- Downloaded weights now default to the shared `~/.cache` Hugging Face cache root.
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
- `qwen35-35b-a3b-mlx`
- `qwen3-14b-mlx`
- `qwen3-32b-mlx`

## Benchmarking

The benchmark runner reuses the same preset registry and backend code as the app.

Run the default Qwen matrix:

```bash
./.venv/bin/python benchmark_local_models.py
```

Run a smaller slice:

```bash
./.venv/bin/python benchmark_local_models.py --preset qwen35-9b-mlx --suite short --suite long
```

Use a custom cache root for benchmark runs:

```bash
./.venv/bin/python benchmark_local_models.py --model-cache-dir /Users/arunabhmishra/.cache
```

Results are written under `benchmarks/results/` as JSON and Markdown.

## Notes

- `setup.sh` installs `mlx-lm`, `mlx-vlm`, plus `torch` and `torchvision`, because Qwen 3.5 MLX presets use the upstream Qwen3-VL processor stack for image chat.
- Qwen 3.5 presets support one active dragged or selected image at a time. The attachment is cleared when you clear the chat or switch presets.
- The first load for any preset can take a while because weights download on demand.
- By default, downloads now reuse the shared cache under `~/.cache/huggingface`.
- After the first download, preset switches reuse the on-disk cache and only pay the weight unload/reload cost in memory.
- The UI links to a family-specific benchmark post for the active preset.
- If you already had a `.venv` under the old `gemma-local-chat` path, rerun `./setup.sh` once after pulling the rename so path-bound virtualenv scripts are refreshed.
- As of April 4, 2026, the official local/open Qwen benchmark scope in this repo covers `Qwen 3.5` and `Qwen 3`. No official open local `Qwen 3.6` target was found.

## CLI

```bash
./run.sh --help
```

Useful flags:

- `--preset <preset-id>`
- `--vision-preset <preset-id>`
- `--port 8099`
- `--max-tokens 512`
- `--open-browser`
- `--system-prompt "You are a concise assistant."`
- `--model-cache-dir /Users/arunabhmishra/.cache`
