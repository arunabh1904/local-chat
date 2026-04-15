#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
DEFAULT_MODEL_CACHE_DIR="${HOME}/.cache"
DEFAULT_PRESET="${LOCAL_CHAT_DEFAULT_PRESET:-gemma4-26b-a4b-mlx}"
DEFAULT_VISION_PRESET="${LOCAL_CHAT_DEFAULT_VISION_PRESET:-qwen35-9b-mlx}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./setup.sh first." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Re-run ./setup.sh to rebuild the virtualenv." >&2
  exit 1
fi

export LOCAL_CHAT_MODEL_CACHE_DIR="${LOCAL_CHAT_MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"

has_preset=0
has_vision_preset=0
passthrough_only=0
for arg in "$@"; do
  case "${arg}" in
    --preset|--preset=*|--runtime|--runtime=*|--model|--model=*)
      has_preset=1
      ;;
    --vision-preset|--vision-preset=*)
      has_vision_preset=1
      ;;
    --help|--list-presets)
      passthrough_only=1
      ;;
  esac
done

default_args=()
if [[ "${passthrough_only}" -eq 0 && "${has_preset}" -eq 0 ]]; then
  default_args+=(--preset "${DEFAULT_PRESET}")
fi
if [[ "${passthrough_only}" -eq 0 && "${has_vision_preset}" -eq 0 ]]; then
  default_args+=(--vision-preset "${DEFAULT_VISION_PRESET}")
fi

if [[ "${#default_args[@]}" -gt 0 ]]; then
  exec "${PYTHON_BIN}" "${ROOT_DIR}/local_chat.py" "${default_args[@]}" "$@"
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/local_chat.py" "$@"
