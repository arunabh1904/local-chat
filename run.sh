#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
DEFAULT_MODEL_CACHE_DIR="$(cd "${ROOT_DIR}/.." && pwd)/models"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./setup.sh first." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Re-run ./setup.sh to rebuild the virtualenv." >&2
  exit 1
fi

export LOCAL_CHAT_MODEL_CACHE_DIR="${LOCAL_CHAT_MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/local_chat.py" "$@"
