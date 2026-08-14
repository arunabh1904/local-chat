#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
DEFAULT_MODEL_CACHE_DIR="${HOME}/.cache"

PYTHON_BIN=""

for candidate in python3.12 python3.13 python3.14 python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON_BIN="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find a usable Python 3 interpreter." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.txt"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "Missing llama-server. Install the native runtime with:" >&2
  echo "  brew install llama.cpp" >&2
  exit 1
fi

LLAMA_VERSION="$(llama-server --version 2>&1 | head -n 1)"
LLAMA_BUILD="$(printf '%s' "${LLAMA_VERSION}" | sed -n 's/^version: \([0-9][0-9]*\).*/\1/p')"
if [[ -n "${LLAMA_BUILD}" && "${LLAMA_BUILD}" -lt 10353 ]]; then
  echo "llama.cpp build ${LLAMA_BUILD} is too old for Muse Glimmer." >&2
  echo "Upgrade it with: brew upgrade llama.cpp" >&2
  exit 1
fi

echo
echo "Setup complete."
echo "Next run:"
echo "  ./run.sh --open-browser"
echo "Direct entrypoint:"
echo "  ./.venv/bin/python local_chat.py --preset muse-glimmer-30b-llama --open-browser"
echo "Default model cache:"
echo "  ${DEFAULT_MODEL_CACHE_DIR}"
echo
echo "If you renamed or moved this checkout, rerun ./setup.sh once to refresh any path-bound virtualenv scripts."
