#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

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

echo
echo "Setup complete."
echo "Next run:"
echo "  ./run.sh --preset gemma4-26b-a4b-mlx --open-browser"
echo "Direct entrypoint:"
echo "  ./.venv/bin/python local_chat.py --preset gemma4-26b-a4b-mlx --open-browser"
echo
echo "If you renamed or moved this checkout, rerun ./setup.sh once to refresh any path-bound virtualenv scripts."
