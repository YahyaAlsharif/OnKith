#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment not found. Run ./install_pi.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec "${VENV_PYTHON}" "${ROOT_DIR}/scripts/check_pi.py" "$@"
