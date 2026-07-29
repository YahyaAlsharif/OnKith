#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "Run ./install.sh first." >&2; exit 2; }
source "${VENV_DIR}/bin/activate"
exec python "${ROOT_DIR}/pipeline/run_pipeline.py" "$@"
