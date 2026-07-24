#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtual environment missing. Run ./install.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
exec python "${ROOT_DIR}/src/moonshine_pi_live.py" "$@"
