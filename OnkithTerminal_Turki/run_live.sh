#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment not found. Run ./install_pi.sh first." >&2
  exit 1
fi

# A fixed port by default. The surface used to take whatever port the kernel
# handed out, which is fine for a browser on this machine and useless for
# anybody reaching it over ssh: the tunnel has to be opened before the app
# has started, so the number has to be knowable in advance rather than
# printed once the session is already running.
PORT="${ONKITH_PORT:-8765}"
CHOSEN=0
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--port" ]]; then
    PORT="${argument}"
    CHOSEN=1
  elif [[ "${argument}" == --port=* ]]; then
    PORT="${argument#--port=}"
    CHOSEN=1
  fi
  previous="${argument}"
done

arguments=("$@")
if [[ "${CHOSEN}" -eq 0 ]]; then
  arguments+=(--port "${PORT}")
fi

# Printed before the models load, because they take the better part of a
# minute and the URL is what somebody needs in order to spend that minute
# opening a browser.
printf '\n  Player UI   http://127.0.0.1:%s/player.html\n' "${PORT}"
printf '  From another machine, open the tunnel first, then the URL above:\n'
printf '    ssh -L %s:127.0.0.1:%s %s@%s\n\n' \
  "${PORT}" "${PORT}" "${USER:-pi}" "$(hostname)"

cd "${ROOT_DIR}"
exec "${VENV_PYTHON}" "${ROOT_DIR}/app/omit_ai_live.py" "${arguments[@]}"
