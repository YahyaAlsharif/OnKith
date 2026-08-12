#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=""
UV_BIN="$(command -v uv || true)"

for candidate in python3.11 python3; do
  if command -v "${candidate}" >/dev/null 2>&1 && \
    "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
    PYTHON_BIN="$(command -v "${candidate}")"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" && -z "${UV_BIN}" ]]; then
  echo "Python 3.11 is required. Install it, then run ./install.sh again." >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]] && ! brew --prefix portaudio >/dev/null 2>&1; then
  echo "PortAudio is required on macOS. Run: brew install portaudio" >&2
  exit 1
fi

if ! command -v mpv >/dev/null 2>&1; then
  echo "[WARNING] mpv was not found. Music cannot play without it."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[WARNING] Install it with: brew install mpv"
  else
    echo "[WARNING] Install it with your package manager, for example: apt install mpv"
  fi
fi

if [[ -n "${UV_BIN}" ]]; then
  if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    "${UV_BIN}" venv --python 3.11 "${ROOT_DIR}/.venv"
  fi
  "${UV_BIN}" pip install \
    --python "${ROOT_DIR}/.venv/bin/python" \
    --requirement "${ROOT_DIR}/requirements-rpi.txt"
else
  if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${ROOT_DIR}/.venv"
  fi
  "${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "${ROOT_DIR}/.venv/bin/python" -m pip install \
    --requirement "${ROOT_DIR}/requirements-rpi.txt"
fi

# yt-dlp is deliberately unpinned: it has to track YouTube extraction
# changes, and a pinned copy stops resolving streams within weeks.
"${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade yt-dlp

ln -sfn "${ROOT_DIR}/onkith" "${ROOT_DIR}/.venv/bin/onkith"
"${ROOT_DIR}/.venv/bin/onkith" --self-test

echo
echo "Onkith terminal is ready."
echo "Run: ${ROOT_DIR}/onkith --list-devices"
echo "Then: ${ROOT_DIR}/onkith --device DEVICE_ID"
