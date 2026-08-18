#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

on_error() {
  local exit_code=$?
  echo
  echo "INSTALLATION FAILED at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  echo "Review the error above. Nothing is hidden." >&2
  exit "${exit_code}"
}
trap on_error ERR

echo "Moonshine Pi live-STT installation"
echo "Project: ${ROOT_DIR}"
echo "Architecture: $(uname -m)"
echo "OS: $(uname -srm)"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3 was not found." >&2
  exit 2
fi

if command -v apt-get >/dev/null 2>&1; then
  if [[ "${SKIP_APT:-0}" != "1" ]]; then
    SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
      if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo is required for apt packages. Re-run as root or set SKIP_APT=1." >&2
        exit 3
      fi
      SUDO="sudo"
    fi
    echo "Installing Raspberry Pi OS audio/build dependencies..."
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y \
      python3-venv python3-pip \
      portaudio19-dev libasound2-dev libsndfile1 \
      alsa-utils
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --requirement "${ROOT_DIR}/requirements.txt"

echo "Downloading and verifying the explicit English Medium Streaming model..."
python - <<'PY'
from moonshine_voice import ModelArch, get_model_for_language

path, arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)
if arch != ModelArch.MEDIUM_STREAMING:
    raise RuntimeError(f"Expected MEDIUM_STREAMING, got {arch!r}")
print(f"Verified model architecture: {arch}")
print(f"Cached model path: {path}")
PY

echo "Running code/config self-test..."
python "${ROOT_DIR}/src/moonshine_pi_live.py" --self-test

echo
echo "INSTALLATION COMPLETE"
echo "1. Connect a USB microphone."
echo "2. List devices: ./run.sh --list-devices"
echo "3. Start live STT: ./run.sh live"
echo "4. Run WER benchmark: ./benchmark.sh"
