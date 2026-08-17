#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: ./install_pi.sh [--check-only]" >&2
  exit 2
fi

ARCH="$(uname -m)"
echo "[SYSTEM] Architecture: ${ARCH}"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
  echo "[WARNING] Target is 64-bit ARM Raspberry Pi OS; static checks may continue here."
fi

command -v python3 >/dev/null || {
  echo "python3 is required." >&2
  exit 1
}
PYTHON_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "[SYSTEM] Python: ${PYTHON_VERSION}"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' || {
  echo "Python 3.11 is required on the target Raspberry Pi." >&2
  exit 1
}

MODEL="${ROOT_DIR}/models/tinybert_pii/model_int8.onnx"
[[ -s "${MODEL}" ]] || {
  echo "Missing TinyBERT INT8 model: ${MODEL}" >&2
  exit 1
}
if head -c 64 "${MODEL}" | grep -q "git-lfs.github.com"; then
  echo "TinyBERT artifact is a Git LFS pointer, not the model." >&2
  exit 1
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "[OK] Static installer checks passed; no packages were installed."
  exit 0
fi

cd "${ROOT_DIR}"

FREE_KB="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {print $4}')"
RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
echo "[SYSTEM] Free disk: $((FREE_KB / 1024)) MiB"
echo "[SYSTEM] Installed RAM: $((RAM_KB / 1024)) MiB"
if (( FREE_KB < 2097152 )); then
  echo "At least 2 GiB of free disk is required." >&2
  exit 1
fi
if (( RAM_KB < 3000000 )); then
  echo "[WARNING] Less than 3 GB RAM detected; target hardware has 4 GB."
fi

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  command -v sudo >/dev/null || {
    echo "sudo is required to install audio system packages." >&2
    exit 1
  }
  SUDO="sudo"
fi

${SUDO} apt-get update
${SUDO} apt-get install -y \
  python3-venv \
  python3-pip \
  portaudio19-dev \
  libasound2-dev \
  libsndfile1 \
  alsa-utils

python3 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --requirement "${ROOT_DIR}/requirements-rpi.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

root = Path.cwd()
model = root / "models" / "tinybert_pii" / "model_int8.onnx"
header = model.read_bytes()[:128]
if header.startswith(b"version https://git-lfs.github.com/spec/"):
    raise SystemExit("TinyBERT is a Git LFS pointer")
if model.stat().st_size < 1_000_000:
    raise SystemExit("TinyBERT model is unexpectedly small")
AutoTokenizer.from_pretrained(
    root / "models" / "tinybert_pii" / "tokenizer",
    local_files_only=True,
    use_fast=True,
)
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("CPUExecutionProvider is unavailable")
ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
print("[OK] TinyBERT tokenizer and INT8 model work fully offline.")
PY

python - <<'PY'
from moonshine_voice import ModelArch, get_model_for_language

model_path, model_arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)
if model_arch != ModelArch.MEDIUM_STREAMING:
    raise SystemExit(f"Expected MEDIUM_STREAMING, received {model_arch!r}")
print(f"[OK] Moonshine English Medium Streaming cached at {model_path}")
PY

"${ROOT_DIR}/check_pi.sh"

echo
echo "Installation complete. Next commands:"
echo "  cd \"${ROOT_DIR}\""
echo "  ./run_live.sh --list-devices"
echo "  ./check_pi.sh --with-mic --device 2"
echo "  ./run_live.sh --device 2"
