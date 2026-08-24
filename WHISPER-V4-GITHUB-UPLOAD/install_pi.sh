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
BITS="$(getconf LONG_BIT)"
echo "[SYSTEM] Architecture: ${ARCH}, userspace: ${BITS}-bit"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
  echo "Unsupported target: Raspberry Pi OS aarch64/arm64 is required." >&2
  exit 1
fi
if [[ "${BITS}" != "64" ]]; then
  echo "Unsupported target: a 64-bit userspace is required." >&2
  exit 1
fi

command -v python3.11 >/dev/null || {
  echo "Python 3.11 is required (python3.11 was not found)." >&2
  exit 1
}
python3.11 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'

# Large local models are GitHub Release assets, not Git history. Download the
# exact files and verify their pinned size and SHA-256 before setup continues.
python3.11 "${ROOT_DIR}/scripts/download_large_models.py"

PII_MODEL="${ROOT_DIR}/models/deberta_pii/model_int8.onnx"
SILERO="${ROOT_DIR}/models/silero/silero_vad.onnx"
[[ -s "${PII_MODEL}" ]] || { echo "Missing DeBERTa PII model: ${PII_MODEL}" >&2; exit 1; }
[[ -s "${SILERO}" ]] || { echo "Missing Silero model: ${SILERO}" >&2; exit 1; }
if head -c 64 "${PII_MODEL}" | grep -q "git-lfs.github.com"; then
  echo "DeBERTa PII artifact is a Git LFS pointer, not the model." >&2
  exit 1
fi
echo "bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8  ${PII_MODEL}" | sha256sum --check --status || {
  echo "DeBERTa PII artifact hash mismatch; this is not the validated INT8 model." >&2
  exit 1
}
echo "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004  ${SILERO}" | sha256sum --check --status || {
  echo "Silero artifact hash mismatch." >&2
  exit 1
}

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "[OK] Architecture, Python, DeBERTa PII, and Silero static checks passed."
  exit 0
fi

FREE_KB="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {print $4}')"
RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
echo "[SYSTEM] Free disk: $((FREE_KB / 1024)) MiB"
echo "[SYSTEM] Installed RAM: $((RAM_KB / 1024)) MiB"
# V3 needed 6 GiB here for NeMo, PyTorch and the 700 MB .nemo artifact. V4's
# recognizer is faster-whisper on CTranslate2: no torch, and base.en is about
# 150 MB.
if (( FREE_KB < 2097152 )); then
  echo "At least 2 GiB of free disk is required." >&2
  exit 1
fi
if (( RAM_KB < 3500000 )); then
  echo "This build targets a Raspberry Pi 5 with 4 GB RAM or more." >&2
  exit 1
fi

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  command -v sudo >/dev/null || { echo "sudo is required." >&2; exit 1; }
  SUDO="sudo"
fi
${SUDO} apt-get update
# V3 needed build-essential and cmake here because four packages in
# nemo_toolkit[asr]==2.5.3's tree publish no aarch64 wheel and had to be
# compiled on the device. faster-whisper and CTranslate2 both ship aarch64
# wheels, so nothing is built from source any more; the toolchain is kept only
# because a wheel occasionally has to be rebuilt for an unusual Python patch.
${SUDO} apt-get install -y \
  python3.11-venv python3.11-dev python3-pip build-essential cmake pkg-config \
  portaudio19-dev libasound2-dev \
  libsndfile1 libopenblas0-pthread ffmpeg alsa-utils espeak-ng

cd "${ROOT_DIR}"
python3.11 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Wheels only: nothing in this runtime may be built from source on a 4 GB Pi.
python -m pip install --only-binary=:all: --requirement requirements-rpi.txt || {
  echo "Pinned ARM64 wheels could not be installed on this OS/Python combination." >&2
  exit 1
}

# Optional hardware is intentionally non-fatal. Without the library or PWM
# overlay, light commands use the local no-op backend and all core stages run.
python -m pip install "rpi-hardware-pwm==0.3.1" || \
  echo "[WARN] rpi-hardware-pwm was not installed; lamp control will be disabled."
python -m pip check || echo "[WARN] pip check reported version drift; see above."

# Fetch base.en once, here, so the first spoken utterance is not waiting on a
# download. CTranslate2 caches it under ~/.cache/huggingface; after this the
# device runs offline.
python - <<'PY'
from faster_whisper import WhisperModel

WhisperModel("base.en", device="cpu", compute_type="int8")
print("[OK] faster-whisper base.en INT8 is cached and loads on CPU")
PY

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

root = Path.cwd()
if "CPUExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("CPUExecutionProvider is unavailable")
AutoTokenizer.from_pretrained(
    root / "models" / "deberta_pii" / "tokenizer",
    local_files_only=True,
    use_fast=True,
)
for relative in (
    "models/deberta_pii/model_int8.onnx",
    "models/silero/silero_vad.onnx",
):
    ort.InferenceSession(str(root / relative), providers=["CPUExecutionProvider"])
print("[OK] DeBERTa PII and Silero load fully offline on CPUExecutionProvider")
PY

"${ROOT_DIR}/check_pi.sh"

echo "Installation complete. Copy .env.example to .env for Groq, then use ./run_live.sh --list-devices."
