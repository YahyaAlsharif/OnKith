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

# Keep large binaries out of Git history. Download the exact release assets and
# verify their pinned size and SHA-256 before the existing integrity checks.
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
# Native Parakeet TDT requires PyTorch, NeMo and the 459 MB pinned .nemo
# checkpoint. Keep enough headroom for the virtualenv and model cache.
if (( FREE_KB < 6291456 )); then
  echo "At least 6 GiB of free disk is required for NeMo, PyTorch, and Parakeet." >&2
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
# Keep the toolchain for any NeMo ASR dependency that has no ARM64 wheel.
${SUDO} apt-get install -y \
  python3.11-venv python3.11-dev python3-pip build-essential cmake pkg-config \
  portaudio19-dev libasound2-dev \
  libsndfile1 libopenblas0-pthread ffmpeg alsa-utils espeak-ng

cd "${ROOT_DIR}"
python3.11 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# The large native packages must be ARM64 wheels; never compile PyTorch or
# ONNX Runtime on a 4 GB Pi.
python -m pip install --only-binary=:all: \
  "numpy==2.0.2" "scipy==1.15.3" "sounddevice==0.5.5" \
  "onnx==1.17.0" "onnxruntime==1.22.1" "torch==2.7.1" \
  "transformers==4.53.3" "huggingface_hub==0.34.4" "psutil==7.0.0" || {
  echo "A pinned ARM64 runtime wheel is unavailable for this OS/Python combination." >&2
  exit 1
}

# NeMo is pure Python, but a small ASR dependency may need the compiler above.
python -m pip install "nemo_toolkit[asr]==2.7.3" || {
  echo "nemo_toolkit[asr]==2.7.3 could not be installed on this device." >&2
  exit 1
}
# Optional hardware is intentionally non-fatal. Without the library or PWM
# overlay, light commands use the local no-op backend and all core stages run.
python -m pip install "rpi-hardware-pwm==0.3.1" || \
  echo "[WARN] rpi-hardware-pwm was not installed; lamp control will be disabled."
python -m pip check || echo "[WARN] pip check reported version drift; see above."

# Fetch and byte-verify the exact measured .nemo checkpoint once. Runtime uses
# this cache offline and refuses a different revision or SHA-256.
python - <<'PY'
from integrated_pipeline.parakeet_adapter import (
    ParakeetAdapter,
    cache_parakeet_checkpoint,
)
from integrated_pipeline.config import PipelineConfig

artifact = cache_parakeet_checkpoint(local_files_only=False)
print(f"[OK] cached {artifact['model_id']} at revision {artifact['revision']}")
adapter = ParakeetAdapter(PipelineConfig())
adapter.load()
decoder = adapter.audit()["active_decoder"]
if decoder["decoder_family"] != "TDT" or decoder["strategy"] != "greedy_batch":
    raise SystemExit(f"Wrong Parakeet decoder active: {decoder}")
adapter.close()
print("[OK] Parakeet 110M TDT loads once on CPU with greedy_batch")
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
