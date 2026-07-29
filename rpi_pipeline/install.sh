#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [[ "$(id -u)" -eq 0 ]] || SUDO="sudo"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y python3-venv python3-pip portaudio19-dev libasound2-dev libsndfile1 alsa-utils
fi
[[ -d "${VENV_DIR}" ]] || "${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${ROOT_DIR}/requirements.txt"
required=(
  "moonshine-pi-v2/src/moonshine_pi_live.py"
  "moonshine-pi-v2/config/final_config.json"
  "tinybert_pii_rpi/model_int8.onnx"
  "tinybert_pii_rpi/tokenizer/tokenizer.json"
  "tinybert_pii_rpi/config/id2label.json"
  "pipeline/run_pipeline.py"
  "pipeline/privacy_masker.py"
)
for rel in "${required[@]}"; do
  [[ -f "${ROOT_DIR}/${rel}" ]] || { echo "Missing required file: ${rel}" >&2; exit 2; }
done
python - <<'PY'
from moonshine_voice import ModelArch, get_model_for_language
path, arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)
assert arch == ModelArch.MEDIUM_STREAMING
print(f"Moonshine model cached at: {path}")
PY
python -m py_compile "${ROOT_DIR}/pipeline/privacy_masker.py" "${ROOT_DIR}/pipeline/run_pipeline.py"
echo "INSTALLATION COMPLETE"
echo "Run: ./run.sh --list-devices  then  ./run.sh"
