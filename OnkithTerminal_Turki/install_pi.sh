#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
WITH_PWM=1
PWM_REBOOT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    # For a board with no lamp wired, or one whose boot config somebody
    # else owns. Nothing but the lamp depends on it.
    --no-pwm) WITH_PWM=0; shift ;;
    *)
      echo "Usage: ./install_pi.sh [--check-only] [--no-pwm]" >&2
      exit 2
      ;;
  esac
done

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
  alsa-utils \
  mpv \
  chromium-browser \
  pulseaudio-utils

python3 -m venv "${ROOT_DIR}/.venv"
source "${ROOT_DIR}/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --requirement "${ROOT_DIR}/requirements-rpi.txt"

# The recogniser this device ships on. It used to be left out of the Pi
# install because it had not been chosen yet, which meant a Pi could pass
# every check in check_pi.sh and still have no Whisper on it, and the
# first anybody heard of that was somebody typing --stt whisper in front
# of an audience. CTranslate2, not torch.
python -m pip install --requirement "${ROOT_DIR}/requirements-whisper.txt"

# Pulled now rather than on first use. faster-whisper fetches base.en
# from HuggingFace the first time it runs, and a demo venue is the wrong
# place to discover that.
python - <<'PY'
from faster_whisper import WhisperModel

WhisperModel("base.en", compute_type="int8")
print("[OK] Whisper base.en is cached on this device")
PY

# yt-dlp is deliberately unpinned and installed into the environment: it
# has to track YouTube extraction changes, and a pinned copy stops
# resolving streams within weeks. Re-run this line to refresh it.
python -m pip install --upgrade yt-dlp

# Kokoro is the voice the device ships with. The ONNX build only: plain
# `kokoro` pulls torch, which this deployment does not allow.
#
# The two model files are downloaded rather than vendored: together they
# are 337 MB, which does not belong in a git repository, and they live
# outside the tree so a fresh clone does not fetch them twice.
VOICES_DIR="${HOME}/onkith-voices"
KOKORO_RELEASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
mkdir -p "${VOICES_DIR}"
if python -m pip install --upgrade "kokoro-onnx==0.6.0"; then
  for file in kokoro-v1.0.onnx voices-v1.0.bin; do
    if [ -s "${VOICES_DIR}/${file}" ]; then
      echo "[OK] ${file} already downloaded."
    else
      echo "[..] Downloading ${file} (this is the slow part)."
      # -f so an HTML error page is not written out as a model file, and
      # a partial download is removed rather than left to fail later
      # inside onnxruntime with an opaque protobuf error.
      curl -fL --retry 3 -o "${VOICES_DIR}/${file}" \
        "${KOKORO_RELEASE}/${file}" || rm -f "${VOICES_DIR}/${file}"
    fi
  done
fi
if [ -s "${VOICES_DIR}/kokoro-v1.0.onnx" ] \
   && [ -s "${VOICES_DIR}/voices-v1.0.bin" ]; then
  echo "[OK] Kokoro installed. Put these two lines in .env:"
  echo "         ONKITH_KOKORO_MODEL=${VOICES_DIR}/kokoro-v1.0.onnx"
  echo "         ONKITH_KOKORO_VOICES=${VOICES_DIR}/voices-v1.0.bin"
else
  echo "[WARNING] Kokoro is not usable; falling back to piper or espeak."
fi

# Piper stays as the lighter fallback. Kokoro holds a 310 MB graph and
# takes a second or two per sentence on a fast laptop; if this board
# cannot afford either, ONKITH_TTS=piper in .env switches without any
# other change. Installed best-effort because its wheels are not always
# published for this architecture.
if python -m pip install --upgrade piper-tts; then
  echo "[OK] piper installed as the fallback voice."
else
  echo "[WARNING] piper-tts did not install on this architecture."
  ${SUDO} apt-get install -y espeak-ng || true
fi

# The one case that must not pass quietly. `|| true` above is right for
# piper, whose wheels are genuinely not always published for aarch64, and
# it was on the espeak fallback too, so a Pi that ended up with no voice
# at all finished the install reporting success. The first device this
# ran on did exactly that: piper, piper-tts and espeak-ng all absent, the
# report otherwise green, and a device that hears and answers and never
# says a word.
if [ -s "${VOICES_DIR}/kokoro-v1.0.onnx" ] \
   || command -v piper >/dev/null || command -v piper-tts >/dev/null \
   || command -v espeak-ng >/dev/null; then
  echo "[OK] A speech backend is installed."
else
  echo "[WARNING] ==============================================="
  echo "[WARNING] No speech backend. This device cannot speak."
  echo "[WARNING] It will hear, mask, answer and show the answer,"
  echo "[WARNING] silently. Fix with:"
  echo "[WARNING]     sudo apt-get install -y espeak-ng"
  echo "[WARNING] and read the error if that fails."
  echo "[WARNING] ==============================================="
fi

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

# The lamp needs a boot overlay. This used to print instructions and
# leave it to somebody to hand-edit config.txt, which was the wrong call:
# by hand there is no backup, nothing checks for an overlay that is
# already there, and nothing notices when the end of the file is inside a
# [pi4] section and the line would apply to one model only.
#
# enable_pwm.sh does it with all three of those handled, and refuses
# rather than guesses when a different pwm overlay is already set. It
# does not reboot, which is the part that stays a decision.
if [[ -e /sys/class/pwm/pwmchip0 ]]; then
  echo "[OK] Hardware PWM is available at /sys/class/pwm/pwmchip0"
elif [[ "${WITH_PWM}" -eq 0 ]]; then
  echo "[SKIP] --no-pwm given; the lamp will do nothing until the overlay"
  echo "       is added. Everything else on the device works without it."
elif "${ROOT_DIR}/scripts/enable_pwm.sh"; then
  PWM_REBOOT=1
else
  echo "[WARNING] The PWM overlay was not set; see above."
  echo "[WARNING] Everything except the lamp works without it."
fi

"${ROOT_DIR}/check_pi.sh"

echo
echo "Installation complete. Next commands:"
echo "  cd \"${ROOT_DIR}\""
echo "  ./run_live.sh --list-devices"
echo "  ./check_pi.sh --with-mic --device 2"
echo "  ./run_live.sh --stt whisper --device 2"
echo
echo "For a clickable icon on the desktop, and to start on boot:"
echo "  ./install_desktop.sh"
if [[ "${PWM_REBOOT}" -eq 1 ]]; then
  echo
  echo "The PWM overlay was added to the boot config. The lamp will not"
  echo "work until this Pi is rebooted:"
  echo "  sudo reboot"
  echo "Then confirm the chip number, and set light.chip in"
  echo "config/live_config.json if it is not pwmchip0:"
  echo "  ls /sys/class/pwm/"
fi
