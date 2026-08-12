#!/usr/bin/env bash
# Everything we need to know from the Pi, in one file to send back.
#
#     ./scripts/pi_report.sh              everything that needs no microphone
#     ./scripts/pi_report.sh --device 2   and the microphone checks too
#
# There are five separate things somebody would otherwise have to run by
# hand and five separate outputs to copy back, and the one that matters
# most, the RTF number, is the one at the end that people stop before
# reaching. So it is one command and one file.
#
# Nothing here changes the device. It reads, it measures, it writes a
# report next to itself. Safe to run twice.
#
# Sections keep going when one fails, because a failing section is a
# result and stopping at the first one hides the rest.

set -Euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    *) echo "Usage: $0 [--device N]" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u '+%Y%m%d-%H%M%S')"
REPORT="${ROOT_DIR}/pi-report-${STAMP}.txt"
PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "No virtual environment at ${PYTHON}. Run ./install_pi.sh first." >&2
  exit 1
fi

RESULTS=()

section() {
  local title="$1"; shift
  {
    echo
    echo "==================================================================="
    echo "== ${title}"
    echo "==================================================================="
  } | tee -a "${REPORT}"
  local status=0
  # The subshell keeps a section's set -e from taking the script with it.
  ( "$@" ) 2>&1 | tee -a "${REPORT}" || status=$?
  # tee is last in the pipe, so ask the pipeline for the real exit code.
  status="${PIPESTATUS[0]}"
  RESULTS+=("${status} ${title}")
  echo "[section exit ${status}] ${title}" | tee -a "${REPORT}" >/dev/null
}

{
  echo "Onkith Pi report"
  echo "generated ${STAMP} UTC"
  echo "root      ${ROOT_DIR}"
  echo "host      $(uname -a)"
} > "${REPORT}"

section "What this board is" "${ROOT_DIR}/run_live.sh" --device-report

section "Hardware PWM, for the lamp" "${ROOT_DIR}/scripts/enable_pwm.sh" --check

section "What is installed" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import shutil
import sys

print(f"python {sys.version.split()[0]}")
for name in ("faster-whisper", "moonshine-voice", "onnxruntime", "numpy",
             "sounddevice", "transformers", "rpi-hardware-pwm", "yt-dlp"):
    try:
        print(f"  {name:20} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:20} NOT INSTALLED")
for tool in ("mpv", "yt-dlp", "piper", "piper-tts", "espeak-ng", "chromium-browser"):
    print(f"  {tool:20} {shutil.which(tool) or '-'}")

# The recogniser we ship has to be on disk before a demo, not fetched
# from HuggingFace in front of people.
try:
    from faster_whisper import WhisperModel

    WhisperModel("base.en", compute_type="int8", local_files_only=True)
    print("  base.en              cached locally")
except Exception as error:
    print(f"  base.en              NOT CACHED ({type(error).__name__})")
PY

section "Readiness check" "${ROOT_DIR}/check_pi.sh"

section "Test suite" "${PYTHON}" -m unittest discover -s tests -q

# The number the whole recogniser decision is waiting on. Needs the
# corpus, which is in the repo, so this runs without a microphone.
section "Whisper on the corpus, and the RTF" \
  "${PYTHON}" eval/run_eval.py --engine whisper --no-record

if [[ -n "${DEVICE}" ]]; then
  section "Microphone ${DEVICE}" "${ROOT_DIR}/check_pi.sh" --with-mic --device "${DEVICE}"
else
  {
    echo
    echo "== Microphone: skipped."
    echo "   Re-run with --device N to include it:"
    echo "     ./run_live.sh --list-devices"
    echo "     ./scripts/pi_report.sh --device N"
  } | tee -a "${REPORT}"
fi

{
  echo
  echo "==================================================================="
  echo "== Summary"
  echo "==================================================================="
  for line in "${RESULTS[@]}"; do
    code="${line%% *}"
    name="${line#* }"
    if [[ "${code}" == "0" ]]; then
      printf '  ok    %s\n' "${name}"
    else
      printf '  FAIL  %s (exit %s)\n' "${name}" "${code}"
    fi
  done
  echo
  echo "Send this file back: ${REPORT}"
} | tee -a "${REPORT}"
