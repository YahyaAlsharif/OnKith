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

# Checked here rather than four minutes later. The N in the instructions
# is a placeholder for a number, and it was typed in literally, so the
# whole report ran and the microphone section failed at the end on an
# argument error.
if [[ -n "${DEVICE}" && ! "${DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "--device wants the number of a microphone, not '${DEVICE}'." >&2
  echo "Find it with:  ./run_live.sh --list-devices" >&2
  exit 2
fi

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
  # The subshell keeps a section's own set -e from taking the script with
  # it. tee is last in the pipe, so the section's exit code has to come
  # from PIPESTATUS rather than $?.
  #
  # PIPESTATUS has to be read on the very next line. The first version of
  # this had `|| status=$?` in between, and that assignment is itself a
  # command, so it reset PIPESTATUS to (0) before it was read. Every
  # section reported success. The first report back from the Pi said
  # "ok Microphone N" for a section that had failed with an argument
  # error, which is the worst way for a reporting script to be wrong.
  ( "$@" ) 2>&1 | tee -a "${REPORT}"
  local status="${PIPESTATUS[0]}"
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
for tool in ("mpv", "yt-dlp", "piper", "piper-tts", "espeak-ng",
             "chromium", "chromium-browser"):
    print(f"  {tool:20} {shutil.which(tool) or '-'}")

# Listing the three as absent was not enough: the first report back had
# all of them missing and every section still said ok, so nobody read it
# as "this device cannot speak". Say it in words.
if not any(shutil.which(t) for t in ("piper", "piper-tts", "espeak-ng")):
    print()
    print("  WARNING: no speech backend. This device cannot speak.")
    print("           sudo apt-get install -y espeak-ng")

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
