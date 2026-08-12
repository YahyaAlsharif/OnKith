#!/usr/bin/env bash
# What the autostart entry runs when the Pi comes up.
#
# Clicking the icon and booting into it are not the same job, which is
# why this exists next to ./onkith rather than instead of it. A person
# clicking the icon is present, is looking at a terminal, and can see
# what went wrong. A device powering on has none of that.
#
# So three things happen here that do not happen when somebody runs the
# app themselves. It waits, because an autostart entry fires before the
# audio server and the network are necessarily up and the microphone is
# the first thing the app reaches for. It writes to a log, because there
# is no terminal at boot and a device that fails silently cannot be
# diagnosed at a demo. And it keeps the previous log, because the
# interesting one is usually the run that just died rather than the one
# that replaced it.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Overridable so a device can be started on the other recogniser without
# editing the desktop entry, which is the file people forget exists.
ONKITH_BOOT_ARGS="${ONKITH_BOOT_ARGS:---stt whisper --no-colour}"
ONKITH_BOOT_DELAY="${ONKITH_BOOT_DELAY:-12}"

LOG_DIR="${ONKITH_LOG_DIR:-${HOME}/.local/state/onkith}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/onkith.log"
if [[ -f "${LOG_FILE}" ]]; then
  mv -f "${LOG_FILE}" "${LOG_FILE}.1"
fi

{
  # Spelled out rather than date -Is, which is GNU only. The Pi has
  # coreutils and a Mac does not, and this gets tried on both before it
  # is ever trusted on the device.
  echo "=== onkith boot $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
  echo "root:  ${ROOT_DIR}"
  echo "args:  ${ONKITH_BOOT_ARGS}"
  echo "delay: ${ONKITH_BOOT_DELAY}s"
} > "${LOG_FILE}"

sleep "${ONKITH_BOOT_DELAY}"

cd "${ROOT_DIR}"
# shellcheck disable=SC2086
exec "${ROOT_DIR}/onkith" ${ONKITH_BOOT_ARGS} >> "${LOG_FILE}" 2>&1
