#!/usr/bin/env bash
# Put the hardware PWM overlay into the Pi's boot config, for the lamp.
#
#     ./scripts/enable_pwm.sh            add it, for GPIO12
#     ./scripts/enable_pwm.sh --pin 18   a different pin
#     ./scripts/enable_pwm.sh --check    say what the state is
#     ./scripts/enable_pwm.sh --remove   take it back out
#
# This was a printed instruction for a while and that was the wrong call.
# Telling somebody to hand-edit their boot config is more dangerous than
# doing it in a script, not less: by hand there is no backup, no check
# for a line that is already there, and nothing to notice that the end of
# config.txt is inside a [pi4] section, which would scope the overlay to
# one model and fail silently on any other.
#
# So it is done here, and the three things that make it safe are done
# properly:
#
#   1. The file is backed up first, with the path printed.
#   2. It is idempotent. Run it twice and the second run changes nothing.
#   3. The line goes under an explicit [all] header, so whatever
#      conditional section the file happened to end in cannot scope it.
#
# It still needs a reboot afterwards, which is the one part no script
# should decide for you.

set -Eeuo pipefail

PIN=12
ACTION="enable"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pin) PIN="${2:-}"; shift 2 ;;
    --check) ACTION="check"; shift ;;
    --remove|--disable) ACTION="remove"; shift ;;
    *)
      echo "Usage: $0 [--pin N] [--check|--remove]" >&2
      exit 2
      ;;
  esac
done

# Which alt function selects PWM depends on the pin, and getting it wrong
# gives a pin that is configured for something else with no error. These
# four are the Pi's hardware PWM pins. Anything else has to be looked up
# with `pinctrl funcs N` on the board itself.
case "${PIN}" in
  12) FUNC=4; CHANNEL=0 ;;
  13) FUNC=4; CHANNEL=1 ;;
  18) FUNC=2; CHANNEL=0 ;;
  19) FUNC=2; CHANNEL=1 ;;
  *)
    echo "GPIO${PIN} is not one of this board's hardware PWM pins." >&2
    echo "Those are 12, 13, 18 and 19. Check yours with: pinctrl funcs ${PIN}" >&2
    exit 2
    ;;
esac

OVERLAY="dtoverlay=pwm,pin=${PIN},func=${FUNC}"

# Bookworm moved it. The override exists so this script can be exercised
# against a copy rather than only on a live Pi.
if [[ -n "${ONKITH_BOOT_CONFIG:-}" ]]; then
  CONFIG="${ONKITH_BOOT_CONFIG}"
elif [[ -f /boot/firmware/config.txt ]]; then
  CONFIG="/boot/firmware/config.txt"
elif [[ -f /boot/config.txt ]]; then
  CONFIG="/boot/config.txt"
else
  echo "[SKIP] No Raspberry Pi boot config found."
  echo "       Looked in /boot/firmware/config.txt and /boot/config.txt."
  echo "       Nothing to do on a machine that is not a Pi."
  exit 0
fi

SUDO=""
if [[ ! -w "${CONFIG}" ]]; then
  if command -v sudo >/dev/null; then
    SUDO="sudo"
  else
    echo "Cannot write ${CONFIG} and sudo is not available." >&2
    exit 1
  fi
fi

# pwm-2chan and pwm-gpio are the same hardware under other names, and the
# first version of this only matched a bare "pwm". It happily appended a
# second overlay next to a pwm-2chan that was already there, which is two
# overlays fighting over one channel: exactly the case this check exists
# to catch.
PWM_OVERLAY_RE="^[[:space:]]*dtoverlay=pwm(-[a-z0-9]+)*([,[:space:]]|\$)"

present() {
  grep -qE "${PWM_OVERLAY_RE}" "${CONFIG}"
}

exact() {
  grep -qxF "${OVERLAY}" "${CONFIG}"
}

if [[ "${ACTION}" == "check" ]]; then
  echo "config:  ${CONFIG}"
  if [[ -e "/sys/class/pwm/pwmchip0" ]]; then
    echo "[OK] /sys/class/pwm/pwmchip0 exists; the lamp can be driven."
  else
    echo "[--] No pwmchip yet."
  fi
  if exact; then
    echo "[OK] ${OVERLAY} is in the boot config."
  elif present; then
    echo "[!!] A different pwm overlay is already set:"
    grep -nE "${PWM_OVERLAY_RE}" "${CONFIG}" | sed 's/^/     /'
  else
    echo "[--] No pwm overlay in the boot config."
  fi
  exit 0
fi

backup() {
  local stamp
  stamp="$(date -u '+%Y%m%d-%H%M%S')"
  local target="${CONFIG}.onkith-${stamp}"
  ${SUDO} cp -p "${CONFIG}" "${target}"
  echo "[OK] Backed up to ${target}"
}

if [[ "${ACTION}" == "remove" ]]; then
  if ! exact; then
    echo "[--] ${OVERLAY} is not in ${CONFIG}; nothing removed."
    exit 0
  fi
  backup
  # Only our exact line goes. A pwm overlay somebody else put there for
  # different hardware is not ours to delete.
  ${SUDO} sed -i.onkith-tmp "\\|^${OVERLAY}\$|d" "${CONFIG}"
  ${SUDO} rm -f "${CONFIG}.onkith-tmp"
  echo "[OK] Removed ${OVERLAY} from ${CONFIG}"
  echo "     Reboot for it to take effect."
  exit 0
fi

if exact; then
  echo "[OK] ${OVERLAY} is already in ${CONFIG}; nothing to do."
  if [[ ! -e "/sys/class/pwm/pwmchip0" ]]; then
    echo "     No pwmchip yet, so this Pi has not been rebooted since."
  fi
  exit 0
fi

if present; then
  echo "[!!] A pwm overlay is already set, and it is not the one for GPIO${PIN}:"
  grep -nE "${PWM_OVERLAY_RE}" "${CONFIG}" | sed 's/^/     /'
  echo "     Left alone. Two pwm overlays on one channel conflict, and"
  echo "     which one you want depends on how the lamp is wired."
  echo "     Edit it by hand, or run: $0 --pin <the pin you used>"
  exit 1
fi

backup
# The [all] header is the point. Conditional filters in config.txt apply
# until the next filter, so appending to a file that happens to end in a
# [pi4] or [cm4] section would quietly scope the overlay to that model.
${SUDO} tee -a "${CONFIG}" >/dev/null <<ENTRY

# Onkith: hardware PWM for the lamp on GPIO${PIN} (PWM0_CHAN${CHANNEL}).
[all]
${OVERLAY}
ENTRY

echo "[OK] Added to ${CONFIG}:"
echo "       ${OVERLAY}"
echo
echo "     Reboot, then confirm the chip number:"
echo "       sudo reboot"
echo "       ls /sys/class/pwm/"
if [[ "${CHANNEL}" -ne 0 ]]; then
  echo
  echo "     GPIO${PIN} is PWM0_CHAN${CHANNEL}, so set light.channel to"
  echo "     ${CHANNEL} in config/live_config.json."
fi
