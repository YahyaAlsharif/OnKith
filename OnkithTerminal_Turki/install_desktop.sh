#!/usr/bin/env bash
# Make Onkith something you click once, and something that comes back on its
# own after a power cut. Both matter for an appliance: a demo should not
# begin with somebody finding a terminal, and a device unplugged overnight
# should be listening again by the time anybody looks at it.
#
#     ./install_desktop.sh                 icon, menu entry and autostart
#     ./install_desktop.sh --no-autostart  the clickable icon only
#     ./install_desktop.sh --status        say what is installed
#     ./install_desktop.sh --remove        take all of it back out
#
# This installs launchers only. It does not install Onkith itself; run
# ./install_pi.sh first. Nothing here needs root, which is deliberate:
# nothing that can be installed without a password should need one to undo.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WITH_AUTOSTART=1
ACTION="install"
case "${1:-}" in
  "") ;;
  --no-autostart) WITH_AUTOSTART=0 ;;
  --status) ACTION="status" ;;
  --remove|--uninstall) ACTION="remove" ;;
  *)
    echo "Usage: ./install_desktop.sh [--no-autostart|--status|--remove]" >&2
    exit 2
    ;;
esac

APPLICATIONS="${HOME}/.local/share/applications"
AUTOSTART="${HOME}/.config/autostart"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${HOME}/Desktop")"
ENTRIES=(
  "${APPLICATIONS}/onkith.desktop"
  "${DESKTOP}/onkith.desktop"
  "${AUTOSTART}/onkith.desktop"
)

if [[ "${ACTION}" == "status" ]]; then
  for entry in "${ENTRIES[@]}"; do
    if [[ -f "${entry}" ]]; then
      echo "[OK] ${entry}"
    else
      echo "[--] ${entry}"
    fi
  done
  exit 0
fi

if [[ "${ACTION}" == "remove" ]]; then
  for entry in "${ENTRIES[@]}"; do
    if [[ -f "${entry}" ]]; then
      rm -f "${entry}"
      echo "[OK] Removed ${entry}"
    fi
  done
  exit 0
fi

[[ -x "${ROOT_DIR}/onkith" ]] || {
  echo "onkith is not where this script is. Run it from the project directory." >&2
  exit 1
}
[[ -x "${ROOT_DIR}/.venv/bin/python" ]] || {
  echo "No virtual environment found. Run ./install_pi.sh first." >&2
  exit 1
}

BOOT="${ROOT_DIR}/scripts/onkith_boot.sh"
chmod +x "${BOOT}" 2>/dev/null || true

# A PNG rather than the SVG the mark is drawn in. An icon theme reads SVG
# only if librsvg is behind its loader, and when it is not there is no
# error anywhere, just a launcher with a blank square on it.
ICON="${ROOT_DIR}/assets/onkith.png"
if [[ ! -f "${ICON}" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/build_icon.py" || true
fi
[[ -f "${ICON}" ]] || ICON="${ROOT_DIR}/assets/player/onkith-favicon.svg"

# Exec is not a plain string. The desktop spec parses it into arguments,
# so a program path containing a space becomes two of them and the entry
# silently does nothing at all: no error, no log, just an icon that does
# not work. The Pi's own path has no space in it today and this is not
# the file anybody would think to check on the morning that changes.
quoted_program() {
  printf '"%s"' "$(printf '%s' "$1" | sed 's/[\\"`$]/\\&/g')"
}

write_entry() {
  local target="$1"
  local name="$2"
  local comment="$3"
  local program="$4"
  local terminal="$5"
  local arguments="${6:-}"
  local command
  command="$(quoted_program "${program}")${arguments:+ ${arguments}}"
  cat > "${target}" <<ENTRY
[Desktop Entry]
Type=Application
Version=1.0
Name=${name}
Comment=${comment}
Exec=${command}
Path=${ROOT_DIR}
Icon=${ICON}
Terminal=${terminal}
Categories=AudioVideo;Audio;Utility;
StartupNotify=false
ENTRY
  chmod +x "${target}"
}

mkdir -p "${APPLICATIONS}"
# Clicked by somebody who is standing there, so it keeps its terminal and
# they can see what it is doing.
write_entry "${APPLICATIONS}/onkith.desktop" "Onkith" \
  "On-device privacy pipeline" "${ROOT_DIR}/onkith" "true" "--stt whisper"
echo "[OK] Menu entry: ${APPLICATIONS}/onkith.desktop"

if [[ -d "${DESKTOP}" ]]; then
  write_entry "${DESKTOP}/onkith.desktop" "Onkith" \
    "On-device privacy pipeline" "${ROOT_DIR}/onkith" "true" "--stt whisper"
  # A launcher dropped on the desktop is refused until it is marked trusted,
  # and the refusal looks exactly like a file that does nothing when clicked.
  if command -v gio >/dev/null; then
    gio set "${DESKTOP}/onkith.desktop" metadata::trusted true 2>/dev/null || true
  fi
  echo "[OK] Desktop icon: ${DESKTOP}/onkith.desktop"
else
  echo "[SKIP] No desktop directory; menu entry only."
fi

if [[ "${WITH_AUTOSTART}" -eq 1 ]]; then
  mkdir -p "${AUTOSTART}"
  # Booting is not clicking. This one goes through the boot wrapper, which
  # waits for the audio server and writes to a log, because at boot there
  # is nobody watching and no terminal to watch in.
  write_entry "${AUTOSTART}/onkith.desktop" "Onkith" \
    "Start Onkith at login" "${BOOT}" "false"
  echo "[OK] Starts at login: ${AUTOSTART}/onkith.desktop"
  echo "     log: ${HOME}/.local/state/onkith/onkith.log (previous run kept as .log.1)"
  echo "     Run ./install_desktop.sh --remove to stop it starting on its own."
fi

# The launcher runs the same command a person would, so it inherits none of
# the terminal's exported variables. Onkith reads its own .env for exactly
# this reason, and saying so here is cheaper than debugging a device that
# starts up reporting it has no key.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  echo "[OK] .env is present and is read at startup."
else
  echo "[WARNING] No .env beside the code. Started from a launcher rather"
  echo "[WARNING] than a terminal, Onkith will have no keys at all."
fi

echo
echo "Click the Onkith icon, or find it in the applications menu."
echo "To try the boot path now without rebooting: ${BOOT}"
