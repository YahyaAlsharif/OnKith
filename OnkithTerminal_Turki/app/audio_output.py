"""Output-side audio: stack detection and core-side level metering.

The device's audio stack is detected at run time rather than assumed, so
the same build works on a Pi running PipeWire, a Pi running PulseAudio,
and a development machine. ``describe_device`` is what confirms the OS
release and audio stack on the device itself.

Level values are computed here, in the core, from the output sink's
monitor. The page no longer holds the audio stream, so it can no longer
measure it; it receives numbers over the websocket instead.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import struct
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MONITOR_SAMPLE_RATE = 16000
MONITOR_BLOCK_FRAMES = 800  # 50 ms
PUBLISH_INTERVAL_BLOCKS = 1


@dataclass
class AudioStack:
    """What the output side of this machine actually offers."""

    server: str = "unknown"
    default_sink: str | None = None
    monitor_source: str | None = None
    capture_command: list[str] = field(default_factory=list)
    tools: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def can_meter(self) -> bool:
        return bool(self.capture_command)


def _run(arguments: list[str], timeout: float = 4.0) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _pulse_default_sink() -> str | None:
    if shutil.which("pactl") is None:
        return None
    name = _run(["pactl", "get-default-sink"])
    if name:
        return name
    # Older pactl builds have no get-default-sink subcommand.
    info = _run(["pactl", "info"])
    if not info:
        return None
    for line in info.splitlines():
        if line.lower().startswith("default sink:"):
            return line.split(":", 1)[1].strip()
    return None


def detect_audio_stack() -> AudioStack:
    """Identify the running sound server and a way to read its output."""

    stack = AudioStack(
        tools={
            name: shutil.which(name)
            for name in (
                "pactl",
                "parec",
                "pw-record",
                "pw-cli",
                "pipewire",
                "pulseaudio",
                "aplay",
                "mpv",
                "yt-dlp",
            )
        }
    )

    if platform.system() == "Darwin":
        stack.server = "coreaudio"
        stack.notes.append(
            "macOS has no loopback monitor without extra software, so output "
            "levels are derived from playback state instead of measured."
        )
        return stack

    pipewire_running = _run(["pgrep", "-x", "pipewire"]) is not None
    pulse_running = _run(["pgrep", "-x", "pulseaudio"]) is not None
    if pipewire_running:
        stack.server = "pipewire"
    elif pulse_running:
        stack.server = "pulseaudio"
    elif stack.tools.get("aplay"):
        stack.server = "alsa"

    sink = _pulse_default_sink()
    if sink:
        stack.default_sink = sink
        stack.monitor_source = f"{sink}.monitor"

    # parec speaks to PulseAudio and to PipeWire's Pulse compatibility
    # layer, so one command covers both of the stacks a Pi is likely to run.
    parec = stack.tools.get("parec")
    if parec and stack.monitor_source:
        stack.capture_command = [
            parec,
            f"--device={stack.monitor_source}",
            "--format=s16le",
            f"--rate={MONITOR_SAMPLE_RATE}",
            "--channels=1",
            "--raw",
        ]
        return stack

    pw_record = stack.tools.get("pw-record")
    if pw_record and stack.monitor_source:
        stack.capture_command = [
            pw_record,
            f"--target={stack.monitor_source}",
            "--format=s16",
            f"--rate={MONITOR_SAMPLE_RATE}",
            "--channels=1",
            "-",
        ]
        return stack

    if stack.server == "alsa":
        stack.notes.append(
            "ALSA without a loopback device exposes no output monitor. "
            "Load snd-aloop, or run PipeWire, to meter output levels."
        )
    elif not stack.monitor_source:
        stack.notes.append(
            "No default sink was reported, so no output monitor could be "
            "resolved."
        )
    return stack


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _board_model() -> str | None:
    for candidate in (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    ):
        try:
            return candidate.read_text(encoding="utf-8").strip("\x00 \n")
        except OSError:
            continue
    return None


def find_chromium() -> str | None:
    for name in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "brave-browser",
        "brave",
    ):
        found = shutil.which(name)
        if found:
            return found
    if platform.system() == "Darwin":
        for candidate in (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ):
            if Path(candidate).is_file():
                return candidate
    return None


def describe_device() -> dict[str, Any]:
    """Report what this machine actually is, for confirmation on the Pi."""

    release = _os_release()
    stack = detect_audio_stack()
    return {
        "board": _board_model(),
        "os": {
            "name": release.get("PRETTY_NAME") or platform.platform(),
            "id": release.get("ID"),
            "version_id": release.get("VERSION_ID"),
            "version_codename": release.get("VERSION_CODENAME"),
            "kernel": platform.release(),
            "architecture": platform.machine(),
        },
        "python": platform.python_version(),
        "audio": asdict(stack),
        "display": {
            "wayland": os.environ.get("WAYLAND_DISPLAY") or None,
            "x11": os.environ.get("DISPLAY") or None,
            "chromium": find_chromium(),
        },
        "playback": {
            "mpv": shutil.which("mpv"),
            "yt-dlp": shutil.which("yt-dlp"),
        },
        "speech": _describe_speech(),
    }


def _describe_speech() -> dict[str, Any]:
    """Which voice the app would actually pick, and where it would play.

    The report used to list the binaries and stop there, which is not the
    same question. A device came back with all three missing and nobody
    read the three dashes as "this cannot speak", and knowing the binary
    exists still does not say whether the sound has anywhere to go: the
    voice plays to the default ALSA device, and on a Pi with no screen
    attached that can be an HDMI sink with nothing on the end of it.
    """

    from app.speech import detect_backend

    outputs: list[str] = []
    try:
        # aplay is in alsa-utils, which install_pi.sh requires.
        listing = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=5
        )
        outputs = [
            line.strip()
            for line in listing.stdout.splitlines()
            if line.startswith("card ")
        ]
    except Exception:
        outputs = []
    return {
        "backend": detect_backend().name,
        "piper": shutil.which("piper") or shutil.which("piper-tts"),
        "piper_voice": os.environ.get("ONKITH_PIPER_MODEL"),
        "espeak": shutil.which("espeak-ng") or shutil.which("espeak"),
        "output_devices": outputs,
    }


def format_device_report(report: dict[str, Any]) -> str:
    """Render the device report for a terminal."""

    lines: list[str] = []
    board = report.get("board")
    operating_system = report.get("os", {})
    lines.append(f"Board          {board or 'unknown'}")
    lines.append(f"OS             {operating_system.get('name')}")
    lines.append(
        "Release        "
        f"{operating_system.get('id')} {operating_system.get('version_id')} "
        f"({operating_system.get('version_codename')})"
    )
    lines.append(
        f"Kernel         {operating_system.get('kernel')} "
        f"{operating_system.get('architecture')}"
    )
    lines.append(f"Python         {report.get('python')}")

    audio = report.get("audio", {})
    lines.append(f"Audio server   {audio.get('server')}")
    lines.append(f"Default sink   {audio.get('default_sink') or 'none'}")
    lines.append(f"Output monitor {audio.get('monitor_source') or 'none'}")
    lines.append(
        "Level metering "
        + ("available" if audio.get("capture_command") else "unavailable")
    )

    playback = report.get("playback", {})
    for name in ("mpv", "yt-dlp"):
        value = playback.get(name)
        lines.append(f"{name:<14} {value or 'MISSING - music cannot play'}")

    display = report.get("display", {})
    session = display.get("wayland") or display.get("x11") or "none"
    lines.append(f"Display        {session}")
    lines.append(f"Chromium       {display.get('chromium') or 'not found'}")

    speech = report.get("speech", {})
    backend = speech.get("backend", "none")
    lines.append(
        f"Voice          {backend}"
        + ("  CANNOT SPEAK" if backend == "none" else "")
    )
    if backend == "none":
        lines.append("Note           sudo apt-get install -y espeak-ng")
    elif backend == "piper" and not speech.get("piper_voice"):
        lines.append("Note           piper has no voice file; set ONKITH_PIPER_MODEL")
    # Where the voice would actually come out. A binary that exists and a
    # sink that plays are different questions, and on a Pi with no screen
    # the default can be an HDMI output with nothing on the end of it.
    for index, card in enumerate(speech.get("output_devices") or []):
        lines.append(f"{'Output' if index == 0 else '':<14} {card}")
    if not speech.get("output_devices") and platform.system() == "Linux":
        # Only where aplay is the right question. A Mac has no aplay and
        # an empty list there means nothing at all.
        lines.append("Output         none found by aplay -l; nothing can be heard")

    for note in audio.get("notes", []):
        lines.append(f"Note           {note}")
    return "\n".join(lines)


class OutputLevelMeter:
    """Measure the output sink's monitor and publish levels to surfaces."""

    def __init__(
        self,
        bus: Any,
        *,
        stack: AudioStack | None = None,
        command: list[str] | None = None,
    ) -> None:
        self.bus = bus
        self.stack = stack if stack is not None else detect_audio_stack()
        self._command = command if command is not None else self.stack.capture_command
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def available(self) -> bool:
        return bool(self._command)

    def start(self) -> bool:
        if not self._command or self._thread is not None:
            return False
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except OSError:
            self._process = None
            return False
        self._thread = threading.Thread(
            target=self._read_loop,
            name="onkith-output-meter",
            daemon=True,
        )
        self._thread.start()
        return True

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        block_bytes = MONITOR_BLOCK_FRAMES * 2
        stream = process.stdout
        while not self._stopping.is_set():
            block = stream.read(block_bytes)
            if not block or len(block) < block_bytes:
                break
            rms, peak = block_levels(block)
            try:
                self.bus.publish(
                    "level", {"rms": round(rms, 4), "peak": round(peak, 4)}
                )
            except Exception:
                continue

    def stop(self) -> None:
        self._stopping.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                except OSError:
                    pass
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
        self._process = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        try:
            self.bus.publish("level", {"rms": 0.0, "peak": 0.0})
        except Exception:
            pass


def block_levels(block: bytes) -> tuple[float, float]:
    """Return the RMS and peak of one signed 16-bit little-endian block."""

    count = len(block) // 2
    if count == 0:
        return 0.0, 0.0
    samples = struct.unpack(f"<{count}h", block[: count * 2])
    total = 0.0
    peak = 0
    for sample in samples:
        total += float(sample) * float(sample)
        magnitude = -sample if sample < 0 else sample
        if magnitude > peak:
            peak = magnitude
    rms = math.sqrt(total / count) / 32768.0
    return min(1.0, rms), min(1.0, peak / 32768.0)
