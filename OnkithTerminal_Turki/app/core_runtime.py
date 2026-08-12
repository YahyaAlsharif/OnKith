"""Assemble the services the core process owns.

Playback, speech, output metering, the localhost surface server, and the
optional kiosk screen are all started and stopped here. Surfaces are
strictly optional: if the screen never starts, or is killed, the audio
and voice path is unaffected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from app.audio_output import OutputLevelMeter, detect_audio_stack
from app.core_server import CoreServer
from app.event_bus import EventBus
from app.kiosk import KioskSurface
from app.onkith_music import create_player_page
from app.playback import MpvPlaybackEngine, PlaybackUnavailable
from app.light import LightController
from app.speech import SpeechEngine

LogFunction = Callable[[str], None]

# Prefixes this project creates under the system temp directory. Both are
# TemporaryDirectory, so both clean up after themselves on any exit that
# runs, and neither does on a power cut or a kill, which for an appliance
# is most of them.
TEMPORARY_PREFIXES = ("onkith-core-", "onkith-mpv-")
# Old enough that nothing running could own it. A device that is
# power-cycled daily accumulates one of these per run forever otherwise,
# and this one has an SD card rather than a disk.
STALE_TEMPORARY_HOURS = 6


def sweep_stale_temporary_directories(
    *,
    older_than_hours: float = STALE_TEMPORARY_HOURS,
    log: LogFunction = lambda message: None,
) -> int:
    """Remove leftovers from runs that did not get to clean up after them.

    3.3 GB of these had accumulated on the machine this was written on,
    across fifty directories, none of which belonged to a live process.
    Every one is a run that was killed or a power cut, which on a device
    that gets switched off at the wall is the ordinary way it ends.

    Age is the only test. Ownership would be better and there is nothing
    reliable to read it from, so the window is set wide enough that a
    directory inside it cannot belong to anything still running.
    """

    import shutil
    import time

    cutoff = time.time() - older_than_hours * 3600
    removed = 0
    root = Path(tempfile.gettempdir())
    for prefix in TEMPORARY_PREFIXES:
        for path in root.glob(f"{prefix}*"):
            try:
                if not path.is_dir() or path.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
            except OSError:
                # Somebody else's, or already gone. Never a reason to
                # stop the device starting.
                continue
    if removed:
        log(f"Removed {removed} leftover temporary directories")
    return removed


class CoreRuntime:
    """Own audio, state, and the optional screen for one session."""

    def __init__(
        self,
        *,
        headless: bool = False,
        log: LogFunction = lambda message: None,
        port: int = 0,
        mpv_path: str | None = None,
        ytdl_path: str | None = None,
        initial_volume: int = 82,
        playback: Any | None = None,
        kiosk: Any | None = None,
        meter: Any | None = None,
        speech: Any | None = None,
        light: Any | None = None,
        light_settings: dict[str, Any] | None = None,
    ) -> None:
        self.headless = headless
        self.log = log
        self.bus = EventBus()
        self.audio_stack = detect_audio_stack()

        sweep_stale_temporary_directories(log=log)
        self._directory = tempfile.TemporaryDirectory(prefix="onkith-core-")
        root = Path(self._directory.name)
        self.web_directory = root / "web"
        self.profile_directory = root / "screen-profile"

        self.playback = (
            playback
            if playback is not None
            else MpvPlaybackEngine(
                self.bus,
                mpv_path=mpv_path,
                ytdl_path=ytdl_path,
                log=log,
                initial_volume=initial_volume,
            )
        )
        # A display asset must never be able to take down the voice path.
        self.surface_available = True
        try:
            create_player_page(self.web_directory)
        except (OSError, FileNotFoundError) as error:
            self.surface_available = False
            log(f"The player surface could not be prepared: {error}")
        self.server = CoreServer(
            self.web_directory, self.bus, on_command=self.handle_command, port=port
        )
        self.speech = (
            speech
            if speech is not None
            else SpeechEngine(
                self.bus,
                duck=self.playback.duck,
                restore=self.playback.restore,
                log=log,
            )
        )
        self.meter = (
            meter
            if meter is not None
            else OutputLevelMeter(self.bus, stack=self.audio_stack)
        )
        # Constructed everywhere, wired to a GPIO only where one exists.
        # LightController settles that for itself, so nothing above here
        # has to know whether this machine is the Pi.
        self.light = (
            light
            if light is not None
            else LightController(self.bus, light_settings, log=log)
        )
        self.kiosk = (
            kiosk
            if kiosk is not None
            else KioskSurface(
                self.surface_url, self.profile_directory, log=log
            )
        )
        self.playback_available = False

    @property
    def surface_url(self) -> str:
        return f"{self.server.url}?kiosk=1"

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.server.start()
        try:
            self.playback.start()
            self.playback_available = True
        except PlaybackUnavailable as error:
            # Voice and transcript work must survive a device with no mpv.
            self.playback_available = False
            self.log(str(error))
            self.bus.status(str(error), "error")

        self.speech.start()
        if self.meter.available:
            self.meter.start()
        elif self.audio_stack.notes:
            self.log(self.audio_stack.notes[0])

        if self.headless:
            self.log("Running headless: no screen is started.")
            return
        if not self.surface_available:
            self.log("Continuing without a screen; its assets are missing.")
            return
        if not self.kiosk.start():
            self.log(
                "Continuing without a screen; audio and voice are unaffected."
            )

    def shutdown(self) -> None:
        # The screen goes first: it owns nothing that the others need.
        try:
            self.kiosk.stop()
        except Exception:
            pass
        try:
            self.speech.shutdown()
        except Exception:
            pass
        try:
            self.meter.stop()
        except Exception:
            pass
        try:
            self.playback.shutdown()
        except Exception:
            pass
        try:
            # Dark on the way out. A lamp left burning by a process that
            # has exited is the one failure here somebody has to walk
            # over and unplug.
            self.light.close()
        except Exception:
            pass
        try:
            self.server.shutdown()
        except Exception:
            pass
        self._directory.cleanup()

    # ------------------------------------------------------------ commands

    def handle_command(self, name: str, payload: dict[str, Any]) -> None:
        """Apply one transport command sent by a surface."""

        if not self.playback_available:
            return
        if name == "toggle":
            self.playback.toggle()
        elif name == "play":
            self.playback.play()
        elif name == "pause":
            self.playback.pause()
        elif name == "stop":
            self.playback.stop()
        elif name == "next":
            self.playback.next()
        elif name == "previous":
            self.playback.previous()
        elif name == "mute":
            if "value" in payload:
                self.playback.set_muted(bool(payload["value"]))
            else:
                self.playback.toggle_mute()
        elif name == "select":
            index = _number(payload.get("index"))
            if index is not None:
                self.playback.select(int(index))
        elif name == "seek":
            ratio = _number(payload.get("ratio"))
            if ratio is not None:
                self.playback.seek_ratio(ratio)
        elif name == "volume":
            value = _number(payload.get("value"))
            if value is not None:
                self.playback.set_volume(value)

    def step_volume(self, delta: int) -> None:
        """Move the level by voice, which has no number to hand over.

        The knob on screen sets a value. A person says "louder", which is
        a direction, so the step is applied to whatever the level is now
        rather than to a remembered one that the knob may have moved.

        Clamping belongs to set_volume, so asking for louder at the top
        is a no-op rather than an error, which is what somebody standing
        in front of it would expect.
        """

        if not self.playback_available:
            return
        self.playback.set_volume(self.playback.volume + int(delta))

    def handle_light(self, intent: str) -> None:
        """One spoken instruction for the lamp.

        The intent arrives rather than a number because on and off are
        not points on the same scale: turning it off has to remember the
        level so that turning it back on returns to it.
        """

        if intent == "LIGHT_ON":
            self.light.on()
        elif intent == "LIGHT_OFF":
            self.light.off()
        elif intent == "BRIGHTER":
            self.light.step(self.light.step_size)
        elif intent == "DIMMER":
            self.light.step(-self.light.step_size)

    # -------------------------------------------------------------- speech

    def speak(self, text: str) -> None:
        self.speech.speak(text)

    def set_assistant_state(self, state: str) -> None:
        self.bus.publish("assistant", {"state": state})

    def set_panel(self, open_panel: bool | None) -> None:
        """Show or hide the player because somebody asked for it out loud.

        None hands the decision back to the screen, which follows whether
        there are any results. A spoken instruction outranks that until the
        next search, and no further, so asking for it to go away does not
        mean it stays away the next time music starts.
        """

        self.bus.publish(
            "panel",
            {"open": None if open_panel is None else bool(open_panel)},
        )

    def set_gate(self, state: str) -> None:
        """Say what became of the last thing anybody said.

        Held stayed on this device, sent went to the model as placeholders,
        blocked was stopped by the masker. It stands until the next utterance
        replaces it, because it is a record of what happened rather than an
        indicator of what is happening.
        """

        self.bus.publish("gate", {"state": state})

    def set_listening(self, hearing: bool) -> None:
        """Say whether the microphone is holding an utterance open.

        This is the one thing a surface has to be told rather than work out.
        The level topic is the output sink, so it moves for music as readily
        as for a voice, and everything else the screen shows happens after an
        utterance has already closed. Without this the device looks asleep
        for exactly as long as somebody is talking to it.
        """

        self.bus.publish("listening", {"hearing": bool(hearing)})


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
