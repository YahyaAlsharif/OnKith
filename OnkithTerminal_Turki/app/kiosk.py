"""Chromium as the display surface, never as the audio source.

The browser is used here as a GUI toolkit pointed at localhost: no tabs,
no address bar, no visible cursor. Audio belongs to the core process, so
this window can be killed at any time without interrupting playback, and
on a device with no display it is simply never started.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from app.audio_output import find_chromium

LogFunction = Callable[[str], None]


def kiosk_arguments(
    browser: str, url: str, profile_directory: Path
) -> list[str]:
    """Build the flags that make Chromium behave as an appliance screen."""

    arguments = [
        browser,
        "--kiosk",
        f"--app={url}",
        f"--user-data-dir={profile_directory}",
        "--no-first-run",
        "--no-default-browser-check",
        "--noerrdialogs",
        "--disable-infobars",
        "--disable-session-crashed-bubble",
        "--disable-features=TranslateUI,Translate",
        "--disable-pinch",
        "--overscroll-history-navigation=0",
        "--check-for-update-interval=31536000",
        # The page holds no audio, but a surface that autoplays nothing
        # still must never prompt.
        "--autoplay-policy=no-user-gesture-required",
    ]
    if platform.system() == "Linux":
        # Raspberry Pi OS runs a Wayland compositor by default; the hint
        # lets one command work on both Wayland and X11 sessions.
        arguments.append("--ozone-platform-hint=auto")
    return arguments


class KioskSurface:
    """Own at most one Chromium window pointed at the local core."""

    def __init__(
        self,
        url: str,
        profile_directory: Path,
        *,
        log: LogFunction = lambda message: None,
        browser: str | None = None,
        launcher: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self.url = url
        self.profile_directory = Path(profile_directory)
        self.log = log
        self._browser = browser
        self._launcher = launcher
        self._process: Any | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if self.running:
            return True
        browser = self._browser or find_chromium()
        if browser is None:
            self.log(
                "No Chromium build was found, so the screen cannot be shown. "
                "Music and voice continue without it."
            )
            return False
        self.profile_directory.mkdir(parents=True, exist_ok=True)
        arguments = kiosk_arguments(browser, self.url, self.profile_directory)
        launcher = self._launcher or self._spawn
        try:
            self._process = launcher(arguments)
        except OSError as error:
            self.log(f"The kiosk screen could not start: {error}")
            self._process = None
            return False
        return True

    @staticmethod
    def _spawn(arguments: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        if sys.platform == "win32":  # pragma: no cover - not a target device
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except OSError:
            pass


def hide_pointer_command() -> list[str] | None:
    """Return a command that hides the idle pointer, when one exists."""

    unclutter = shutil.which("unclutter")
    if unclutter is None:
        return None
    return [unclutter, "-idle", "0", "-root"]
