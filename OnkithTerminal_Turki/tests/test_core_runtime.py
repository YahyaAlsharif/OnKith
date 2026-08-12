"""Integration cover for the phase 1 acceptance criteria.

These exercise the assembled core: a surface can drive playback, killing
the screen does not touch audio, headless starts nothing visual, and
speech ducks and restores the music level.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core_runtime import CoreRuntime, sweep_stale_temporary_directories
from app.kiosk import KioskSurface, kiosk_arguments
from app.playback import DUCK_RATIO, MpvPlaybackEngine
from tests.test_core_server import WebSocketTestClient

FAKE_MPV = ROOT / "tests" / "fake_mpv.py"

TRACKS = [
    {"video_id": "aaaaaaaaaaa", "title": "First", "channel": "One", "thumbnail": ""},
    {"video_id": "bbbbbbbbbbb", "title": "Second", "channel": "Two", "thumbnail": ""},
]


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def temporary_profile_for(case) -> str:
    """A Chromium profile directory that goes away with the test.

    These were mkdtemp with nothing removing them, so a run left one
    behind every time and eighteen had built up. Leaking exactly what
    the sweep in core_runtime exists to clear is not a good look.
    """

    import shutil

    path = tempfile.mkdtemp(prefix="onkith-profile-")
    case.addCleanup(shutil.rmtree, path, True)
    return path


class RecordingKiosk:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def start(self) -> bool:
        self.starts += 1
        return True

    def stop(self) -> None:
        self.stops += 1


class SilentBackend:
    name = "silent"

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        time.sleep(0.1)


@unittest.skipUnless(hasattr(__import__("socket"), "AF_UNIX"), "needs AF_UNIX")
class CoreRuntimeTest(unittest.TestCase):
    def build(self, *, headless: bool = True, kiosk=None):
        self.directory = tempfile.TemporaryDirectory(prefix="onkith-runtime-")
        self.log_path = Path(self.directory.name) / "commands.log"
        self.log_path.touch()
        self.addCleanup(self.directory.cleanup)

        runtime = CoreRuntime(headless=headless, kiosk=kiosk)
        # Replace the engine with one pointed at the fake mpv server.
        runtime.playback = MpvPlaybackEngine(
            runtime.bus,
            mpv_path=str(FAKE_MPV),
            initial_volume=82,
            extra_arguments=(f"--onkith-log={self.log_path}",),
        )
        runtime.speech._duck = runtime.playback.duck
        runtime.speech._restore = runtime.playback.restore
        self.backend = SilentBackend()
        runtime.speech._backend = self.backend
        runtime.start()
        self.addCleanup(runtime.shutdown)
        return runtime

    def commands(self) -> list[list]:
        text = self.log_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def commands_named(self, name: str) -> list[list]:
        return [entry for entry in self.commands() if entry and entry[0] == name]

    def test_headless_starts_no_screen_but_serves_and_plays(self) -> None:
        kiosk = RecordingKiosk()
        runtime = self.build(headless=True, kiosk=kiosk)
        self.assertEqual(kiosk.starts, 0)
        self.assertTrue(runtime.playback_available)

        runtime.playback.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 2)
        )
        # Commands still work with no display attached.
        runtime.handle_command("toggle", {})
        self.assertTrue(wait_for(lambda: self.commands_named("cycle")))

    def test_the_screen_starts_when_not_headless(self) -> None:
        kiosk = RecordingKiosk()
        self.build(headless=False, kiosk=kiosk)
        self.assertEqual(kiosk.starts, 1)

    def test_a_surface_drives_playback_over_the_websocket(self) -> None:
        runtime = self.build(headless=True, kiosk=RecordingKiosk())
        runtime.playback.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 2)
        )

        client = WebSocketTestClient("127.0.0.1", runtime.server.port)
        self.addCleanup(client.close)
        snapshot = client.receive_json()
        self.assertEqual(snapshot["type"], "snapshot")
        self.assertEqual(len(snapshot["tracks"]["items"]), 2)

        client.send_json({"command": "volume", "payload": {"value": 45}})
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == 45.0
                    for entry in self.commands_named("set_property")
                )
            )
        )
        client.send_json({"command": "next", "payload": {}})
        self.assertTrue(wait_for(lambda: self.commands_named("playlist-next")))
        client.send_json({"command": "stop", "payload": {}})
        self.assertTrue(wait_for(lambda: self.commands_named("seek")))

    def test_killing_the_screen_does_not_interrupt_playback(self) -> None:
        # A real child process stands in for Chromium so it can be killed.
        spawned: list[subprocess.Popen] = []

        def launcher(arguments: list[str]) -> subprocess.Popen:
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            spawned.append(process)
            return process

        kiosk = KioskSurface(
            "http://127.0.0.1:1/player.html",
            Path(temporary_profile_for(self)),
            browser="/bin/true",
            launcher=launcher,
        )
        runtime = self.build(headless=False, kiosk=kiosk)
        self.assertEqual(len(spawned), 1)
        self.assertIsNone(spawned[0].poll())

        runtime.playback.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 2)
        )

        # Kill the screen outright, the way a crash or pkill would.
        spawned[0].kill()
        spawned[0].wait(timeout=5)
        self.assertIsNotNone(spawned[0].poll())

        # Audio is untouched: the process is alive and still takes commands.
        self.assertTrue(runtime.playback.running)
        before = len(self.commands_named("cycle"))
        runtime.handle_command("toggle", {})
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("cycle")) == before + 1)
        )

    def test_speech_ducks_the_music_and_restores_it(self) -> None:
        runtime = self.build(headless=True, kiosk=RecordingKiosk())
        runtime.playback.load(TRACKS)
        runtime.playback.set_volume(80)
        self.assertTrue(
            wait_for(lambda: runtime.bus.snapshot()["playback"]["volume"] == 80)
        )

        runtime.speak("Opening Dancing with Monsters")
        ducked = round(80 * DUCK_RATIO, 2)
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == ducked
                    for entry in self.commands_named("set_property")
                )
            ),
            "music was not ducked while speech played",
        )
        self.assertTrue(runtime.speech.wait_idle(5))
        self.assertTrue(
            wait_for(
                lambda: self.commands_named("set_property")[-1][2] == 80.0
            ),
            "music level was not restored after speech",
        )
        self.assertEqual(self.backend.spoken, ["Opening Dancing with Monsters"])
        self.assertFalse(runtime.playback.ducked)

    def test_a_spoken_step_moves_the_level_from_wherever_it_is(self) -> None:
        """Voice has no number to hand over, only a direction.

        So the step applies to the level as it stands, not to one this
        runtime remembered, because the knob on screen moves it too.
        """

        runtime = self.build(headless=True, kiosk=RecordingKiosk())
        runtime.playback.load(TRACKS)
        runtime.playback.set_volume(50)
        runtime.step_volume(12)
        self.assertEqual(runtime.playback.volume, 62)
        runtime.step_volume(-12)
        self.assertEqual(runtime.playback.volume, 50)

    def test_asking_for_louder_at_the_top_is_not_an_error(self) -> None:
        # Somebody standing in front of it expects nothing to happen,
        # not a failure, and certainly not a level above 100.
        runtime = self.build(headless=True, kiosk=RecordingKiosk())
        runtime.playback.load(TRACKS)
        runtime.playback.set_volume(96)
        runtime.step_volume(12)
        self.assertEqual(runtime.playback.volume, 100)
        runtime.playback.set_volume(4)
        runtime.step_volume(-12)
        self.assertEqual(runtime.playback.volume, 0)

    def test_the_player_page_is_served_from_localhost(self) -> None:
        runtime = self.build(headless=True, kiosk=RecordingKiosk())
        with urllib.request.urlopen(runtime.server.url, timeout=5) as response:
            body = response.read().decode("utf-8")
        self.assertIn('id="machine-viewer"', body)
        self.assertIn("assets/player-app-v1.mjs", body)
        # The recorder is served from this device too, never from a CDN.
        self.assertIn("assets/spline/spline-viewer.js", body)
        self.assertTrue(runtime.surface_url.endswith("?kiosk=1"))


class KioskArgumentsTest(unittest.TestCase):
    def test_the_window_is_an_appliance_screen(self) -> None:
        arguments = kiosk_arguments(
            "/usr/bin/chromium", "http://localhost:9999/player.html", Path("/tmp/p")
        )
        self.assertIn("--kiosk", arguments)
        self.assertIn("--app=http://localhost:9999/player.html", arguments)
        self.assertIn("--no-first-run", arguments)
        self.assertIn("--noerrdialogs", arguments)
        # A kiosk must never show browser chrome for navigation.
        self.assertFalse(any(item.startswith("--new-window") for item in arguments))

    def test_a_missing_browser_is_reported_without_raising(self) -> None:
        messages: list[str] = []
        surface = KioskSurface(
            "http://localhost:1/player.html",
            Path(temporary_profile_for(self)),
            log=messages.append,
            browser=None,
        )
        with patch("app.kiosk.find_chromium", return_value=None):
            self.assertFalse(surface.start())
        self.assertTrue(messages)
        self.assertIn("Chromium", messages[0])


if __name__ == "__main__":
    unittest.main()


class StaleTemporaryDirectoryTests(unittest.TestCase):
    """A device switched off at the wall never gets to clean up.

    Both temporary directories this project makes are TemporaryDirectory,
    so both remove themselves on any exit that runs, and a power cut is
    not one. 3.3 GB across fifty directories had built up on the machine
    this was written on, and the target has an SD card.
    """

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="onkith-sweep-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def make(self, name: str, *, hours_old: float) -> Path:
        path = self.root / name
        path.mkdir()
        (path / "leftover").write_text("x")
        when = time.time() - hours_old * 3600
        os.utime(path, (when, when))
        return path

    def sweep(self, **kwargs) -> int:
        with patch("tempfile.gettempdir", return_value=str(self.root)):
            return sweep_stale_temporary_directories(**kwargs)

    def test_old_leftovers_go(self) -> None:
        core = self.make("onkith-core-abc", hours_old=48)
        mpv = self.make("onkith-mpv-def", hours_old=7)
        self.assertEqual(self.sweep(), 2)
        self.assertFalse(core.exists())
        self.assertFalse(mpv.exists())

    def test_a_directory_that_could_still_be_in_use_stays(self) -> None:
        # The running session's own directory is the obvious one to
        # delete by accident, and it is minutes old, not hours.
        live = self.make("onkith-core-live", hours_old=0.1)
        self.assertEqual(self.sweep(), 0)
        self.assertTrue(live.exists())

    def test_nothing_else_in_the_temp_directory_is_touched(self) -> None:
        other = self.make("something-else", hours_old=99)
        self.sweep()
        self.assertTrue(other.exists())

    def test_a_directory_it_cannot_remove_does_not_stop_startup(self) -> None:
        self.make("onkith-core-locked", hours_old=48)
        with patch("shutil.rmtree", side_effect=OSError("busy")):
            # Raising here would mean a full temp directory stops the
            # device from starting, which is the wrong way round.
            self.sweep()
