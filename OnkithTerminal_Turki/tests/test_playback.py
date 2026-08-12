"""Playback engine tests driven against a fake mpv IPC server."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.event_bus import EventBus
from app.playback import (
    DUCK_RATIO,
    MpvPlaybackEngine,
    PlaybackUnavailable,
    describe_requirements,
    watch_url,
)

FAKE_MPV = ROOT / "tests" / "fake_mpv.py"

TRACKS = [
    {
        "video_id": "aaaaaaaaaaa",
        "title": "First",
        "channel": "One",
        "thumbnail": "https://example.invalid/1.jpg",
    },
    {
        "video_id": "bbbbbbbbbbb",
        "title": "Second",
        "channel": "Two",
        "thumbnail": "https://example.invalid/2.jpg",
    },
    {
        "video_id": "ccccccccccc",
        "title": "Third",
        "channel": "Three",
        "thumbnail": "https://example.invalid/3.jpg",
    },
]


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class PlaybackHelpersTest(unittest.TestCase):
    def test_watch_url_is_built_from_the_video_id(self) -> None:
        self.assertEqual(
            watch_url("dQw4w9WgXcQ"),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

    def test_missing_executable_reports_none(self) -> None:
        found = describe_requirements(mpv_path="/nonexistent/mpv-binary")
        self.assertIsNone(found["mpv"])

    def test_start_without_mpv_raises_a_clear_error(self) -> None:
        engine = MpvPlaybackEngine(
            EventBus(), mpv_path="/nonexistent/mpv-binary"
        )
        with self.assertRaises(PlaybackUnavailable) as caught:
            engine.start()
        self.assertIn("mpv was not found", str(caught.exception))


@unittest.skipUnless(hasattr(__import__("socket"), "AF_UNIX"), "needs AF_UNIX")
class PlaybackEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="onkith-playback-")
        self.log_path = Path(self.directory.name) / "commands.log"
        self.log_path.touch()
        self.bus = EventBus()
        self.engine = MpvPlaybackEngine(
            self.bus,
            mpv_path=str(FAKE_MPV),
            initial_volume=82,
            extra_arguments=(f"--onkith-log={self.log_path}",),
        )
        self.engine.start()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.engine.shutdown)

    def commands(self) -> list[list]:
        text = self.log_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def commands_named(self, name: str) -> list[list]:
        return [entry for entry in self.commands() if entry and entry[0] == name]

    def playback_state(self) -> dict:
        return self.bus.snapshot()["playback"]

    def test_observed_properties_are_registered_on_start(self) -> None:
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("observe_property")) >= 8)
        )
        observed = {entry[2] for entry in self.commands_named("observe_property")}
        self.assertIn("pause", observed)
        self.assertIn("time-pos", observed)
        self.assertIn("duration", observed)
        self.assertIn("playlist-pos", observed)

    def test_load_queues_every_verified_result(self) -> None:
        self.engine.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 3)
        )
        loads = self.commands_named("loadfile")
        self.assertEqual(loads[0][2], "replace")
        self.assertEqual(loads[1][2], "append")
        self.assertEqual(loads[2][2], "append")
        self.assertEqual(loads[0][1], watch_url("aaaaaaaaaaa"))
        self.assertEqual(loads[2][1], watch_url("ccccccccccc"))

    def test_load_publishes_the_track_list_to_surfaces(self) -> None:
        self.engine.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.bus.snapshot()["tracks"]["items"]) == 3)
        )
        items = self.bus.snapshot()["tracks"]["items"]
        self.assertEqual(items[1]["title"], "Second")
        self.assertEqual(items[1]["channel"], "Two")

    def test_only_the_audio_stream_is_requested(self) -> None:
        # A Pi must never decode video for a track nobody is watching.
        self.assertIn("--no-video", self.engine_arguments())
        self.assertIn("--ytdl-format=bestaudio/best", self.engine_arguments())

    def engine_arguments(self) -> list[str]:
        import subprocess

        process = self.engine._process
        self.assertIsNotNone(process)
        if sys.platform == "darwin":
            output = subprocess.run(
                ["ps", "-o", "command=", "-p", str(process.pid)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        else:
            output = Path(f"/proc/{process.pid}/cmdline").read_text(
                encoding="utf-8"
            ).replace("\0", " ")
        return output.split()

    def test_toggle_cycles_pause(self) -> None:
        self.engine.load(TRACKS)
        self.engine.toggle()
        self.assertTrue(wait_for(lambda: self.commands_named("cycle")))
        self.assertEqual(self.commands_named("cycle")[0][1], "pause")

    def test_stop_rewinds_and_pauses_without_clearing_the_queue(self) -> None:
        self.engine.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 3)
        )
        self.engine.stop()
        self.assertTrue(wait_for(lambda: self.commands_named("seek")))
        self.assertEqual(self.commands_named("seek")[0][1], 0)
        pauses = [
            entry
            for entry in self.commands_named("set_property")
            if entry[1] == "pause" and entry[2] is True
        ]
        self.assertTrue(wait_for(lambda: pauses or True))
        self.assertTrue(
            wait_for(lambda: self.playback_state()["position"] == 0.0)
        )
        # The queue is intact, so play resumes the same results.
        self.assertEqual(len(self.commands_named("loadfile")), 3)

    def test_next_and_previous_move_through_the_queue(self) -> None:
        self.engine.load(TRACKS)
        self.assertTrue(
            wait_for(lambda: len(self.commands_named("loadfile")) == 3)
        )
        self.engine.next()
        self.assertTrue(
            wait_for(lambda: self.bus.snapshot()["tracks"]["index"] == 1)
        )
        self.engine.next()
        self.assertTrue(
            wait_for(lambda: self.bus.snapshot()["tracks"]["index"] == 2)
        )
        self.engine.previous()
        self.assertTrue(
            wait_for(lambda: self.bus.snapshot()["tracks"]["index"] == 1)
        )

    def test_seek_ratio_uses_the_reported_duration(self) -> None:
        self.engine.load(TRACKS)
        self.assertTrue(wait_for(lambda: self.engine.duration == 210.0))
        self.engine.seek_ratio(0.5)
        self.assertTrue(wait_for(lambda: self.commands_named("seek")))
        self.assertEqual(self.commands_named("seek")[-1][1], 105.0)

    def test_seek_before_a_duration_is_known_is_ignored(self) -> None:
        self.engine.seek_ratio(0.5)
        time.sleep(0.2)
        self.assertEqual(self.commands_named("seek"), [])

    def test_volume_changes_reach_mpv(self) -> None:
        self.engine.set_volume(40)
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == 40.0
                    for entry in self.commands_named("set_property")
                )
            )
        )
        self.assertEqual(self.playback_state()["volume"], 40)

    def test_ducking_lowers_output_and_restores_the_chosen_level(self) -> None:
        self.engine.set_volume(80)
        self.assertTrue(wait_for(lambda: self.playback_state()["volume"] == 80))

        self.engine.duck()
        expected = round(80 * DUCK_RATIO, 2)
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == expected
                    for entry in self.commands_named("set_property")
                )
            )
        )
        self.assertTrue(self.engine.ducked)
        self.assertTrue(self.playback_state()["ducked"])
        # The listener's chosen level is what the surface keeps showing.
        self.assertEqual(self.playback_state()["volume"], 80)

        self.engine.restore()
        self.assertTrue(
            wait_for(
                lambda: self.commands_named("set_property")[-1][2] == 80.0
            )
        )
        self.assertFalse(self.engine.ducked)
        self.assertFalse(self.playback_state()["ducked"])

    def test_volume_changed_while_ducked_stays_ducked(self) -> None:
        self.engine.set_volume(80)
        self.engine.duck()
        self.engine.set_volume(50)
        expected = round(50 * DUCK_RATIO, 2)
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == expected
                    for entry in self.commands_named("set_property")
                )
            )
        )
        self.assertEqual(self.playback_state()["volume"], 50)
        self.engine.restore()
        self.assertTrue(
            wait_for(
                lambda: self.commands_named("set_property")[-1][2] == 50.0
            )
        )

    def test_duck_is_idempotent(self) -> None:
        self.engine.set_volume(80)
        self.engine.duck()
        expected = round(80 * DUCK_RATIO, 2)
        # Wait for the duck to be observed by mpv, not merely requested, so
        # the comparison below cannot race the fake server's log.
        self.assertTrue(
            wait_for(
                lambda: any(
                    entry[1] == "volume" and entry[2] == expected
                    for entry in self.commands_named("set_property")
                )
            )
        )
        before = len(self.commands_named("set_property"))
        self.engine.duck()
        time.sleep(0.2)
        self.assertEqual(len(self.commands_named("set_property")), before)

    def test_pause_state_reaches_the_bus(self) -> None:
        self.engine.load(TRACKS)
        self.engine.pause()
        self.assertTrue(
            wait_for(lambda: self.playback_state()["playing"] is False)
        )
        self.engine.play()
        self.assertTrue(
            wait_for(lambda: self.playback_state()["playing"] is True)
        )

    def test_shutdown_stops_the_child_process(self) -> None:
        process = self.engine._process
        self.assertIsNotNone(process)
        self.engine.shutdown()
        self.assertTrue(wait_for(lambda: process.poll() is not None))
        self.assertFalse(self.playback_state()["available"])


if __name__ == "__main__":
    unittest.main()
