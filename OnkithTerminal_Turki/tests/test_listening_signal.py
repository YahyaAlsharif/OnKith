"""The screen is told when the microphone is holding an utterance open.

Everything else a surface shows happens after an utterance has already
closed: the assistant only wakes up once there is masked text, and the level
topic is the output sink, which moves for music as readily as for a voice.
Without this the device looked asleep for exactly as long as somebody was
talking to it, which is the worst possible moment to look asleep.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.event_bus import EventBus
from app.omit_ai_live import HEARING_BY_EVENT, audio_event_bridge


class Core:
    """Only the part of the runtime the bridge is allowed to touch."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    def set_listening(self, hearing: bool) -> None:
        self.bus.publish("listening", {"hearing": bool(hearing)})


class ExplodingCore(Core):
    def set_listening(self, hearing: bool) -> None:
        raise RuntimeError("the surface went away")


class ListeningSignalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = EventBus()
        self.seen: list[tuple[str, dict]] = []
        self.relay = audio_event_bridge(Core(self.bus), self.record)

    def record(self, name: str, payload: dict) -> None:
        self.seen.append((name, payload))

    def hearing(self) -> bool:
        return self.bus.snapshot()["listening"]["hearing"]

    def test_it_starts_quiet(self) -> None:
        self.assertFalse(self.hearing())

    def test_an_utterance_lights_it_and_closing_puts_it_out(self) -> None:
        self.relay("recording_started", {"rms": 0.2})
        self.assertTrue(self.hearing())
        self.relay("segment_closed", {"reason": "endpoint_silence"})
        self.assertFalse(self.hearing())

    def test_a_long_utterance_stays_lit_across_a_forced_split(self) -> None:
        # Anything past the maximum is cut into pieces mid sentence. Each cut
        # closes a segment, so without the continuation the screen would go
        # dark on anyone who talks for more than a few seconds.
        self.relay("recording_started", {})
        self.relay("segment_closed", {"reason": "maximum_utterance"})
        self.relay("forced_split_continuation", {"overlap_seconds": 0.35})
        self.assertTrue(self.hearing())
        self.relay("segment_closed", {"reason": "endpoint_silence"})
        self.assertFalse(self.hearing())

    def test_shutting_down_mid_utterance_still_puts_it_out(self) -> None:
        self.relay("recording_started", {})
        self.relay("segment_closed", {"reason": "shutdown"})
        self.assertFalse(self.hearing())

    def test_every_other_event_leaves_it_alone(self) -> None:
        self.relay("recording_started", {})
        for name in ("microphone_status", "queue_overflow", "error", "status"):
            with self.subTest(name=name):
                self.relay(name, {})
                self.assertTrue(self.hearing())

    def test_the_terminal_still_receives_everything(self) -> None:
        # The bridge is a relay, not a filter. The terminal log must not lose
        # events just because a screen is also listening.
        for name in ("microphone_open", "recording_started", "segment_closed"):
            self.relay(name, {"n": name})
        self.assertEqual([name for name, _ in self.seen], [
            "microphone_open",
            "recording_started",
            "segment_closed",
        ])

    def test_a_surface_that_fails_cannot_stop_the_microphone(self) -> None:
        relay = audio_event_bridge(ExplodingCore(self.bus), self.record)
        relay("recording_started", {})
        self.assertEqual(self.seen, [("recording_started", {})])

    def test_it_works_with_no_core_at_all(self) -> None:
        # Headless runs and the text-only paths have no surface to tell.
        relay = audio_event_bridge(None, self.record)
        relay("recording_started", {})
        self.assertEqual(len(self.seen), 1)

    def test_the_names_it_watches_are_names_the_capture_lane_emits(self) -> None:
        source = (ROOT / "app" / "live_audio.py").read_text(encoding="utf-8")
        for name in HEARING_BY_EVENT:
            with self.subTest(name=name):
                self.assertIn(f'"{name}"', source)


if __name__ == "__main__":
    unittest.main()
