"""Ducking behaviour, speech state, and output-level computation."""

from __future__ import annotations

import struct
import sys
import threading
import time
import os
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio_output import (
    AudioStack,
    OutputLevelMeter,
    block_levels,
    describe_device,
    format_device_report,
)
from app.event_bus import EventBus
from app.speech import NullSpeechBackend, SpeechEngine


class RecordingBackend:
    name = "recording"

    def __init__(self, delay: float = 0.0) -> None:
        self.spoken: list[str] = []
        self.delay = delay
        self.level_during_speech: list[bool] = []
        self.observer = None

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        if self.observer is not None:
            self.level_during_speech.append(self.observer())
        if self.delay:
            time.sleep(self.delay)


class FailingBackend:
    name = "failing"

    def speak(self, text: str) -> None:
        raise RuntimeError("synthesis exploded")


class DuckingRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.ducked = False

    def duck(self) -> None:
        self.ducked = True
        self.events.append("duck")

    def restore(self) -> None:
        self.ducked = False
        self.events.append("restore")


class SpeechEngineTest(unittest.TestCase):
    def _engine(self, backend, recorder=None):
        recorder = recorder or DuckingRecorder()
        bus = EventBus()
        engine = SpeechEngine(
            bus,
            duck=recorder.duck,
            restore=recorder.restore,
            backend=backend,
        )
        engine.start()
        self.addCleanup(engine.shutdown)
        return engine, bus, recorder

    def test_speaking_ducks_and_then_restores(self) -> None:
        backend = RecordingBackend()
        engine, bus, recorder = self._engine(backend)
        backend.observer = lambda: recorder.ducked

        engine.speak("Opening Dancing with Monsters")
        self.assertTrue(engine.wait_idle(5))

        self.assertEqual(backend.spoken, ["Opening Dancing with Monsters"])
        self.assertEqual(recorder.events, ["duck", "restore"])
        # Music was actually held down while the voice was playing.
        self.assertEqual(backend.level_during_speech, [True])
        self.assertFalse(recorder.ducked)
        self.assertEqual(bus.snapshot()["assistant"]["state"], "idle")

    def test_the_assistant_state_reports_speaking(self) -> None:
        seen: list[str] = []
        backend = RecordingBackend(delay=0.15)
        engine, bus, _ = self._engine(backend)
        bus.subscribe(
            lambda message: seen.append(message.get("state"))
            if message.get("type") == "assistant"
            else None
        )
        engine.speak("hello")
        self.assertTrue(engine.wait_idle(5))
        self.assertEqual(seen, ["speaking", "idle"])

    def test_failed_synthesis_still_restores_the_level(self) -> None:
        engine, _, recorder = self._engine(FailingBackend())
        engine.speak("this will fail")
        self.assertTrue(engine.wait_idle(5))
        self.assertEqual(recorder.events, ["duck", "restore"])
        self.assertFalse(recorder.ducked)

    def test_back_to_back_utterances_duck_once(self) -> None:
        backend = RecordingBackend(delay=0.05)
        engine, _, recorder = self._engine(backend)
        for index in range(4):
            engine.speak(f"line {index}")
        self.assertTrue(engine.wait_idle(10))
        self.assertEqual(len(backend.spoken), 4)
        # The level must not flap between consecutive replies.
        self.assertEqual(recorder.events.count("duck"), 1)
        self.assertEqual(recorder.events.count("restore"), 1)

    def test_empty_text_is_not_spoken(self) -> None:
        backend = RecordingBackend()
        engine, _, recorder = self._engine(backend)
        engine.speak("   ")
        engine.speak("")
        time.sleep(0.2)
        self.assertEqual(backend.spoken, [])
        self.assertEqual(recorder.events, [])

    def test_shutdown_without_start_releases_any_duck(self) -> None:
        bus = EventBus()
        recorder = DuckingRecorder()
        engine = SpeechEngine(
            bus,
            duck=recorder.duck,
            restore=recorder.restore,
            backend=RecordingBackend(),
        )
        engine.shutdown()
        self.assertFalse(recorder.ducked)

    def test_null_backend_reports_itself(self) -> None:
        messages: list[str] = []
        backend = NullSpeechBackend(messages.append)
        backend.speak("one two three")
        self.assertEqual(len(messages), 1)
        self.assertIn("No local speech backend", messages[0])


class WhichVoiceWins(unittest.TestCase):
    """Which voice is chosen, and what happens when it is not installed."""

    def test_kokoro_is_the_voice_the_device_ships_with(self) -> None:
        from app import speech

        self.assertEqual(speech._AUTOMATIC_ORDER[0], "kokoro")

    def test_a_lighter_voice_is_always_behind_it(self) -> None:
        # Kokoro holds a 310 MB graph and takes a second or two a
        # sentence on a fast laptop. A board that cannot afford it has to
        # land on something rather than going mute.
        from app import speech

        order = speech._AUTOMATIC_ORDER
        self.assertEqual(order, ("kokoro", "piper", "espeak", "say"))
        for lighter in ("piper", "espeak", "say"):
            self.assertIn(lighter, order)

    def test_every_named_voice_can_actually_be_built(self) -> None:
        from app import speech

        for name in speech._AUTOMATIC_ORDER:
            with self.subTest(voice=name):
                self.assertIn(name, speech._BACKEND_BUILDERS)

    def test_mpv_can_play_the_raw_stream_when_nothing_else_will(self) -> None:
        # The three sinks ahead of it are Linux-only, which left both
        # neural voices unselectable on a Mac and made them impossible to
        # hear before they reached the Pi. mpv is not a new dependency:
        # it already plays every note of music this device makes.
        from app import speech

        with mock.patch.object(
            speech.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/mpv" if name == "mpv" else None,
        ):
            command = speech._raw_player(24000)
        self.assertIsNotNone(command)
        self.assertIn("/usr/bin/mpv", command)
        self.assertIn("--demuxer-rawaudio-rate=24000", command)

    def test_a_machine_with_no_sink_at_all_reports_none(self) -> None:
        from app import speech

        with mock.patch.object(speech.shutil, "which", return_value=None):
            self.assertIsNone(speech._raw_player(24000))

    def test_kokoro_without_its_two_files_is_skipped(self) -> None:
        # Both are required. Half a configuration used to fall through in
        # silence and the device just sounded wrong.
        from app import speech

        with mock.patch.dict(
            os.environ,
            {"ONKITH_KOKORO_MODEL": "/nowhere/model.onnx"},
            clear=False,
        ):
            os.environ.pop("ONKITH_KOKORO_VOICES", None)
            self.assertIsNone(speech.build_kokoro_backend())


class OutputLevelTest(unittest.TestCase):
    def test_silence_measures_zero(self) -> None:
        rms, peak = block_levels(struct.pack("<800h", *([0] * 800)))
        self.assertEqual(rms, 0.0)
        self.assertEqual(peak, 0.0)

    def test_full_scale_measures_one(self) -> None:
        rms, peak = block_levels(struct.pack("<100h", *([32767] * 100)))
        self.assertAlmostEqual(rms, 1.0, places=3)
        self.assertAlmostEqual(peak, 1.0, places=3)

    def test_peak_exceeds_rms_for_a_sparse_signal(self) -> None:
        samples = [0] * 99 + [32000]
        rms, peak = block_levels(struct.pack("<100h", *samples))
        self.assertGreater(peak, rms)
        self.assertAlmostEqual(peak, 32000 / 32768, places=3)

    def test_empty_block_is_safe(self) -> None:
        self.assertEqual(block_levels(b""), (0.0, 0.0))

    def test_meter_without_a_monitor_is_unavailable(self) -> None:
        meter = OutputLevelMeter(EventBus(), stack=AudioStack(), command=[])
        self.assertFalse(meter.available)
        self.assertFalse(meter.start())

    def test_meter_publishes_levels_from_a_capture_command(self) -> None:
        bus = EventBus()
        loud = struct.pack("<800h", *([16000] * 800))
        script = (
            "import sys;"
            "sys.stdout.buffer.write(" + repr(loud) + " * 4);"
            "sys.stdout.buffer.flush()"
        )
        meter = OutputLevelMeter(
            bus, stack=AudioStack(), command=[sys.executable, "-c", script]
        )
        self.assertTrue(meter.available)
        self.assertTrue(meter.start())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if bus.snapshot()["level"]["rms"] > 0:
                break
            time.sleep(0.02)
        measured = bus.snapshot()["level"]["rms"]
        meter.stop()
        self.assertAlmostEqual(measured, 16000 / 32768, places=2)
        # Stopping the meter returns the surface to silence.
        self.assertEqual(bus.snapshot()["level"]["rms"], 0.0)


class DeviceReportTest(unittest.TestCase):
    def test_report_contains_the_fields_needed_to_confirm_a_device(self) -> None:
        report = describe_device()
        self.assertIn("os", report)
        self.assertIn("kernel", report["os"])
        self.assertIn("architecture", report["os"])
        self.assertIn("audio", report)
        self.assertIn("server", report["audio"])
        self.assertIn("playback", report)
        self.assertIn("mpv", report["playback"])

    def test_report_renders_without_raising(self) -> None:
        text = format_device_report(describe_device())
        self.assertIn("Audio server", text)
        self.assertIn("mpv", text)
        self.assertIn("Chromium", text)


if __name__ == "__main__":
    unittest.main()
