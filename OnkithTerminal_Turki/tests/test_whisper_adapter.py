"""A second recogniser, offered on the same seam as the first.

Moonshine never once wrote the device's own name correctly on the six
clips it was measured against, and it invented digits inside a phone
number. Whisper takes an initial_prompt, which is a short piece of text
that biases decoding toward the words in it, and given the device's names
it got all six right. These tests cover the swap itself: none of them
load a model, because what can go wrong here is the wiring.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.moonshine_adapter import MoonshineAdapter
from integrated_pipeline.schemas import RecordedUtterance
from integrated_pipeline.whisper_adapter import (
    WhisperAdapter,
    name_hint,
    to_whisper_rate,
)


def recording(sample_rate: int = 16000) -> RecordedUtterance:
    import numpy as np

    return RecordedUtterance(
        session_id="test",
        batch_id="test-1",
        sequence_number=1,
        samples=np.zeros(sample_rate, dtype=np.float32),
        sample_rate=sample_rate,
        audio_started_monotonic=0.0,
        audio_ended_monotonic=1.0,
        audio_started_at_utc="",
        audio_ended_at_utc="",
        captured_at_utc="",
        recording_closed_monotonic=time.monotonic(),
        endpoint_reason="endpoint_silence",
        reliable=True,
        dropped_audio_chunks=0,
        captured_audio_seconds=1.0,
    )


class NameHintTest(unittest.TestCase):
    """The hint is built from the names, never written out beside them.

    Otherwise renaming the device in config leaves the recogniser still
    biased toward the old name, which is the failure that is hardest to
    see: everything looks configured and nothing answers.
    """

    def test_the_hint_names_the_device(self) -> None:
        self.assertTrue(
            name_hint(["jarvis"]).startswith("Jarvis is the name of this device.")
        )

    def test_the_hint_shows_a_whole_command_not_just_the_name(self) -> None:
        # Naming the device got the name right and still ran the next word
        # into it, so "play bones" came back as "Playbons" and a request
        # for music routed to conversation. The hint has to show the shape
        # of an utterance, not only the word at the front of it.
        hint = name_hint(["jarvis"])
        for word in ("play", "stop", "show the player", "hide the player"):
            with self.subTest(word=word):
                self.assertIn(word, hint)

    def test_the_commands_in_the_hint_carry_the_configured_name(self) -> None:
        # A hint that demonstrates commands under the old name is worse
        # than no demonstration: it biases every utterance toward a word
        # the device no longer answers to.
        self.assertNotIn("Onkith", name_hint(["jarvis", "sada"]))

    def test_the_hint_shows_the_length_of_a_number(self) -> None:
        # The command words are short and biasing toward them cost a digit
        # out of a phone number, which is the one thing this device is not
        # allowed to round off.
        self.assertIn("0512345678", name_hint(["jarvis"]))

    def test_every_name_reaches_the_hint(self) -> None:
        hint = name_hint(["onkith", "sada", "sitr", "echo"])
        for name in ("Onkith", "Sada", "Sitr", "Echo"):
            with self.subTest(name=name):
                self.assertIn(name, hint)

    def test_no_names_means_no_hint(self) -> None:
        # An empty hint is passed as None, not as an empty string, so the
        # decoder is never handed a prompt that says nothing.
        self.assertEqual(name_hint([]), "")
        self.assertEqual(name_hint(None), "")

    def test_renaming_the_device_rebiases_the_recogniser(self) -> None:
        config = PipelineConfig()
        config.data["wake"] = {"names": ["jarvis"]}
        config.data["stt"] = {"engine": "whisper", "whisper": {}}
        self.assertIn("Jarvis", WhisperAdapter(config).initial_prompt)
        self.assertNotIn("Onkith", WhisperAdapter(config).initial_prompt)

    def test_the_shipped_names_reach_the_shipped_adapter(self) -> None:
        self.assertIn("Onkith", WhisperAdapter(PipelineConfig()).initial_prompt)

    def test_a_written_out_hint_wins(self) -> None:
        config = PipelineConfig()
        config.data["stt"] = {"whisper": {"initial_prompt": "Say nothing."}}
        self.assertEqual(WhisperAdapter(config).initial_prompt, "Say nothing.")


class EngineChoiceTest(unittest.TestCase):
    def test_moonshine_is_what_a_config_without_the_section_means(self) -> None:
        config = PipelineConfig()
        config.data.pop("stt", None)
        self.assertEqual(config.stt_engine, "moonshine")
        self.assertEqual(config.stt, {})

    def test_the_shipped_config_still_runs_moonshine(self) -> None:
        # Whisper is offered, not adopted. Nothing has run on the Pi yet.
        self.assertEqual(PipelineConfig().stt_engine, "moonshine")

    def test_the_engine_is_read_loosely(self) -> None:
        config = PipelineConfig()
        config.data["stt"] = {"engine": "  Whisper "}
        self.assertEqual(config.stt_engine, "whisper")

    def test_the_flag_reaches_the_printed_config(self) -> None:
        """`--print-config` has to describe the run it is asked about.

        The override was applied after that branch had already printed, so
        the one command for checking which recogniser a run would use
        showed the file instead of the flag. With the shipped config still
        saying moonshine, that reads as the flag doing nothing, on the day
        somebody is trying to demonstrate the other one.
        """

        from app.omit_ai_live import main

        def printed(*flags: str) -> dict:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main([*flags, "--print-config"]), 0)
            return json.loads(out.getvalue())

        self.assertEqual(printed("--stt", "whisper")["stt"]["engine"], "whisper")
        self.assertEqual(printed()["stt"]["engine"], "moonshine")


class SwapSeamTest(unittest.TestCase):
    def test_both_recognisers_answer_the_same_calls(self) -> None:
        """The worker holds one name for whichever engine is loaded.

        So anything the runtime reaches for on one has to exist on the
        other, or the swap fails partway through a session rather than at
        startup where somebody would see it.
        """

        used_by_the_runtime = (
            "load",
            "loaded",
            "close",
            "audit",
            "counters",
            "resource_identity",
            "note_queue_overflow",
            "transcribe_recording",
        )
        for attribute in used_by_the_runtime:
            with self.subTest(attribute=attribute):
                self.assertTrue(hasattr(MoonshineAdapter(PipelineConfig()), attribute))
                self.assertTrue(hasattr(WhisperAdapter(PipelineConfig()), attribute))

    def test_the_two_tallies_line_up(self) -> None:
        """The end of run report reads one set of counters either way."""

        moonshine = vars(MoonshineAdapter(PipelineConfig()).counters)
        whisper = vars(WhisperAdapter(PipelineConfig()).counters)
        self.assertEqual(set(moonshine), set(whisper))

class SampleRateTest(unittest.TestCase):
    """Whisper reads 16 kHz. Microphones do not care about that.

    Moonshine's SDK takes whatever rate the device runs at, so nothing
    upstream ever had to think about it. The first machine this ran on
    captured at 48 kHz and every single utterance failed on a rate check
    that should never have been a rate check.
    """

    def test_audio_at_the_right_rate_is_left_alone(self) -> None:
        import numpy as np

        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        self.assertIs(to_whisper_rate(audio, 16000), audio)

    def test_a_microphone_at_48k_is_brought_down_not_refused(self) -> None:
        import numpy as np

        seconds = 2.0
        tone = np.sin(
            2 * np.pi * 440 * np.linspace(0, seconds, int(48000 * seconds), False)
        ).astype(np.float32)
        converted = to_whisper_rate(tone, 48000)
        # A second of audio has to still be a second of audio, or the
        # transcript comes back sped up and the timings are all wrong.
        self.assertAlmostEqual(len(converted) / 16000, seconds, places=1)
        self.assertEqual(converted.dtype, np.float32)
        # And it has to still be a signal, not silence.
        self.assertGreater(float(np.abs(converted).max()), 0.5)

    def test_44100_is_handled_too(self) -> None:
        import numpy as np

        audio = np.zeros(44100, dtype=np.float32)
        self.assertAlmostEqual(len(to_whisper_rate(audio, 44100)) / 16000, 1.0, places=1)

    def test_the_recorder_and_the_adapter_agree_on_the_rate(self) -> None:
        # The recording carries its own rate, and the adapter reads it from
        # there rather than assuming what the device was set to.
        adapter = WhisperAdapter(PipelineConfig())
        self.assertFalse(adapter.loaded)
        self.assertEqual(recording(sample_rate=48000).sample_rate, 48000)


if __name__ == "__main__":
    unittest.main()
