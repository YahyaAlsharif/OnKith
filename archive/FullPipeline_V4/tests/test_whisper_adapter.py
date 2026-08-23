"""V4's recogniser, on the seam V3's runtime already called.

Ported from the teammate's ``test_whisper_adapter.py``, which was written
against a Moonshine-era ``RecordedUtterance -> STTResult`` pipeline. The tests
that were about *their* pipeline (the engine-choice flag, the Moonshine
comparison, the wake-word table they were editing at the same time) do not
describe this one and are not carried over; the tests that are about the
recogniser - the hint, the resampler, the counters, the seam - are.

Nothing here loads a model. What can go wrong at this layer is the wiring.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import SegmentTranscript
from integrated_pipeline.whisper_adapter import (
    COMMAND_SHAPES,
    DIGIT_SHAPE,
    WhisperAdapter,
    describe_whisper_backend,
    name_hint,
    to_whisper_rate,
)
from tests.helpers import canary_segment


class ScriptedWhisperModel:
    """A stand-in for ``faster_whisper.WhisperModel``.

    faster-whisper returns ``(segments, info)`` where ``segments`` is a lazy
    generator, so this returns a generator too: the adapter times the decode
    across the loop that drains it, and a list would hide that.
    """

    def __init__(self, *texts: str) -> None:
        self.texts = texts or ("hello world",)
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        self.last_audio = audio

        def stream():
            for text in self.texts:
                yield types.SimpleNamespace(text=text)

        return stream(), types.SimpleNamespace(language="en")


def adapter(*texts: str) -> tuple[WhisperAdapter, ScriptedWhisperModel]:
    model = ScriptedWhisperModel(*texts)
    return (
        WhisperAdapter(
            PipelineConfig(), model_loader=lambda name: model, verify_versions=False
        ),
        model,
    )


class NameHintTests(unittest.TestCase):
    """The hint is built from the names, never written out beside them.

    Otherwise renaming the device in config leaves the recogniser still biased
    toward the old name, which is the failure that is hardest to see:
    everything looks configured and nothing answers.
    """

    def test_the_hint_names_the_device(self) -> None:
        self.assertTrue(
            name_hint(["jarvis"]).startswith("Jarvis is the name of this device.")
        )

    def test_the_hint_shows_a_whole_command_not_just_the_name(self) -> None:
        # Naming the device got the name right and still ran the next word into
        # it, so "play bones" came back as "Playbons" and a request for music
        # routed to conversation. The hint has to show the shape of an
        # utterance, not only the word at the front of it.
        hint = name_hint(["jarvis"])
        for word in ("play", "stop", "show the player", "hide the player"):
            with self.subTest(word=word):
                self.assertIn(word, hint)

    def test_the_commands_in_the_hint_carry_the_configured_name(self) -> None:
        # A hint that demonstrates commands under the old name is worse than no
        # demonstration: it biases every utterance toward a word the device no
        # longer answers to.
        self.assertNotIn("Onkith", name_hint(["jarvis", "sada"]))

    def test_the_hint_shows_the_length_of_a_number(self) -> None:
        # The command words are short and biasing toward them cost a digit out
        # of a phone number, which is the one thing this device is not allowed
        # to round off.
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

    def test_the_balance_the_prompt_was_tuned_at_is_still_here(self) -> None:
        # Six command shapes against one digit sentence is the pair that held
        # 6/6 telephone numbers. Changing either side without re-measuring is
        # the documented way to lose a digit, so the counts are asserted.
        self.assertEqual(len(COMMAND_SHAPES), 7)
        self.assertEqual(DIGIT_SHAPE.count("."), 2)

    def test_the_shipped_names_reach_the_shipped_adapter(self) -> None:
        self.assertIn("Onkith", WhisperAdapter(PipelineConfig()).initial_prompt)

    def test_renaming_the_device_rebiases_the_recogniser(self) -> None:
        config = PipelineConfig()
        config.data["whisper"] = dict(config.whisper, names=["jarvis"])
        self.assertIn("Jarvis", WhisperAdapter(config).initial_prompt)
        self.assertNotIn("Onkith", WhisperAdapter(config).initial_prompt)

    def test_a_written_out_hint_wins(self) -> None:
        config = PipelineConfig()
        config.data["whisper"] = dict(config.whisper, initial_prompt="Say nothing.")
        self.assertEqual(WhisperAdapter(config).initial_prompt, "Say nothing.")


class SampleRateTests(unittest.TestCase):
    """Whisper reads 16 kHz. Microphones do not care about that.

    V4's capture path is pinned to 16 kHz, so in the deployment this is a
    no-op - but a fixture or a machine whose microphone only offers 48 kHz
    still reaches the adapter, and refusing the audio was the wrong answer.
    """

    def test_audio_at_the_right_rate_is_left_alone(self) -> None:
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        self.assertIs(to_whisper_rate(audio, 16000), audio)

    def test_a_microphone_at_48k_is_brought_down_not_refused(self) -> None:
        seconds = 2.0
        tone = np.sin(
            2 * np.pi * 440 * np.linspace(0, seconds, int(48000 * seconds), False)
        ).astype(np.float32)
        converted = to_whisper_rate(tone, 48000)
        # A second of audio has to still be a second of audio, or the transcript
        # comes back sped up and the timings are all wrong.
        self.assertAlmostEqual(len(converted) / 16000, seconds, places=1)
        self.assertEqual(converted.dtype, np.float32)
        # And it has to still be a signal, not silence.
        self.assertGreater(float(np.abs(converted).max()), 0.5)

    def test_44100_is_handled_too(self) -> None:
        audio = np.zeros(44100, dtype=np.float32)
        self.assertAlmostEqual(len(to_whisper_rate(audio, 44100)) / 16000, 1.0, places=1)


class SeamTests(unittest.TestCase):
    """The interface the V3 runtime already called, answered by Whisper."""

    def test_a_segment_in_is_a_segment_transcript_out(self) -> None:
        whisper, model = adapter("hello", "world")
        result = whisper.transcribe_segment(canary_segment(0, final=True))
        self.assertIsInstance(result, SegmentTranscript)
        self.assertEqual(result.text, "hello world")
        self.assertGreaterEqual(result.processing_seconds, 0.0)
        self.assertGreaterEqual(result.queue_wait_seconds, 0.0)
        self.assertTrue(result.stt_started_at_utc.endswith("Z"))
        self.assertTrue(result.stt_completed_at_utc.endswith("Z"))
        # One warmup at load, then one call per segment.
        self.assertEqual(model.calls, 2)
        whisper.close()

    def test_the_decoding_settings_are_the_pinned_ones(self) -> None:
        whisper, model = adapter()
        whisper.transcribe_segment(canary_segment(0, final=True))
        self.assertEqual(model.last_kwargs["beam_size"], 5)
        self.assertEqual(model.last_kwargs["language"], "en")
        self.assertIn("Onkith", model.last_kwargs["initial_prompt"])
        self.assertEqual(whisper.compute_type, "int8")
        self.assertEqual(whisper.device, "cpu")
        self.assertEqual(whisper.model_name, "base.en")
        whisper.close()

    def test_the_model_loads_once_and_warms_up_once(self) -> None:
        whisper, _ = adapter()
        whisper.load()
        whisper.load()
        whisper.transcribe_segment(canary_segment(0, final=True))
        whisper.transcribe_segment(canary_segment(0, final=True, utterance=2))
        self.assertEqual(whisper.counters.model_load_count, 1)
        self.assertEqual(whisper.counters.warmup_count, 1)
        self.assertEqual(whisper.counters.inference_count, 2)
        whisper.close()
        self.assertEqual(whisper.counters.close_count, 1)
        self.assertFalse(whisper.loaded)

    def test_the_runtime_finds_every_attribute_it_reaches_for(self) -> None:
        """The STT worker holds one name for whichever engine is loaded.

        So anything the runtime, the end-of-run audit or ``check_pi`` reaches
        for has to exist here, or the swap fails partway through a session
        rather than at startup where somebody would see it.
        """

        whisper, _ = adapter()
        for attribute in (
            "load",
            "loaded",
            "close",
            "audit",
            "counters",
            "resource_identity",
            "transcribe_segment",
        ):
            with self.subTest(attribute=attribute):
                self.assertTrue(hasattr(whisper, attribute))

    def test_the_audit_names_the_engine_and_the_decoder(self) -> None:
        whisper, _ = adapter()
        whisper.load()
        audit = whisper.audit()
        self.assertEqual(audit["engine"], "whisper")
        self.assertEqual(audit["device"], "cpu")
        self.assertEqual(audit["precision"], "int8")
        self.assertEqual(audit["active_decoder"]["beam_size"], 5)
        self.assertEqual(audit["active_decoder"]["model"], "base.en")
        self.assertIsNotNone(whisper.resource_identity)
        whisper.close()

    def test_a_partial_callback_sees_the_sentence_build_up(self) -> None:
        whisper, _ = adapter("one", "two", "three")
        seen: list[str] = []
        whisper.transcribe_segment(canary_segment(0, final=True), seen.append)
        self.assertEqual(seen, ["one", "one two", "one two three"])
        self.assertEqual(whisper.counters.callback_count, 3)
        whisper.close()

    def test_a_display_fault_never_loses_a_transcript(self) -> None:
        whisper, _ = adapter("one", "two")

        def explode(_text: str) -> None:
            raise RuntimeError("the screen fell over")

        result = whisper.transcribe_segment(canary_segment(0, final=True), explode)
        self.assertEqual(result.text, "one two")
        whisper.close()

    def test_a_closed_adapter_will_not_load_again(self) -> None:
        whisper, _ = adapter()
        whisper.load()
        whisper.close()
        with self.assertRaises(RuntimeError):
            whisper.load()

    def test_the_backend_report_never_raises(self) -> None:
        report = describe_whisper_backend()
        self.assertIn("available", report)
        self.assertIn("faster_whisper_version", report)
        self.assertIsInstance(report["errors"], list)


if __name__ == "__main__":
    unittest.main()
