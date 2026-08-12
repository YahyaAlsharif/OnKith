"""A microphone that cannot do 16 kHz is still a usable microphone.

The Pi's mic is a UACDemoV1.0 on hw:2,0, and `arecord --dump-hw-params`
says RATE: 48000 with no range: one fixed hardware rate, not a driver
setting. FullPipeline_V3 required native 16 kHz capture and so could not
open it at all, and a streaming decimator had to be built there to get
around it.

This pipeline never had that requirement. prepare_device() takes the
device's own rate whatever it is, the rate travels with every chunk and
every recording, the segmenter divides by it rather than by a constant,
and the Whisper adapter resamples a finished utterance once, offline,
where filter state and drift do not exist as concepts.

These tests are here because that is now a claim somebody is relying on
rather than a happy accident. They prove the whole path at 48 kHz: a
mic's worth of chunks in, the same words out.
"""

from __future__ import annotations

import sys
import time
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_audio import AudioChunk, LiveSegmentationStateMachine, rms_and_peak
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import RecordedUtterance

CLIPS = ROOT / "eval" / "corpus" / "clips"
# What the Pi's mic actually delivers, and how much of it per callback.
PI_MIC_RATE = 48000
PI_MIC_BLOCK = 4800


def read_clip(name: str):
    import numpy as np

    with wave.open(str(CLIPS / name)) as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype="int16").astype("float32") / 32768.0
    return audio, rate


def at_48k(audio):
    """The same sound as a 48 kHz device would hand it over.

    Sample repetition rather than a real interpolator on purpose: it is
    the crudest possible upsample, so anything that survives it is not
    surviving because the fixture was kind.
    """

    import numpy as np

    return np.repeat(audio, 3).astype("float32")


def recording_of(samples, rate: int) -> RecordedUtterance:
    import numpy as np

    return RecordedUtterance(
        session_id="rate",
        batch_id="rate-1",
        sequence_number=1,
        samples=np.ascontiguousarray(samples, dtype="float32"),
        sample_rate=rate,
        audio_started_monotonic=0.0,
        audio_ended_monotonic=len(samples) / rate,
        audio_started_at_utc="",
        audio_ended_at_utc="",
        captured_at_utc="",
        recording_closed_monotonic=time.monotonic(),
        endpoint_reason="endpoint_silence",
        reliable=True,
        dropped_audio_chunks=0,
        captured_audio_seconds=len(samples) / rate,
    )


class SegmentationAtTheMicsOwnRate(unittest.TestCase):
    def segment(self, audio, rate: int):
        import numpy as np

        emitted: list = []
        machine = LiveSegmentationStateMachine(
            PipelineConfig(),
            "capture-rate",
            on_event=lambda name, data: None,
            on_recording=emitted.append,
            clock=lambda: 100.0,
            utc_now=lambda: "2026-08-12T00:00:00Z",
        )
        block = max(1, round(PI_MIC_BLOCK * rate / PI_MIC_RATE))

        def feed(samples):
            for start in range(0, len(samples), block):
                part = samples[start : start + block]
                if len(part) == 0:
                    continue
                loudness, peak = rms_and_peak(part)
                machine.process(AudioChunk(part, rate, 0.0, loudness, peak))

        feed(np.zeros(rate * 3, dtype="float32"))
        feed(audio)
        feed(np.zeros(rate * 4, dtype="float32"))
        return emitted

    def test_a_48k_microphone_produces_one_recording_at_its_own_rate(self) -> None:
        audio, rate = read_clip("wake-play-artist--turki.wav")
        emitted = self.segment(at_48k(audio), rate * 3)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].sample_rate, PI_MIC_RATE)

    def test_the_duration_is_measured_in_seconds_not_in_samples(self) -> None:
        """The failure this catches is silent and total.

        Divide 48 kHz of samples by a hardcoded 16 kHz and every duration
        is three times too long: calibration, the endpoint silence window
        and the maximum utterance length all stretch, and the device
        stops closing recordings when the person stops talking.
        """

        audio, rate = read_clip("wake-play-artist--turki.wav")
        one = self.segment(audio, rate)
        three = self.segment(at_48k(audio), rate * 3)
        self.assertEqual(len(one), len(three))
        self.assertAlmostEqual(
            one[0].captured_audio_seconds,
            three[0].captured_audio_seconds,
            places=2,
        )


class WhisperAtTheMicsOwnRate(unittest.TestCase):
    """The adapter resamples a finished utterance, so this is the check.

    Offline and whole-array, which is why none of the machinery a
    streaming resampler needs applies here: there is no filter state to
    carry between calls and no phase to drift.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from integrated_pipeline.whisper_adapter import WhisperAdapter
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"faster-whisper is not installed: {error}")
        cls.adapter = WhisperAdapter(PipelineConfig())
        try:
            cls.adapter.load()
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"Whisper could not load: {error}")

    def heard(self, samples, rate: int) -> str:
        return self.adapter.transcribe_recording(recording_of(samples, rate)).final_text.strip()

    def test_the_same_words_come_back_at_either_rate(self) -> None:
        for name in ("wake-play-artist--turki.wav", "wake-stop--turki.wav"):
            with self.subTest(clip=name):
                audio, rate = read_clip(name)
                self.assertEqual(self.heard(audio, rate), self.heard(at_48k(audio), rate * 3))

    def test_a_phone_number_survives_the_resample(self) -> None:
        # The one thing that must not be rounded off, through the one
        # step that could round it off.
        audio, rate = read_clip("pii-phone--turki.wav")
        self.assertIn("0501234567", self.heard(at_48k(audio), rate * 3))


if __name__ == "__main__":
    unittest.main()
