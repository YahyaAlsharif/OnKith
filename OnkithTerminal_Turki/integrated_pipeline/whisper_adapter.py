"""Whisper through CTranslate2, offered alongside Moonshine.

Moonshine never once heard the device's own name. Over six spoken clips it
wrote Conketh, Honketh, Honkus, Hank if, and on the sixth it dropped the
name out of the sentence altogether. None of those are in the spellings
list, so the device stayed silent and nothing on screen said why. It also
invented digits inside a phone number, which is worse than a mishearing in
a product whose whole claim is that it catches phone numbers.

Whisper takes an initial_prompt, a short piece of text that biases
decoding toward the words in it. Given one sentence naming the device it
got the name right on all six. That is the reason this exists: it treats
the cause rather than adding a fourteenth spelling.

The hint is built from the configured names rather than written out here,
so renaming the device re-biases the recogniser at the same time. A device
that cannot be renamed is one with its name compiled in.

Nothing here pulls torch. faster-whisper runs on CTranslate2, and the only
place torch appears in its metadata is an optional extra for converting
models, which is not installed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable

from .config import PipelineConfig
from .schemas import RecordedUtterance, STTResult

WHISPER_SAMPLE_RATE = 16000
DEFAULT_MODEL = "base.en"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class WhisperCounters:
    """The same tally Moonshine keeps, so the report reads the same.

    Deliberately a separate declaration rather than a shared one. The
    Moonshine adapter is the accepted deployment and is not worth editing
    for a swap that has not yet proven itself on the Pi.
    """

    model_resource_load_count: int = 0
    utterance_session_count: int = 0
    callback_count: int = 0
    queue_overflow_count: int = 0
    stt_processing_seconds: float = 0.0
    finalization_seconds: float = 0.0
    total_stt_seconds: float = 0.0
    total_queue_wait_seconds: float = 0.0


# Whatever a person actually says to this device is one of these shapes.
# Naming the device alone got the name right and still ran the next word
# into it: "Onkith Playbons" is the name heard perfectly and the command
# lost, which routes to conversation and answers a request to play music
# with a sentence. Showing the shape of a whole utterance rather than just
# the name took intent from 83% to 100% on the recorded corpus and the
# word error rate on those clips from 20.7% to 5.0%.
#
# A command the hint does not demonstrate is a command the recogniser is
# not biased toward, so this list and the lanes in onkith_music.py have
# to be added to together.
COMMAND_SHAPES = (
    "{name}, play Bones by Imagine Dragons.",
    "{name}, stop the music.",
    "{name}, show the player.",
    "{name}, hide the player.",
    "{name}, turn the volume up.",
    "{name}, turn it down.",
    "{name}, turn the light on.",
    "{name}, turn the light off.",
)

# A long number is the one thing this device must never round off, and the
# hint above biases toward short command words, which cost a digit:
# 0501234567 came back as 050123457. One example of a full-length number
# restores it. Both values are shaped like the real thing and are not
# anybody's: a phone number in the local format and a ten digit id.
DIGIT_SHAPE = "My number is 0512345678 and my id is 1234567890."


def name_hint(names: Any) -> str:
    """The vocabulary Whisper is weighed toward, built from the config.

    Whisper weighs the words in this when it decodes, which is what turns
    Honketh back into Onkith. It is written from the configured names and
    the commands the device answers to, so renaming the device re-biases
    the recogniser rather than leaving a stale word compiled in here.

    Longer than a prompt wants to be. The limit is real, and a prompt that
    wanders starts pulling the transcript toward its own subject matter,
    so this stays to the two things the device cannot afford to lose: its
    name, and the shape of a number.
    """

    spoken = [str(name).strip() for name in (names or []) if str(name).strip()]
    if not spoken:
        return ""
    titled = [name[:1].upper() + name[1:] for name in spoken]
    if len(titled) == 1:
        opening = f"{titled[0]} is the name of this device."
    else:
        opening = (
            f"{titled[0]} is the name of this device. "
            f"It also answers to {', '.join(titled[1:])}."
        )
    commands = " ".join(shape.format(name=titled[0]) for shape in COMMAND_SHAPES)
    return f"{opening} {commands} {DIGIT_SHAPE}"


def to_whisper_rate(audio: Any, sample_rate: int) -> Any:
    """Bring captured audio down to the only rate Whisper reads.

    Moonshine's SDK takes whatever the microphone runs at, so nothing
    upstream ever had a reason to care. Whisper reads 16 kHz and nothing
    else, and the first machine this ran on captured at 48 kHz, so every
    utterance failed. Refusing the audio was the wrong answer: the rate is
    the device's to choose.

    Resampled through av, which faster-whisper already installs, because
    it filters on the way down. Dropping two samples in three is cheaper
    and folds everything above 8 kHz back into the speech as noise, which
    is exactly the band the recogniser is listening to.
    """

    import numpy as np

    if int(sample_rate) == WHISPER_SAMPLE_RATE:
        return audio
    import av

    resampler = av.AudioResampler(
        format="flt", layout="mono", rate=WHISPER_SAMPLE_RATE
    )
    frame = av.AudioFrame.from_ndarray(
        audio.reshape(1, -1), format="flt", layout="mono"
    )
    frame.sample_rate = int(sample_rate)
    frame.pts = None
    pieces = [
        part.to_ndarray().reshape(-1)
        # The flush matters: the tail of an utterance sits inside the
        # resampler until it is asked for, and the tail is where the last
        # word is.
        for part in (*resampler.resample(frame), *resampler.resample(None))
    ]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


class WhisperAdapter:
    """Load one model and transcribe completed recordings on the caller's thread.

    The surface is the one the STT worker already calls. Nothing else in the
    runtime needs to know which recogniser it is talking to.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        settings = config.stt.get("whisper", {})
        self.model_name = str(settings.get("model", DEFAULT_MODEL))
        self.compute_type = str(
            settings.get("compute_type", DEFAULT_COMPUTE_TYPE)
        )
        self.beam_size = int(settings.get("beam_size", DEFAULT_BEAM_SIZE))
        configured_hint = settings.get("initial_prompt")
        self.initial_prompt = (
            str(configured_hint)
            if configured_hint is not None
            else name_hint(config.wake.get("names"))
        )
        self.counters = WhisperCounters()
        self._model: Any | None = None
        self._resource_identity: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resource_identity(self) -> str | None:
        return self._resource_identity

    def load(self) -> None:
        if self.loaded:
            return
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install faster-whisper to use the whisper engine, or set "
                "stt.engine back to moonshine."
            ) from exc

        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=self.compute_type,
        )
        self.counters.model_resource_load_count += 1
        identity = (
            f"{self.model_name}|{self.compute_type}|{self.beam_size}|"
            f"{self.initial_prompt}"
        )
        self._resource_identity = hashlib.sha256(identity.encode()).hexdigest()

    def transcribe_recording(
        self,
        recording: RecordedUtterance,
        partial_callback: Callable[[str], None] | None = None,
    ) -> STTResult:
        """Transcribe one already-completed recording on the caller's thread."""

        self.load()
        assert self._model is not None

        queue_wait = max(
            0.0, time.monotonic() - recording.recording_closed_monotonic
        )
        total_started = time.perf_counter()
        stt_started_at = _utc_now_iso()

        # The array is read-only by the time it reaches here and the decoder
        # wants to own its input, so it is copied rather than handed over.
        import numpy as np

        audio = np.array(recording.samples, dtype=np.float32, copy=True)
        audio = to_whisper_rate(audio, recording.sample_rate)

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt or None,
        )

        # faster-whisper yields segments as it decodes them, so a caller
        # that wants partials gets the sentence building up rather than
        # nothing at all. It is not the running partial Moonshine gives
        # while somebody is still talking, and it is off by default.
        spoken: list[str] = []
        callback_count = 0
        processing_started = time.perf_counter()
        for segment in segments:
            piece = segment.text.strip()
            if not piece:
                continue
            spoken.append(piece)
            if partial_callback is not None:
                callback_count += 1
                try:
                    partial_callback(" ".join(spoken))
                except Exception:
                    # A display fault must not lose a transcript.
                    pass
        processing_seconds = time.perf_counter() - processing_started

        self.counters.utterance_session_count += 1
        self.counters.callback_count += callback_count
        self.counters.stt_processing_seconds += processing_seconds
        total_stt_seconds = time.perf_counter() - total_started
        self.counters.total_stt_seconds += total_stt_seconds
        self.counters.total_queue_wait_seconds += queue_wait
        submitted_seconds = recording.submitted_audio_seconds

        return replace(
            STTResult(
                final_text=" ".join(spoken).strip(),
                duration_seconds=submitted_seconds,
                processing_seconds=processing_seconds,
                finalization_seconds=0.0,
                rtf=(
                    total_stt_seconds / submitted_seconds
                    if submitted_seconds
                    else float("nan")
                ),
                callback_count=callback_count,
                reliable=recording.reliable,
            ),
            session_id=recording.session_id,
            batch_id=recording.batch_id,
            sequence_number=recording.sequence_number,
            endpoint_reason=recording.endpoint_reason,
            dropped_audio_chunks=recording.dropped_audio_chunks,
            audio_started_monotonic=recording.audio_started_monotonic,
            audio_ended_monotonic=recording.audio_ended_monotonic,
            recording_closed_monotonic=recording.recording_closed_monotonic,
            audio_started_at_utc=recording.audio_started_at_utc,
            audio_ended_at_utc=recording.audio_ended_at_utc,
            stt_started_at_utc=stt_started_at,
            stt_completed_at_utc=_utc_now_iso(),
            captured_audio_seconds=recording.captured_audio_seconds,
            submitted_audio_seconds=submitted_seconds,
            queue_wait_seconds=queue_wait,
            total_stt_seconds=total_stt_seconds,
        )

    def note_queue_overflow(self, count: int = 1) -> None:
        self.counters.queue_overflow_count += int(count)

    def audit(self) -> dict[str, Any]:
        return {
            **vars(self.counters),
            "engine": "whisper",
            "resource_identity": self._resource_identity,
            "model_path": self.model_name,
            "model_arch": self.compute_type,
            "initial_prompt": self.initial_prompt,
        }

    def close(self) -> None:
        self._model = None
