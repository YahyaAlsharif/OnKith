"""Pinned, serialized faster-whisper ``base.en`` CPU/INT8 inference adapter.

This is V4's speech recognizer, and the one difference between V4 and the
archived V3 pipeline. Everything downstream of it - VAD, segmentation,
transcript merging, the wake-word repair, the inactivity buffer, DeBERTa-v3
masking, span merging, spoken-digit rendering, storage and the UI - is V3's
code, unchanged.

Where the recogniser came from
------------------------------
The decoding configuration here is a teammate's, measured on their own recorded
corpus, and it is reproduced rather than re-derived. Their finding was that the
recogniser they were replacing never once wrote the device's own name: over six
spoken clips it produced Conketh, Honketh, Honkus and "Hank if", and on the
sixth it dropped the name out of the sentence altogether. It also invented
digits inside a telephone number, which is worse than a mishearing in a product
whose whole claim is that it catches telephone numbers.

Whisper takes an ``initial_prompt``, a short piece of text that biases decoding
toward the words in it. Given one sentence naming the device it got the name
right on all six. That is the reason this exists: it treats the cause rather
than adding a fourteenth spelling to the wake-word table.

The hint is built from the configured names rather than written out here, so
renaming the device re-biases the recogniser at the same time. A device that
cannot be renamed is one with its name compiled in.

Nothing here pulls torch. faster-whisper runs on CTranslate2, and the only place
torch appears in its metadata is an optional extra for converting models, which
is not installed.

What this adapter adds to the teammate's version
------------------------------------------------
The teammate's adapter sat on a ``RecordedUtterance -> STTResult`` seam, which
is the shape their pipeline recorded whole utterances in. V4's segmenter hands
the recogniser one ``CanarySegment`` at a time and expects a
``SegmentTranscript`` back, so this presents that interface instead. The
decoding call itself, the prompt, the resampler and the streaming-partial
behaviour are theirs.

The single-inference lock, the warmup and the counter names are V3's, kept so
the runtime, ``--print-config``, ``check_pi.py`` and the end-of-run audit line
read the same as they always did.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig, SAMPLE_RATE
from .schemas import CanarySegment, SegmentTranscript

WHISPER_SAMPLE_RATE = 16_000
DEFAULT_MODEL = "base.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 5
DEFAULT_LANGUAGE = "en"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def describe_whisper_backend() -> dict[str, Any]:
    """Report whether faster-whisper and CTranslate2 are importable here.

    Returns a report rather than raising, the way ``describe_nemo_backend`` did
    in V3, so a benchmark can record exactly what was missing instead of dying
    on an import.
    """

    import importlib.metadata

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    report: dict[str, Any] = {
        "faster_whisper_version": version("faster-whisper"),
        "ctranslate2_version": version("ctranslate2"),
        "av_version": version("av"),
        "available": False,
        "errors": [],
    }
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as exc:
        report["errors"].append(
            f"faster_whisper import failed: {type(exc).__name__}: {exc}"
        )
    try:
        import ctranslate2  # noqa: F401
    except Exception as exc:
        report["errors"].append(
            f"ctranslate2 import failed: {type(exc).__name__}: {exc}"
        )
    report["available"] = not report["errors"]
    return report


# Whatever a person actually says to this device is one of these shapes.
# Naming the device alone got the name right and still ran the next word into
# it: "Onkith Playbons" is the name heard perfectly and the command lost, which
# routes to conversation and answers a request to play music with a sentence.
# Showing the shape of a whole utterance rather than just the name took intent
# from 83% to 100% on the teammate's recorded corpus and the word error rate on
# those clips from 20.7% to 5.0%.
#
# Two of these lanes - the lamp and the volume - do not exist in this pipeline,
# whose interceptor answers PLAY and CLOSE only. They are kept anyway because
# the list was measured as a whole against DIGIT_SHAPE below: six command shapes
# is the count that held 6/6 telephone numbers, and dropping to four changes a
# balance that was tuned rather than guessed. Adding a lane to `onkith_music.py`
# should add its shape here and re-measure the digit side.
COMMAND_SHAPES = (
    "{name}, play Bones by Imagine Dragons.",
    "{name}, stop the music.",
    "{name}, show the player.",
    "{name}, hide the player.",
    "{name}, turn the volume up.",
    "{name}, turn the light off.",
    "{name}, set the light to 40 percent.",
)

# A long number is the one thing this device must never round off, and the hint
# above biases toward short command words, which costs a digit: 0501234567 came
# back as 050123457. A full-length number in the hint restores it.
#
# The two sides have to be kept in balance, and adding to one without
# re-measuring breaks it. Measured on the six corpus clips carrying a verbatim
# number, digits only:
#
#   4 commands, 1 combined number sentence      6/6
#   8 commands, 2 number sentences              5/6
#   6 commands, 2 number sentences              6/6   <- this
#   8 commands, 3 number sentences              6/6
#   8 commands, 2 sentences, unrelated digits   3/6
#
# That last row is the one to be careful about: changing the example numbers to
# unrelated values lost half of them, so part of what makes this work is the
# examples resembling the answer. On a number the corpus has never seen, expect
# worse than 100%.
#
# Both values are shaped like the real thing and are nobody's: a telephone
# number in the local format and a ten digit id.
DIGIT_SHAPE = "My number is 0512345678. My id card number is 1234567890."


def name_hint(names: Any) -> str:
    """The vocabulary Whisper is weighed toward, built from the config.

    Whisper weighs the words in this when it decodes, which is what turns
    Honketh back into Onkith. It is written from the configured names and the
    commands the device answers to, so renaming the device re-biases the
    recogniser rather than leaving a stale word compiled in here.

    Longer than a prompt wants to be. The limit is real, and a prompt that
    wanders starts pulling the transcript toward its own subject matter, so this
    stays to the two things the device cannot afford to lose: its name, and the
    shape of a number.
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

    V4's capture path is pinned to 16 kHz and its segments carry a 16 kHz
    assertion, so in the deployment this is a no-op. It is kept because it is
    the teammate's tested handling and because a fixture or a machine whose
    microphone only offers 48 kHz can still reach this adapter.

    Resampled through av, which faster-whisper already installs, because it
    filters on the way down. Dropping two samples in three is cheaper and folds
    everything above 8 kHz back into the speech as noise, which is exactly the
    band the recogniser is listening to.
    """

    if int(sample_rate) == WHISPER_SAMPLE_RATE:
        return audio
    import av

    resampler = av.AudioResampler(
        format="flt", layout="mono", rate=WHISPER_SAMPLE_RATE
    )
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1),
        format="flt",
        layout="mono",
    )
    frame.sample_rate = int(sample_rate)
    frame.pts = None
    pieces = [
        part.to_ndarray().reshape(-1)
        # The flush matters: the tail of an utterance sits inside the resampler
        # until it is asked for, and the tail is where the last word is.
        for part in (*resampler.resample(frame), *resampler.resample(None))
    ]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


@dataclass
class WhisperCounters:
    """V3's tally, kept name for name so the end-of-run audit line is unchanged.

    ``callback_count`` is the teammate's addition: faster-whisper yields
    segments as it decodes, so a caller that wants partial text gets the
    sentence building up rather than nothing at all.
    """

    model_load_count: int = 0
    warmup_count: int = 0
    inference_count: int = 0
    concurrent_inference_rejections: int = 0
    close_count: int = 0
    callback_count: int = 0
    model_processing_seconds: float = 0.0
    total_queue_wait_seconds: float = 0.0


class WhisperAdapter:
    """Load ``base.en`` once and permit one inference at a time.

    The serialization is V3's rule, not faster-whisper's: a 4 GB Raspberry Pi
    running two decodes at once is a device that swaps, and the segmenter has no
    reason to ever ask for two.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        model_loader: Callable[[str], Any] | None = None,
        verify_versions: bool = True,
    ) -> None:
        self.config = config
        self.model_loader = model_loader
        self.verify_versions = bool(verify_versions)
        settings = config.whisper
        self.model_name = str(settings.get("model", DEFAULT_MODEL))
        self.device = str(settings.get("device", DEFAULT_DEVICE))
        self.compute_type = str(settings.get("compute_type", DEFAULT_COMPUTE_TYPE))
        self.beam_size = int(settings.get("beam_size", DEFAULT_BEAM_SIZE))
        self.language = str(settings.get("language", DEFAULT_LANGUAGE))
        configured_hint = settings.get("initial_prompt")
        self.initial_prompt = (
            str(configured_hint)
            if configured_hint
            else name_hint(settings.get("names"))
        )
        self.counters = WhisperCounters()
        self.model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False
        self._resource_identity: str | None = None
        self._active_decoder: dict[str, Any] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def resource_identity(self) -> str | None:
        return self._resource_identity

    @property
    def active_decoder(self) -> dict[str, Any]:
        return dict(self._active_decoder)

    def _verify_package_versions(self) -> None:
        import importlib.metadata

        required = {"faster-whisper": str(self.config.whisper["package_version"])}
        wrong: list[str] = []
        for distribution, expected in required.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = "not installed"
            if actual != expected:
                wrong.append(f"{distribution}={actual!r} (required {expected!r})")
        if wrong:
            raise RuntimeError("Pinned Whisper backend mismatch:\n" + "\n".join(wrong))

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if self._closed:
                raise RuntimeError("Whisper adapter has been closed")
            if self.verify_versions and self.model_loader is None:
                self._verify_package_versions()
            if self.model_loader is None:
                try:
                    from faster_whisper import WhisperModel
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "faster-whisper is not installed. Install it with "
                        "`pip install --requirement requirements-whisper.txt`."
                    ) from exc
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                self.model = self.model_loader(self.model_name)
            self._active_decoder = {
                "runtime": "faster-whisper/ctranslate2",
                "model": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "beam_size": self.beam_size,
                "language": self.language,
                "initial_prompt_sha256": hashlib.sha256(
                    self.initial_prompt.encode("utf-8")
                ).hexdigest(),
            }
            self._resource_identity = hashlib.sha256(
                json.dumps(self._active_decoder, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self.counters.model_load_count += 1
            self._warmup()

    def _run_model(
        self,
        audio: Any,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        partial_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, float, int]:
        assert self.model is not None
        # The segment's array is read-only and the decoder wants to own its
        # input, so it is copied rather than handed over.
        samples = to_whisper_rate(
            np.array(audio, dtype=np.float32, copy=True), sample_rate
        )
        started = time.perf_counter()
        segments, _info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt or None,
        )
        # faster-whisper is lazy: the decode happens while this loop pulls, so
        # the timer has to stay open across it.
        spoken: list[str] = []
        callbacks = 0
        for segment in segments:
            piece = segment.text.strip()
            if not piece:
                continue
            spoken.append(piece)
            if partial_callback is not None:
                callbacks += 1
                try:
                    partial_callback(" ".join(spoken))
                except Exception:
                    # A display fault must not lose a transcript.
                    pass
        elapsed = time.perf_counter() - started
        return " ".join(spoken).strip(), elapsed, callbacks

    def _warmup(self) -> None:
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Whisper warmup encountered concurrent inference")
        try:
            self._run_model(np.zeros(SAMPLE_RATE, dtype=np.float32))
            self.counters.warmup_count += 1
        finally:
            self._inference_lock.release()

    def transcribe_segment(
        self,
        segment: CanarySegment,
        partial_callback: Callable[[str], None] | None = None,
    ) -> SegmentTranscript:
        """The seam V4's STT worker calls, unchanged from V3's signature."""

        self.load()
        if segment.sample_rate != SAMPLE_RATE:
            raise ValueError("Whisper accepts only exact 16 kHz segment audio")
        queue_wait = max(0.0, time.monotonic() - segment.recording_closed_monotonic)
        if not self._inference_lock.acquire(blocking=False):
            self.counters.concurrent_inference_rejections += 1
            raise RuntimeError("Concurrent Whisper inference is forbidden")
        started_at = _utc_now_iso()
        try:
            text, elapsed, callbacks = self._run_model(
                segment.samples, segment.sample_rate, partial_callback
            )
        finally:
            self._inference_lock.release()
        completed_at = _utc_now_iso()
        self.counters.inference_count += 1
        self.counters.callback_count += callbacks
        self.counters.model_processing_seconds += elapsed
        self.counters.total_queue_wait_seconds += queue_wait
        return SegmentTranscript(
            segment=segment,
            text=text,
            processing_seconds=elapsed,
            queue_wait_seconds=queue_wait,
            stt_started_at_utc=started_at,
            stt_completed_at_utc=completed_at,
        )

    def audit(self) -> dict[str, Any]:
        return {
            **vars(self.counters),
            "engine": "whisper",
            "loaded": self.loaded,
            "resource_identity": self._resource_identity,
            "model_path": self.model_name,
            "active_decoder": dict(self._active_decoder),
            "initial_prompt": self.initial_prompt,
            "device": self.device,
            "precision": self.compute_type,
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._inference_lock.locked():
            raise RuntimeError("Cannot close Whisper during inference")
        self.model = None
        self._closed = True
        self.counters.close_count += 1
