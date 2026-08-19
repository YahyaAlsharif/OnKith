"""Local text to speech, owned by the core process.

Speech lives here rather than in a surface, so it behaves identically
whether the device is showing the kiosk UI, showing nothing, or running
in terminal mode. Every utterance ducks music while it plays and
restores the previous level afterwards, and the restore is guaranteed
even when synthesis fails.

Backends are all local processes. Piper is preferred because it runs on
onnxruntime, matching the rest of the inference path on this device.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Protocol

LogFunction = Callable[[str], None]

_PIPER_MODEL_ENVIRONMENT = "ONKITH_PIPER_MODEL"
_PIPER_SAMPLE_RATE = 22050
_KOKORO_MODEL_ENVIRONMENT = "ONKITH_KOKORO_MODEL"
_KOKORO_VOICES_ENVIRONMENT = "ONKITH_KOKORO_VOICES"
_KOKORO_VOICE_ENVIRONMENT = "ONKITH_KOKORO_VOICE"
_KOKORO_SAMPLE_RATE = 24000
_BACKEND_ENVIRONMENT = "ONKITH_TTS"


class SpeechBackend(Protocol):
    name: str

    def speak(self, text: str) -> None:
        ...


class NullSpeechBackend:
    """Report that no local voice is installed, without silent failure.

    The duck and restore cycle still runs, so the rest of the device is
    correct and demonstrable before a voice is installed on the Pi.
    """

    name = "none"

    def __init__(self, log: LogFunction = lambda message: None) -> None:
        self.log = log
        self._warned = False

    def speak(self, text: str) -> None:
        if not self._warned:
            self._warned = True
            self.log(
                "No local speech backend is installed; install piper or "
                "espeak-ng on the device to hear replies."
            )
        # Approximate the duration of the utterance so the ducking window
        # and the speaking state match what a real voice would produce.
        words = max(1, len(text.split()))
        threading.Event().wait(min(6.0, words / 2.8))


class PiperSpeechBackend:
    """Synthesize with piper and play the raw stream through the sink."""

    name = "piper"

    def __init__(
        self,
        executable: str,
        model: Path,
        player: list[str],
        *,
        sample_rate: int = _PIPER_SAMPLE_RATE,
    ) -> None:
        self.executable = executable
        self.model = model
        self.player = player
        self.sample_rate = sample_rate

    def speak(self, text: str) -> None:
        synthesis = subprocess.Popen(
            [
                self.executable,
                "--model",
                str(self.model),
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        playback = subprocess.Popen(
            self.player,
            stdin=synthesis.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if synthesis.stdout is not None:
            synthesis.stdout.close()
        if synthesis.stdin is not None:
            try:
                synthesis.stdin.write(text.encode("utf-8"))
                synthesis.stdin.close()
            except OSError:
                pass
        playback.wait()
        synthesis.wait()


class KokoroSpeechBackend:
    """Synthesize with kokoro-onnx and play the raw stream through the sink.

    Only the ONNX build of Kokoro is usable here. The plain ``kokoro``
    package pulls a full deep-learning framework onto the device, which
    this runtime does not allow; ``kokoro-onnx`` runs on the same
    onnxruntime as the rest of the inference path.
    """

    name = "kokoro"

    def __init__(
        self,
        model: Path,
        voices: Path,
        player: list[str],
        *,
        voice: str = "af_sarah",
        sample_rate: int = _KOKORO_SAMPLE_RATE,
    ) -> None:
        self.model = model
        self.voices = voices
        self.player = player
        self.voice = voice
        self.sample_rate = sample_rate
        self._engine: Any | None = None

    def _load(self) -> Any:
        if self._engine is None:
            from kokoro_onnx import Kokoro

            self._engine = Kokoro(str(self.model), str(self.voices))
        return self._engine

    def speak(self, text: str) -> None:
        import numpy as np

        samples, sample_rate = self._load().create(
            text, voice=self.voice, speed=1.0, lang="en-us"
        )
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        player = [
            item.replace(str(self.sample_rate), str(int(sample_rate)))
            for item in self.player
        ]
        playback = subprocess.Popen(
            player,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if playback.stdin is not None:
            try:
                playback.stdin.write(pcm)
                playback.stdin.close()
            except OSError:
                pass
        playback.wait()


class CommandSpeechBackend:
    """Speak with a single command that plays audio itself."""

    def __init__(self, name: str, build: Callable[[str], list[str]]) -> None:
        self.name = name
        self._build = build

    def speak(self, text: str) -> None:
        subprocess.run(
            self._build(text),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _raw_player(sample_rate: int) -> list[str] | None:
    paplay = shutil.which("paplay")
    if paplay:
        return [
            paplay,
            "--raw",
            "--format=s16le",
            f"--rate={sample_rate}",
            "--channels=1",
        ]
    pw_play = shutil.which("pw-play")
    if pw_play:
        return [
            pw_play,
            "--format=s16",
            f"--rate={sample_rate}",
            "--channels=1",
            "-",
        ]
    aplay = shutil.which("aplay")
    if aplay:
        return [aplay, "-q", "-f", "S16_LE", "-r", str(sample_rate), "-t", "raw", "-"]
    # Last, and not a new dependency: mpv already plays every note of
    # music this device makes, so if it is absent nothing works anyway.
    # It is here because the three sinks above are all Linux, which left
    # piper and kokoro unselectable on a Mac and made the neural voices
    # impossible to hear before they reached the Pi.
    mpv = shutil.which("mpv")
    if mpv:
        return [
            mpv,
            "--no-video",
            "--really-quiet",
            "--no-terminal",
            "--demuxer=rawaudio",
            "--demuxer-rawaudio-format=s16le",
            f"--demuxer-rawaudio-rate={sample_rate}",
            "--demuxer-rawaudio-channels=1",
            "-",
        ]
    return None


def build_piper_backend() -> SpeechBackend | None:
    piper = shutil.which("piper") or shutil.which("piper-tts")
    model_path = os.environ.get(_PIPER_MODEL_ENVIRONMENT)
    if not piper or not model_path:
        return None
    model = Path(model_path).expanduser()
    player = _raw_player(_PIPER_SAMPLE_RATE)
    if not model.is_file() or player is None:
        return None
    return PiperSpeechBackend(piper, model, player)


def build_kokoro_backend() -> SpeechBackend | None:
    model_path = os.environ.get(_KOKORO_MODEL_ENVIRONMENT)
    voices_path = os.environ.get(_KOKORO_VOICES_ENVIRONMENT)
    if not model_path or not voices_path:
        return None
    model = Path(model_path).expanduser()
    voices = Path(voices_path).expanduser()
    player = _raw_player(_KOKORO_SAMPLE_RATE)
    if not model.is_file() or not voices.is_file() or player is None:
        return None
    return KokoroSpeechBackend(
        model,
        voices,
        player,
        voice=os.environ.get(_KOKORO_VOICE_ENVIRONMENT) or "af_sarah",
    )


def build_espeak_backend() -> SpeechBackend | None:
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        return None
    return CommandSpeechBackend(
        Path(espeak).name,
        lambda text: [espeak, "--", text],
    )


def build_say_backend() -> SpeechBackend | None:
    say = shutil.which("say")
    if not say:
        return None
    return CommandSpeechBackend("say", lambda text: [say, "--", text])


_BACKEND_BUILDERS = {
    "piper": build_piper_backend,
    "kokoro": build_kokoro_backend,
    "espeak": build_espeak_backend,
    "say": build_say_backend,
}
# Kokoro first: it is the voice the device ships with. Piper stays ahead
# of espeak as the lighter fallback for a board that cannot hold Kokoro's
# 310 MB graph beside the recogniser and the classifier, and espeak and
# say are there so a device with neither model still speaks rather than
# going silent.
_AUTOMATIC_ORDER = ("kokoro", "piper", "espeak", "say")


def detect_backend(log: LogFunction = lambda message: None) -> SpeechBackend:
    """Choose the local voice, honouring an explicit ONKITH_TTS choice."""

    requested = (os.environ.get(_BACKEND_ENVIRONMENT) or "").strip().lower()
    if requested == "none":
        return NullSpeechBackend(log)
    if requested:
        builder = _BACKEND_BUILDERS.get(requested)
        if builder is None:
            log(
                f"{_BACKEND_ENVIRONMENT}={requested} is not a known voice; "
                f"choose one of {', '.join(_BACKEND_BUILDERS)} or none."
            )
        else:
            backend = builder()
            if backend is not None:
                return backend
            log(
                f"{_BACKEND_ENVIRONMENT}={requested} was requested but its "
                "model or command is not installed; falling back."
            )

    for name in _AUTOMATIC_ORDER:
        backend = _BACKEND_BUILDERS[name]()
        if backend is not None:
            return backend
    return NullSpeechBackend(log)


class SpeechEngine:
    """Serialise utterances and hold music down for their duration."""

    def __init__(
        self,
        bus: Any,
        *,
        duck: Callable[[], None] | None = None,
        restore: Callable[[], None] | None = None,
        backend: SpeechBackend | None = None,
        log: LogFunction = lambda message: None,
    ) -> None:
        self.bus = bus
        self.log = log
        self._duck = duck
        self._restore = restore
        self._backend = backend if backend is not None else detect_backend(log)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()
        self._ducked = False
        self._shutdown = False

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="onkith-speech-worker",
            daemon=False,
        )
        self._thread.start()

    def speak(self, text: str) -> None:
        message = (text or "").strip()
        if not message or self._shutdown:
            return
        self._idle.clear()
        self._queue.put(message)

    def _apply_duck(self) -> None:
        if self._ducked or self._duck is None:
            return
        try:
            self._duck()
            self._ducked = True
        except Exception as error:
            self.log(f"Music could not be ducked: {error}")

    def _release_duck(self) -> None:
        if not self._ducked or self._restore is None:
            self._ducked = False
            return
        try:
            self._restore()
        except Exception as error:
            self.log(f"Music level could not be restored: {error}")
        finally:
            self._ducked = False

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                # Duck once for a run of utterances rather than flapping the
                # level between back-to-back replies.
                self._apply_duck()
                self.bus.publish("assistant", {"state": "speaking"})
                try:
                    self._backend.speak(text)
                except Exception as error:
                    self.log(f"Speech failed: {error}")
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    # The level is restored even when synthesis raised.
                    self._release_duck()
                    self.bus.publish("assistant", {"state": "idle"})
                    self._idle.set()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self._idle.wait(timeout)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._thread is None:
            self._release_duck()
            return
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._release_duck()
