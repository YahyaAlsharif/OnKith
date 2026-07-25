"""Cached Moonshine resources with a fresh mutable stream per utterance."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from moonshine_runtime.moonshine_pi_live import (
    MoonshineRuntimeConfig,
    TranscriptCollector,
    verify_package_version,
)

from .config import MOONSHINE_SESSION_POLICY, PipelineConfig
from .schemas import STTResult


@dataclass
class MoonshineCounters:
    model_resource_load_count: int = 0
    utterance_session_count: int = 0
    callback_count: int = 0
    queue_overflow_count: int = 0
    stt_processing_seconds: float = 0.0
    finalization_seconds: float = 0.0


class MoonshineUtteranceSession:
    """One isolated SDK stream, collector, listener, and callback state."""

    def __init__(
        self,
        adapter: "MoonshineAdapter",
        stream: Any,
        listener: Any,
        collector: TranscriptCollector,
    ):
        self.adapter = adapter
        self.stream = stream
        self.listener = listener
        self.collector = collector
        self.active = False
        self.closed = False
        self.processing_seconds = 0.0
        self.audio_frames = 0
        self.sample_rate: int | None = None

    def start(self) -> None:
        if self.active or self.closed:
            raise RuntimeError("Moonshine utterance session has invalid start state")
        self.stream.start()
        self.active = True

    def add_audio(self, chunk: Any, sample_rate: int) -> None:
        if not self.active or self.closed:
            raise RuntimeError("Moonshine utterance stream is not active")
        import numpy as np

        audio = np.ascontiguousarray(
            np.asarray(chunk, dtype=np.float32).reshape(-1)
        )
        if audio.size == 0:
            return
        if self.sample_rate is not None and self.sample_rate != int(sample_rate):
            raise ValueError("Microphone sample rate changed during an utterance")
        self.sample_rate = int(sample_rate)
        started = time.perf_counter()
        self.stream.add_audio(audio, self.sample_rate)
        self.processing_seconds += time.perf_counter() - started
        self.audio_frames += int(audio.size)

    def finalize(self, *, reliable: bool = True) -> STTResult:
        if self.closed:
            raise RuntimeError("Moonshine utterance session is already closed")
        finalization_started = time.perf_counter()
        try:
            if self.active:
                self.stream.stop()
                self.active = False
            # Callbacks may be delivered immediately after stop returns.
            time.sleep(0.05)
            final_text = self.collector.final_text().strip()
        finally:
            finalization_seconds = time.perf_counter() - finalization_started
            callback_count = self.collector.callback_count
            self.adapter.counters.callback_count += callback_count
            self.adapter.counters.stt_processing_seconds += self.processing_seconds
            self.adapter.counters.finalization_seconds += finalization_seconds
            self.close()

        duration = (
            self.audio_frames / float(self.sample_rate)
            if self.sample_rate and self.audio_frames
            else 0.0
        )
        total_stt = self.processing_seconds + finalization_seconds
        return STTResult(
            final_text=final_text,
            duration_seconds=duration,
            processing_seconds=self.processing_seconds,
            finalization_seconds=finalization_seconds,
            rtf=total_stt / duration if duration else float("nan"),
            callback_count=callback_count,
            reliable=reliable,
        )

    def abort(self) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.active:
                self.stream.stop()
        except Exception:
            pass
        self.active = False
        if hasattr(self.stream, "remove_listener"):
            try:
                self.stream.remove_listener(self.listener)
            except Exception:
                pass
        if hasattr(self.stream, "close"):
            try:
                self.stream.close()
            except Exception:
                pass
        self.collector.clear()
        self.closed = True
        self.adapter._release_session()


class MoonshineAdapter:
    """Load the model once and serialize fresh utterance stream creation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.runtime_config = MoonshineRuntimeConfig.from_json(
            config.moonshine_config_path
        )
        self.runtime_config.validate()
        self.counters = MoonshineCounters()
        self._resource: Any | None = None
        self._listener_class: Any | None = None
        self._model_path: str | None = None
        self._model_arch: Any | None = None
        self._resource_identity: str | None = None
        self._session_lock = threading.Lock()
        self._session_active = False

    @property
    def loaded(self) -> bool:
        return self._resource is not None

    @property
    def resource_identity(self) -> str | None:
        return self._resource_identity

    def load(self) -> None:
        if self.loaded:
            return
        self.config.validate(require_artifacts=True)
        verify_package_version(self.runtime_config.tested_moonshine_voice_version)
        try:
            from moonshine_voice import (
                ModelArch,
                Transcriber,
                TranscriptEventListener,
                get_model_for_language,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install moonshine-voice==0.0.69") from exc

        model_path, model_arch = get_model_for_language(
            self.runtime_config.model_language,
            ModelArch.MEDIUM_STREAMING,
        )
        if model_arch != ModelArch.MEDIUM_STREAMING:
            raise RuntimeError("Moonshine did not return MEDIUM_STREAMING")
        options = {
            "vad_threshold": str(self.runtime_config.vad_threshold),
            "vad_window_duration": str(
                self.runtime_config.vad_window_duration_seconds
            ),
            "vad_max_segment_duration": str(
                self.runtime_config.vad_max_segment_duration_seconds
            ),
            "max_tokens_per_second": str(
                self.runtime_config.max_tokens_per_second
            ),
        }
        self._resource = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self.runtime_config.update_interval_seconds,
            options=options,
        )
        self._listener_class = TranscriptEventListener
        self._model_path = str(model_path)
        self._model_arch = model_arch
        self.counters.model_resource_load_count += 1
        identity = (
            f"{Path(model_path).expanduser().resolve()}|{model_arch!s}|"
            f"{sorted(options.items())}"
        )
        self._resource_identity = hashlib.sha256(identity.encode()).hexdigest()

    def open_utterance(
        self, partial_callback: Callable[[str], None] | None = None
    ) -> MoonshineUtteranceSession:
        """Create one fresh stream; concurrent Moonshine streams are forbidden."""

        self.load()
        if self.config.moonshine_session_policy != MOONSHINE_SESSION_POLICY:
            raise RuntimeError("Only isolated_cached_resources is allowed")
        with self._session_lock:
            if self._session_active:
                raise RuntimeError(
                    "Concurrent Moonshine streams are not enabled or assumed thread-safe"
                )
            self._session_active = True
        assert self._resource is not None
        assert self._listener_class is not None
        collector = TranscriptCollector(partial_callback)

        class Listener(self._listener_class):
            def on_line_started(self, event: Any) -> None:
                collector.update(event)

            def on_line_updated(self, event: Any) -> None:
                collector.update(event)

            def on_line_text_changed(self, event: Any) -> None:
                collector.update(event)

            def on_line_completed(self, event: Any) -> None:
                collector.update(event)

        try:
            stream = self._resource.create_stream(
                update_interval=self.runtime_config.update_interval_seconds
            )
            listener = Listener()
            stream.add_listener(listener)
            session = MoonshineUtteranceSession(
                self, stream, listener, collector
            )
            session.start()
        except Exception:
            with self._session_lock:
                self._session_active = False
            raise
        self.counters.utterance_session_count += 1
        return session

    def _release_session(self) -> None:
        with self._session_lock:
            self._session_active = False

    def note_queue_overflow(self, count: int = 1) -> None:
        self.counters.queue_overflow_count += int(count)

    def audit(self) -> dict[str, Any]:
        return {
            **vars(self.counters),
            "session_policy": MOONSHINE_SESSION_POLICY,
            "resource_identity": self._resource_identity,
            "model_path": self._model_path,
            "model_arch": str(self._model_arch),
        }

    def close(self) -> None:
        if self._resource is not None and hasattr(self._resource, "close"):
            self._resource.close()
        self._resource = None
