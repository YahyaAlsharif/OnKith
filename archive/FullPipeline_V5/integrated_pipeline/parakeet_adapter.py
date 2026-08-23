"""Pinned, serialized NVIDIA Parakeet 110M TDT CPU inference adapter.

This module replaces only V4's former Whisper adapter. It preserves the
``CanarySegment -> SegmentTranscript`` seam consumed by ``LiveSTTRuntime`` and
loads one exact, SHA-256-verified NeMo checkpoint for the whole process.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import (
    PARAKEET_CHECKPOINT_BYTES,
    PARAKEET_CHECKPOINT_FILENAME,
    PARAKEET_CHECKPOINT_SHA256,
    PARAKEET_DECODER_STRATEGY,
    PARAKEET_DECODER_TYPE,
    PARAKEET_DEVICE,
    PARAKEET_GREEDY_MAX_SYMBOLS,
    PARAKEET_MODEL_ID,
    PARAKEET_PACKAGE_VERSION,
    PARAKEET_PRECISION,
    PARAKEET_REVISION,
    PARAKEET_TEMPERATURE,
    PARAKEET_TORCH_VERSION,
    PARAKEET_VERIFIED_PACKAGE_VERSIONS,
    PipelineConfig,
    SAMPLE_RATE,
)
from .schemas import CanarySegment, SegmentTranscript


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def describe_parakeet_backend() -> dict[str, Any]:
    """Report native NeMo/Torch availability without raising."""

    report: dict[str, Any] = {
        "nemo_toolkit_version": _distribution_version("nemo_toolkit"),
        "torch_version": _distribution_version("torch"),
        "huggingface_hub_version": _distribution_version("huggingface_hub"),
        "available": False,
        "errors": [],
    }
    try:
        import torch

        report["torch_cpu_available"] = True
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        report["errors"].append(f"torch import failed: {type(exc).__name__}: {exc}")
    try:
        import nemo.collections.asr  # noqa: F401
    except Exception as exc:
        report["errors"].append(
            f"nemo.collections.asr import failed: {type(exc).__name__}: {exc}"
        )
    report["available"] = not report["errors"]
    return report


def cache_parakeet_checkpoint(*, local_files_only: bool = False) -> dict[str, Any]:
    """Download or resolve the exact measured checkpoint and verify its bytes."""

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve the pinned Parakeet checkpoint"
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=PARAKEET_MODEL_ID,
                revision=PARAKEET_REVISION,
                allow_patterns=[PARAKEET_CHECKPOINT_FILENAME],
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "cached" if local_files_only else "downloaded"
        raise RuntimeError(
            f"The exact Parakeet checkpoint could not be {mode}. "
            "Run the platform setup script while online first."
        ) from exc
    matches = sorted(snapshot.rglob(PARAKEET_CHECKPOINT_FILENAME))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one pinned Parakeet .nemo checkpoint; "
            f"found {len(matches)} under {snapshot}"
        )
    checkpoint = matches[0].resolve()
    actual_size = checkpoint.stat().st_size
    if actual_size != PARAKEET_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"Parakeet checkpoint size mismatch: {actual_size}; "
            f"expected {PARAKEET_CHECKPOINT_BYTES}"
        )
    actual_hash = _sha256_file(checkpoint)
    if actual_hash != PARAKEET_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Parakeet checkpoint SHA-256 mismatch: {actual_hash}; "
            f"expected {PARAKEET_CHECKPOINT_SHA256}"
        )
    return {
        "model_id": PARAKEET_MODEL_ID,
        "revision": PARAKEET_REVISION,
        "path": str(checkpoint),
        "filename": PARAKEET_CHECKPOINT_FILENAME,
        "sha256": actual_hash,
        "size_bytes": actual_size,
    }


def _get(node: Any, name: str, default: Any = None) -> Any:
    if node is None:
        return default
    try:
        return node.get(name, default)
    except Exception:
        return getattr(node, name, default)


def _has(node: Any, name: str) -> bool:
    if node is None:
        return False
    try:
        return name in node
    except Exception:
        return hasattr(node, name)


def _set(node: Any, name: str, value: Any) -> None:
    try:
        setattr(node, name, value)
    except Exception:
        node[name] = value


def _as_dict(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(config, resolve=True))
    except Exception:
        try:
            return json.loads(json.dumps(config, default=str))
        except Exception:
            return {"repr": repr(config)}


@dataclass
class ParakeetCounters:
    model_load_count: int = 0
    warmup_count: int = 0
    inference_count: int = 0
    concurrent_inference_rejections: int = 0
    close_count: int = 0
    callback_count: int = 0
    model_processing_seconds: float = 0.0
    total_queue_wait_seconds: float = 0.0


class ParakeetAdapter:
    """Load Parakeet once and expose V4's existing segment transcript contract."""

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
        settings = config.parakeet
        self.model_id = str(settings["model_id"])
        self.device = str(settings.get("device", PARAKEET_DEVICE))
        self.precision = str(settings.get("precision", PARAKEET_PRECISION))
        self.decoder_type = str(settings.get("decoder_type", PARAKEET_DECODER_TYPE))
        self.decoder_strategy = str(
            settings.get("decoder_strategy", PARAKEET_DECODER_STRATEGY)
        )
        self.greedy_max_symbols = int(
            settings.get("greedy_max_symbols", PARAKEET_GREEDY_MAX_SYMBOLS)
        )
        self.temperature = float(settings.get("temperature", PARAKEET_TEMPERATURE))
        self.counters = ParakeetCounters()
        self.model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False
        self._resource_identity: str | None = None
        self._active_decoder: dict[str, Any] = {}
        self._checkpoint: dict[str, Any] | None = None

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
        """Refuse a NeMo/torch pair this build has never been proven on.

        ``PARAKEET_VERIFIED_PACKAGE_VERSIONS`` is a short allowlist rather than a
        single pin because the version the platform setup scripts install and the
        interpreter that produced a given measurement are not always the same.
        An unlisted release is still refused here, the checkpoint bytes are still
        verified against the pinned SHA-256, and the live model still has to read
        back as TDT in :meth:`_assert_tdt`, so nothing load-bearing was relaxed.
        :meth:`audit` records the version that actually ran, so no result folder
        is ambiguous about which one produced it.
        """

        required: dict[str, tuple[str, ...]] = {
            "nemo_toolkit": tuple(PARAKEET_VERIFIED_PACKAGE_VERSIONS),
            "torch": (PARAKEET_TORCH_VERSION,),
        }
        wrong = [
            f"{name}={_distribution_version(name)!r} "
            f"(verified: {', '.join(accepted)})"
            for name, accepted in required.items()
            if _distribution_version(name) not in accepted
        ]
        if wrong:
            raise RuntimeError("Pinned Parakeet backend mismatch:\n" + "\n".join(wrong))

    def _extract_loss_name(self) -> str | None:
        assert self.model is not None
        if hasattr(self.model, "extract_rnnt_loss_cfg"):
            try:
                name, _kwargs = self.model.extract_rnnt_loss_cfg(
                    _get(self.model.cfg, "loss")
                )
                return str(name).lower()
            except Exception:
                pass
        loss = _get(self.model.cfg, "loss", {})
        name = _get(loss, "loss_name", _get(loss, "name"))
        return str(name).lower() if name is not None else None

    def _extract_durations(self) -> list[int]:
        assert self.model is not None
        candidates = []
        loss = _get(self.model.cfg, "loss", {})
        loss_kwargs = _get(loss, "tdt_kwargs", _get(loss, "loss_kwargs", {}))
        for node in (loss_kwargs, _get(self.model.cfg, "decoding"), self.model.decoding):
            for key in ("durations", "tdt_durations"):
                value = _get(node, key)
                if value is not None:
                    candidates.append(value)
        for value in candidates:
            try:
                return [int(item) for item in list(value)]
            except Exception:
                continue
        raise RuntimeError("TDT duration classes are unavailable in the checkpoint")

    def _configure_tdt_decoder(self) -> None:
        assert self.model is not None
        if not hasattr(self.model, "change_decoding_strategy"):
            raise RuntimeError("Loaded Parakeet model cannot change decoding strategy")
        decoding = copy.deepcopy(_get(self.model.cfg, "decoding"))
        if decoding is None:
            raise RuntimeError("Loaded Parakeet model has no decoding configuration")
        try:
            from omegaconf import OmegaConf, open_dict

            editable = open_dict(decoding) if OmegaConf.is_config(decoding) else contextlib.nullcontext()
        except Exception:
            editable = contextlib.nullcontext()
        with editable:
            _set(decoding, "strategy", self.decoder_strategy)
            _set(decoding, "temperature", self.temperature)
            _set(decoding, "compute_timestamps", False)
            _set(decoding, "preserve_alignments", False)
            greedy = _get(decoding, "greedy")
            if greedy is None:
                raise RuntimeError("Parakeet decoder exposes no greedy configuration")
            max_fields = [
                field
                for field in ("max_symbols", "max_symbols_per_step")
                if _has(greedy, field)
            ]
            if not max_fields:
                raise RuntimeError("Parakeet greedy decoder exposes no max-symbol field")
            for field in max_fields:
                _set(greedy, field, self.greedy_max_symbols)
            _set(greedy, "preserve_alignments", False)
            if _has(greedy, "compute_timestamps"):
                _set(greedy, "compute_timestamps", False)
            if _has(greedy, "use_cuda_graph_decoder"):
                _set(greedy, "use_cuda_graph_decoder", False)
            if _has(greedy, "allow_cuda_graphs"):
                _set(greedy, "allow_cuda_graphs", False)
            if _has(greedy, "ngram_lm_model"):
                _set(greedy, "ngram_lm_model", None)
        self.model.change_decoding_strategy(
            decoding_cfg=decoding,
            decoder_type=self.decoder_type,
            verbose=False,
        )
        self._active_decoder = self._assert_tdt()

    def _assert_tdt(self) -> dict[str, Any]:
        assert self.model is not None
        branch = str(getattr(self.model, "cur_decoder", "")).lower()
        if branch != "rnnt":
            raise RuntimeError(f"Parakeet TDT branch is not active: {branch!r}")
        loss_name = self._extract_loss_name()
        if loss_name != "tdt":
            raise RuntimeError(f"Active RNNT branch is not TDT-trained: {loss_name!r}")
        wrapper = getattr(self.model, "decoding", None)
        implementation = getattr(wrapper, "decoding", None)
        implementation_name = type(implementation).__name__
        if "TDT" not in implementation_name.upper():
            raise RuntimeError(
                "Active decoder implementation is not TDT: " + implementation_name
            )
        decoding = _get(self.model.cfg, "decoding")
        greedy = _get(decoding, "greedy")
        strategy = str(_get(decoding, "strategy", ""))
        if strategy != PARAKEET_DECODER_STRATEGY:
            raise RuntimeError(f"Active decoder strategy is {strategy!r}, not greedy_batch")
        max_symbols = _get(greedy, "max_symbols", _get(greedy, "max_symbols_per_step"))
        if int(max_symbols) != PARAKEET_GREEDY_MAX_SYMBOLS:
            raise RuntimeError(f"Active greedy max symbols is {max_symbols!r}, not 10")
        temperature = float(_get(decoding, "temperature"))
        if temperature != PARAKEET_TEMPERATURE:
            raise RuntimeError(f"Active decoder temperature is {temperature!r}, not 1.0")
        if bool(_get(decoding, "compute_timestamps", False)):
            raise RuntimeError("Parakeet timestamps unexpectedly enabled")
        if bool(_get(decoding, "preserve_alignments", False)) or bool(
            _get(greedy, "preserve_alignments", False)
        ):
            raise RuntimeError("Parakeet alignments unexpectedly enabled")
        return {
            "decoder_family": "TDT",
            "decoder_type": "rnnt",
            "strategy": strategy,
            "greedy_max_symbols": int(max_symbols),
            "temperature": temperature,
            "timestamps": False,
            "preserve_alignments": False,
            "implementation": implementation_name,
            "loss_name": loss_name,
            "tdt_duration_values": self._extract_durations(),
        }

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if self._closed:
                raise RuntimeError("Parakeet adapter has been closed")
            if self.verify_versions and self.model_loader is None:
                self._verify_package_versions()
            if self.model_loader is None:
                self._checkpoint = cache_parakeet_checkpoint(local_files_only=True)
                os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
                try:
                    import torch
                    import nemo.collections.asr as nemo_asr
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Parakeet requires torch and nemo_toolkit[asr]. "
                        "Run the platform setup script first."
                    ) from exc
                self.model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=self._checkpoint["path"],
                    map_location=torch.device("cpu"),
                    strict=True,
                )
                self.model.to(torch.device("cpu"))
                self.model.float()
            else:
                self.model = self.model_loader(self.model_id)
                self._checkpoint = {
                    "model_id": self.model_id,
                    "revision": PARAKEET_REVISION,
                    "path": "injected-test-model",
                    "filename": PARAKEET_CHECKPOINT_FILENAME,
                    "sha256": PARAKEET_CHECKPOINT_SHA256,
                    "size_bytes": PARAKEET_CHECKPOINT_BYTES,
                }
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(self.model, "freeze"):
                self.model.freeze()
            self.counters.model_load_count += 1
            self._configure_tdt_decoder()
            identity = {
                "model_id": self.model_id,
                "revision": PARAKEET_REVISION,
                "checkpoint_sha256": PARAKEET_CHECKPOINT_SHA256,
                "device": self.device,
                "precision": self.precision,
                "decoder": self._active_decoder,
            }
            self._resource_identity = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self._warmup()

    def _inference_context(self) -> Any:
        try:
            import torch
        except ImportError:
            if self.model_loader is None:
                raise
            return contextlib.nullcontext()
        return torch.inference_mode()

    def _transcribe_kwargs(self) -> dict[str, Any]:
        assert self.model is not None
        signature = inspect.signature(self.model.transcribe)
        supported = signature.parameters
        settings = {
            "batch_size": 1,
            "return_hypotheses": True,
            "num_workers": 0,
            "verbose": False,
            "timestamps": False,
            "use_lhotse": False,
            "channel_selector": None,
            "augmentor": None,
            "pad_min_duration": 0.0,
            "pad_direction": "right",
        }
        if "override_config" in supported and hasattr(self.model, "get_transcribe_config"):
            override = self.model.get_transcribe_config()
            for name, value in settings.items():
                _set(override, name, value)
            return {"override_config": override}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in supported.values()
        )
        if accepts_kwargs:
            return settings
        return {name: value for name, value in settings.items() if name in supported}

    @staticmethod
    def _extract_text(output: Any) -> str:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, list):
            if not output:
                return ""
            output = output[0]
        if hasattr(output, "text"):
            return str(output.text).strip()
        if isinstance(output, dict) and "text" in output:
            return str(output["text"]).strip()
        return str(output).strip()

    def _run_model(
        self,
        audio: Any,
        partial_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, float, int]:
        assert self.model is not None
        samples = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not samples.size or not np.all(np.isfinite(samples)):
            raise ValueError("Parakeet requires a non-empty finite mono segment")
        started = time.perf_counter()
        with self._inference_context():
            output = self.model.transcribe(samples, **self._transcribe_kwargs())
        elapsed = time.perf_counter() - started
        text = self._extract_text(output)
        callbacks = 0
        if text and partial_callback is not None:
            callbacks = 1
            try:
                partial_callback(text)
            except Exception:
                pass
        return text, elapsed, callbacks

    def _warmup(self) -> None:
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Parakeet warmup encountered concurrent inference")
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
        """Return the same segment transcript object the existing runtime expects."""

        self.load()
        if segment.sample_rate != SAMPLE_RATE:
            raise ValueError("Parakeet accepts only exact 16 kHz segment audio")
        queue_wait = max(0.0, time.monotonic() - segment.recording_closed_monotonic)
        if not self._inference_lock.acquire(blocking=False):
            self.counters.concurrent_inference_rejections += 1
            raise RuntimeError("Concurrent Parakeet inference is forbidden")
        started_at = _utc_now_iso()
        try:
            text, elapsed, callbacks = self._run_model(
                segment.samples, partial_callback
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
            "engine": "parakeet",
            "loaded": self.loaded,
            "resource_identity": self._resource_identity,
            "model_id": self.model_id,
            "revision": PARAKEET_REVISION,
            "checkpoint_sha256": PARAKEET_CHECKPOINT_SHA256,
            "checkpoint_path": None if self._checkpoint is None else self._checkpoint["path"],
            "active_decoder": dict(self._active_decoder),
            "device": self.device,
            "precision": self.precision,
            "nemo_toolkit_version": _distribution_version("nemo_toolkit"),
            "torch_version": _distribution_version("torch"),
            "pinned_nemo_toolkit_version": PARAKEET_PACKAGE_VERSION,
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._inference_lock.locked():
            raise RuntimeError("Cannot close Parakeet during inference")
        self.model = None
        self._closed = True
        self.counters.close_count += 1
        gc.collect()

