"""Pinned, serialized NeMo Canary 180M Flash CPU/FP32 inference adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig, SAMPLE_RATE
from .schemas import CanarySegment, SegmentTranscript


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _cfg_value(config: Any, dotted: str) -> Any:
    value = config
    for part in dotted.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    return value


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def describe_nemo_backend() -> dict[str, Any]:
    """Introspect the installed NeMo/torch API this build depends on.

    Returns a report rather than raising, so callers can record exactly which
    parts of the contract are verifiable in the current environment. NeMo 2.5.3
    and the 2.7.3 benchmark reference are not assumed to be identical: the
    accepted keyword names are read from the live signatures.
    """

    report: dict[str, Any] = {
        "nemo_version": _installed_version("nemo-toolkit"),
        "torch_version": _installed_version("torch"),
        "available": False,
        "errors": [],
    }
    try:
        import torch  # noqa: F401
    except Exception as exc:
        report["errors"].append(f"torch import failed: {type(exc).__name__}: {exc}")
    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel
        from nemo.collections.asr.models.aed_multitask_models import (
            MultiTaskTranscriptionConfig,
        )
    except Exception as exc:
        report["errors"].append(f"nemo import failed: {type(exc).__name__}: {exc}")
        return report

    def parameters(target: Any) -> list[str]:
        try:
            return sorted(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            return []

    report["available"] = not report["errors"]
    report["restore_from_parameters"] = parameters(EncDecMultiTaskModel.restore_from)
    report["change_decoding_strategy_parameters"] = parameters(
        EncDecMultiTaskModel.change_decoding_strategy
    )
    report["transcribe_parameters"] = parameters(EncDecMultiTaskModel.transcribe)
    try:
        report["transcription_config_fields"] = sorted(
            getattr(MultiTaskTranscriptionConfig, "__dataclass_fields__", {})
        )
    except Exception as exc:  # pragma: no cover - defensive
        report["errors"].append(f"config introspection failed: {exc}")
    required = {
        "restore_from_parameters": {"restore_path", "map_location", "strict"},
        "transcribe_parameters": {"audio", "override_config"},
    }
    missing = {
        name: sorted(expected - set(report.get(name, [])))
        for name, expected in required.items()
        if not expected.issubset(set(report.get(name, [])))
    }
    report["missing_parameters"] = missing
    report["signature_contract_satisfied"] = not missing
    return report


@dataclass
class CanaryCounters:
    model_load_count: int = 0
    warmup_count: int = 0
    inference_count: int = 0
    concurrent_inference_rejections: int = 0
    close_count: int = 0
    model_processing_seconds: float = 0.0
    total_queue_wait_seconds: float = 0.0


class CanaryAdapter:
    """Load the exact local .nemo once and permit one inference at a time."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        model_loader: Callable[[Path], Any] | None = None,
        transcription_config_factory: Callable[..., Any] | None = None,
        verify_artifacts: bool = True,
        verify_versions: bool = True,
    ) -> None:
        self.config = config
        self.model_loader = model_loader
        self.transcription_config_factory = transcription_config_factory
        self.verify_artifacts = bool(verify_artifacts)
        self.verify_versions = bool(verify_versions)
        self.model: Any | None = None
        self.transcription_config: Any | None = None
        self.counters = CanaryCounters()
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False
        self._resource_identity: str | None = None
        self._active_decoder: dict[str, Any] = {}
        self._active_prompt: list[dict[str, Any]] = []
        self._inference_mode_active: bool | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def resource_identity(self) -> str | None:
        return self._resource_identity

    def verify_model_artifact(self) -> dict[str, Any]:
        self.config.validate(require_stt_artifacts=True)
        path = self.config.canary_model_path
        manifest_path = self.config.canary_manifest_path
        actual_hash = _sha256(path)
        expected_hash = str(self.config.canary["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Canary artifact SHA-256 mismatch: {actual_hash}; expected {expected_hash}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "model_id": self.config.canary["model_id"],
            "revision": self.config.canary["revision"],
            "artifact_file": self.config.canary["artifact_file"],
            "sha256": expected_hash,
            "backend": self.config.canary["backend"],
            "precision": "fp32",
            "nemo_version": self.config.canary["nemo_version"],
            "torch_version": self.config.canary["torch_version"],
        }
        mismatches = [
            f"{key}={manifest.get(key)!r} (required {value!r})"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError("Canary manifest mismatch:\n" + "\n".join(mismatches))
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
            "manifest": manifest,
        }

    def _verify_package_versions(self) -> None:
        required = {
            "nemo-toolkit": str(self.config.canary["nemo_version"]),
            "torch": str(self.config.canary["torch_version"]),
        }
        wrong: list[str] = []
        for distribution, expected in required.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual = "not installed"
            if actual != expected:
                wrong.append(f"{distribution}={actual!r} (required {expected!r})")
        if wrong:
            raise RuntimeError("Pinned Canary backend mismatch:\n" + "\n".join(wrong))

    # The exact multitask prompt shape this build sends to Canary. NeMo 2.5.3
    # and 2.7.3 both accept this single user turn with slot values as strings;
    # it is verified after construction so a version change cannot silently
    # alter the task, language, punctuation or timestamp request.
    REQUIRED_PROMPT_SLOTS = {
        "source_lang": "en",
        "target_lang": "en",
        "task": "asr",
        "pnc": "no",
        "timestamp": "no",
    }

    def _verify_prompt(self, prompt: Any) -> None:
        # NeMo accepts a single-turn prompt as a dict and normalises it into a
        # list of turns, so accept either shape here and verify the slots that
        # actually reached the model. The slot values themselves are fixed.
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise RuntimeError("Canary prompt must be exactly one user turn")
            turn = prompt[0]
        elif isinstance(prompt, dict) and "turns" in prompt:
            turns = prompt["turns"]
            if not isinstance(turns, list) or len(turns) != 1:
                raise RuntimeError("Canary prompt must be exactly one user turn")
            turn = turns[0]
        else:
            turn = prompt
        if not isinstance(turn, dict) or turn.get("role") != "user":
            raise RuntimeError("Canary prompt turn must have role 'user'")
        slots = turn.get("slots")
        if not isinstance(slots, dict):
            raise RuntimeError("Canary prompt turn must carry a slots mapping")
        wrong = [
            f"{name}={slots.get(name)!r} (required {expected!r})"
            for name, expected in self.REQUIRED_PROMPT_SLOTS.items()
            if slots.get(name) != expected
        ]
        extra = sorted(set(slots) - set(self.REQUIRED_PROMPT_SLOTS))
        if extra:
            wrong.append(f"unexpected prompt slots: {extra}")
        if wrong:
            raise RuntimeError("Canary prompt mismatch:\n" + "\n".join(wrong))
        self._active_prompt = [
            {"role": "user", "slots": dict(self.REQUIRED_PROMPT_SLOTS)}
        ]

    def _inference_context(self) -> Any:
        """Always run real inference under ``torch.inference_mode()``.

        Mock-backed tests inject ``model_loader`` and may run in environments
        with no torch installed; the real backend cannot, so a missing torch is
        an error there rather than a silent downgrade.
        """

        try:
            import torch
        except ImportError:
            if self.model_loader is None:
                raise
            self._inference_mode_active = False
            return nullcontext()
        self._inference_mode_active = True
        return torch.inference_mode()

    def _make_transcription_config(self) -> Any:
        # Single-turn prompt in the shape NeMo's parse_multitask_prompt accepts:
        # it indexes the prompt with .items(), so it must be a mapping. Passing
        # a list (the parser's *output* shape) raises
        # "AttributeError: 'list' object has no attribute 'items'" on NeMo
        # 2.5.3. The task, languages, punctuation and timestamp values are
        # unchanged; only the container differs.
        prompt = {
            "role": "user",
            "slots": dict(self.REQUIRED_PROMPT_SLOTS),
        }
        kwargs = {
            "batch_size": 1,
            "return_hypotheses": True,
            "num_workers": 0,
            "verbose": False,
            "timestamps": False,
            "prompt": prompt,
            "enable_chunking": False,
        }
        if self.transcription_config_factory is not None:
            result = self.transcription_config_factory(**kwargs)
        else:
            from nemo.collections.asr.models.aed_multitask_models import (
                MultiTaskTranscriptionConfig,
            )

            result = MultiTaskTranscriptionConfig(**kwargs)
        for name, expected in {
            "batch_size": 1,
            "return_hypotheses": True,
            "num_workers": 0,
            "timestamps": False,
            "enable_chunking": False,
        }.items():
            actual = getattr(result, name)
            if actual != expected:
                raise RuntimeError(
                    f"Active transcription config {name}={actual!r}; required {expected!r}"
                )
        self._verify_prompt(getattr(result, "prompt", None))
        return result

    def _configure_and_verify_decoder(self) -> None:
        assert self.model is not None
        if isinstance(self.model.cfg.decoding, dict):
            decoding = copy.deepcopy(self.model.cfg.decoding)
            decoding["strategy"] = "beam"
            decoding["temperature"] = 1.0
            decoding.setdefault("beam", {})["beam_size"] = 2
            decoding["beam"]["return_best_hypothesis"] = True
            decoding["beam"]["max_generation_delta"] = 50
        else:
            from omegaconf import OmegaConf, open_dict

            decoding = OmegaConf.create(
                OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
            )
            with open_dict(decoding):
                decoding.strategy = "beam"
                decoding.temperature = 1.0
                decoding.beam.beam_size = 2
                decoding.beam.return_best_hypothesis = True
                decoding.beam.max_generation_delta = 50
        self.model.change_decoding_strategy(decoding)

        checks = {
            "strategy": "beam",
            "temperature": 1.0,
            "beam.beam_size": 2,
            "beam.return_best_hypothesis": True,
            "beam.max_generation_delta": 50,
        }
        wrong = []
        for location, active_cfg in (
            ("model.cfg.decoding", self.model.cfg.decoding),
            ("model.decoding.cfg", self.model.decoding.cfg),
        ):
            for name, expected in checks.items():
                actual = _cfg_value(active_cfg, name)
                if actual != expected:
                    wrong.append(
                        f"{location}.{name}={actual!r} (required {expected!r})"
                    )

        engine = self.model.decoding.decoding
        engine_checks = {
            "engine.class": (engine.__class__.__name__, "TransformerAEDBeamInfer"),
            "engine.beam_size": (getattr(engine, "beam_size", None), 2),
            "engine.return_best_hypothesis": (
                getattr(engine, "return_best_hypothesis", None),
                True,
            ),
            "engine.max_generation_delta": (
                getattr(getattr(engine, "beam_search", None), "max_delta_len", None),
                50,
            ),
        }
        wrong.extend(
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in engine_checks.items()
            if actual != expected
        )
        if wrong:
            raise RuntimeError("Active Canary decoder mismatch:\n" + "\n".join(wrong))
        self._active_decoder = {
            "strategy": "beam",
            "beam_size": 2,
            "return_best_hypothesis": True,
            "temperature": 1.0,
            "max_generation_delta": 50,
            "engine": engine.__class__.__name__,
        }

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if self._closed:
                raise RuntimeError("Canary adapter has been closed")
            artifact = self.verify_model_artifact() if self.verify_artifacts else None
            if self.verify_versions:
                self._verify_package_versions()
            if self.model_loader is None:
                import torch
                from nemo.collections.asr.models import EncDecMultiTaskModel

                self.model = EncDecMultiTaskModel.restore_from(
                    restore_path=str(self.config.canary_model_path),
                    map_location=torch.device("cpu"),
                    strict=True,
                )
                self.model.to(torch.device("cpu"))
                self.model.float()
                self.model.eval()
            else:
                self.model = self.model_loader(self.config.canary_model_path)
            self._configure_and_verify_decoder()
            self.transcription_config = self._make_transcription_config()
            identity_source = {
                "model_id": self.config.canary["model_id"],
                "revision": self.config.canary["revision"],
                "sha256": self.config.canary["sha256"],
                "backend": self.config.canary["backend"],
                "decoder": self._active_decoder,
                "artifact_size": None if artifact is None else artifact["size_bytes"],
            }
            self._resource_identity = hashlib.sha256(
                json.dumps(identity_source, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self.counters.model_load_count += 1
            self._warmup()

    @staticmethod
    def _extract_one_text(outputs: Any) -> str:
        if isinstance(outputs, tuple):
            raise RuntimeError("Canary returned an unexpected tuple of output streams")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("Canary must return exactly one best hypothesis")
        hypothesis = outputs[0]
        if isinstance(hypothesis, (list, tuple)):
            raise RuntimeError("Canary returned N-best hypotheses despite return_best=true")
        if isinstance(hypothesis, str):
            return hypothesis.strip()
        text = getattr(hypothesis, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("Canary hypothesis has no text string")
        return text.strip()

    def _run_model(self, audio: np.ndarray) -> tuple[str, float]:
        assert self.model is not None and self.transcription_config is not None
        started = time.perf_counter()
        with self._inference_context():
            outputs = self.model.transcribe(
                audio=[np.ascontiguousarray(audio, dtype=np.float32)],
                override_config=self.transcription_config,
            )
        elapsed = time.perf_counter() - started
        return self._extract_one_text(outputs), elapsed

    def _warmup(self) -> None:
        acquired = self._inference_lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("Canary warmup encountered concurrent inference")
        try:
            self._run_model(np.zeros(SAMPLE_RATE, dtype=np.float32))
            self.counters.warmup_count += 1
        finally:
            self._inference_lock.release()

    def transcribe_segment(self, segment: CanarySegment) -> SegmentTranscript:
        self.load()
        if segment.sample_rate != SAMPLE_RATE:
            raise ValueError("Canary accepts only exact 16 kHz segment audio")
        queue_wait = max(0.0, time.monotonic() - segment.recording_closed_monotonic)
        if not self._inference_lock.acquire(blocking=False):
            self.counters.concurrent_inference_rejections += 1
            raise RuntimeError("Concurrent Canary inference is forbidden")
        started_at = _utc_now_iso()
        try:
            text, elapsed = self._run_model(segment.samples)
        finally:
            self._inference_lock.release()
        completed_at = _utc_now_iso()
        self.counters.inference_count += 1
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
            "loaded": self.loaded,
            "resource_identity": self._resource_identity,
            "model_path": str(self.config.canary_model_path),
            "active_decoder": dict(self._active_decoder),
            "active_prompt": [dict(item) for item in self._active_prompt],
            "inference_mode_active": self._inference_mode_active,
            "device": "cpu",
            "precision": "fp32",
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._inference_lock.locked():
            raise RuntimeError("Cannot close Canary during inference")
        self.model = None
        self.transcription_config = None
        self._closed = True
        self.counters.close_count += 1
