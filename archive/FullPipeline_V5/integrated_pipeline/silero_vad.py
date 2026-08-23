"""Pinned stateful Silero VAD v6 ONNX probability wrapper."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig, SAMPLE_RATE, VAD_FRAME_SAMPLES


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


class SileroOnnxVAD:
    """Return one probability for each consecutive 512-sample 16 kHz frame."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.path = config.silero_model_path
        self.session_factory = session_factory
        self.session: Any | None = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self.reset_count = 0
        self.call_count = 0
        self.load_count = 0

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def verify_artifact(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(f"Silero ONNX artifact is missing: {self.path}")
        size = self.path.stat().st_size
        expected_size = int(self.config.silero["size_bytes"])
        if size != expected_size:
            raise RuntimeError(
                f"Silero artifact size mismatch: {size}; expected {expected_size}"
            )
        actual_hash = sha256_file(self.path)
        expected_hash = str(self.config.silero["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Silero artifact SHA-256 mismatch: {actual_hash}; expected {expected_hash}"
            )
        try:
            import onnx
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install onnx==1.17.0 to verify Silero opset") from exc
        graph = onnx.load(str(self.path), load_external_data=False)
        opsets = {item.domain: int(item.version) for item in graph.opset_import}
        if opsets.get("") != int(self.config.silero["opset"]):
            raise RuntimeError(f"Silero default-domain opset mismatch: {opsets}")
        return {"path": str(self.path), "size_bytes": size, "sha256": actual_hash, "opsets": opsets}

    def load(self) -> None:
        if self.loaded:
            return
        self.verify_artifact()
        if self.session_factory is None:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install onnxruntime==1.22.1") from exc
            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CPUExecutionProvider is unavailable for Silero")
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.path),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
        else:
            self.session = self.session_factory(str(self.path))
        names = {item.name for item in self.session.get_inputs()}
        if names != {"input", "state", "sr"}:
            raise RuntimeError(f"Unexpected Silero ONNX inputs: {sorted(names)}")
        outputs = {item.name for item in self.session.get_outputs()}
        if outputs != {"output", "stateN"}:
            raise RuntimeError(f"Unexpected Silero ONNX outputs: {sorted(outputs)}")
        self.load_count += 1
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self.reset_count += 1

    def probability(self, frame: Any) -> float:
        self.load()
        values = np.ascontiguousarray(
            np.asarray(frame, dtype=np.float32).reshape(-1), dtype=np.float32
        )
        if values.size != VAD_FRAME_SAMPLES:
            raise ValueError("Silero requires exactly 512 samples per call")
        if not np.all(np.isfinite(values)):
            raise ValueError("Silero input contains non-finite samples")
        assert self.session is not None
        framed = np.concatenate((self._context, values.reshape(1, -1)), axis=1)
        output, next_state = self.session.run(
            ["output", "stateN"],
            {
                "input": framed,
                "state": self._state,
                "sr": np.asarray(SAMPLE_RATE, dtype=np.int64),
            },
        )
        probability = float(np.asarray(output).reshape(-1)[0])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid Silero probability: {probability!r}")
        self._state = np.ascontiguousarray(next_state, dtype=np.float32)
        self._context = np.ascontiguousarray(framed[:, -64:], dtype=np.float32)
        self.call_count += 1
        return probability

    def close(self) -> None:
        self.reset()
        self.session = None

    def audit(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "load_count": self.load_count,
            "call_count": self.call_count,
            "reset_count": self.reset_count,
            "artifact": str(self.path),
        }
