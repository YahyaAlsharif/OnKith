"""Local TinyBERT INT8 token classification plus structured PII recognition.

Tokenization uses the low-level ``tokenizers`` library rather than
``transformers``. The Hugging Face auto-tokenizer pulls in the full
``transformers`` stack, which imports PyTorch whenever it is installed and costs
several hundred megabytes of resident memory. On a 4 GB Raspberry Pi that memory
is needed by the Moonshine speech model, so this module talks to the same
underlying Rust tokenizer directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PII_WORD_POLICY, PipelineConfig
from .schemas import PIIResult
from .span_merger import (
    SpanCandidate,
    consolidate_predictions_to_words,
    count_partial_word_boundaries,
    decision_to_dict,
    derive_word_ids,
    merge_pii_spans,
)
from .structured_pii import StructuredPIIRecognizer

CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128
EXPECTED_LABEL_COUNT = 67
BATCH_SIZE = 4


def _repair_bio(labels: list[str]) -> tuple[list[str], int]:
    repaired: list[str] = []
    previous_type: str | None = None
    repairs = 0
    for label in labels:
        fixed = label
        if label.startswith("I-"):
            entity_type = label[2:]
            if previous_type != entity_type:
                fixed = f"B-{entity_type}"
                repairs += 1
        repaired.append(fixed)
        previous_type = None if fixed == "O" else fixed[2:]
    return repaired, repairs


class PIIMaskerAdapter:
    """Load only the packaged tokenizer, label map, and INT8 ONNX model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._tokenizer: Any | None = None
        self._session: Any | None = None
        self._id2label: dict[int, str] = {}
        self._structured = StructuredPIIRecognizer()
        self._pad_id = 0
        self._input_names: set[str] = set()
        self.load_count = 0

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._session is not None

    @staticmethod
    def reject_lfs_pointer(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"TinyBERT INT8 model is missing: {path}")
        with path.open("rb") as stream:
            header = stream.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not an ONNX model")
        if path.stat().st_size < 1_000_000:
            raise RuntimeError(f"{path} is unexpectedly small for the INT8 model")

    def load(self) -> None:
        if self.loaded:
            return
        self.config.validate(require_artifacts=True)
        model_path = self.config.tinybert_model_path
        tokenizer_path = self.config.tinybert_tokenizer_path
        label_path = self.config.tinybert_label_map_path
        self.reject_lfs_pointer(model_path)

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned onnxruntime package") from exc
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned tokenizers package") from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider is unavailable")
        labels = {
            int(key): str(value)
            for key, value in json.loads(
                label_path.read_text(encoding="utf-8")
            ).items()
        }
        if len(labels) != EXPECTED_LABEL_COUNT or set(labels) != set(
            range(EXPECTED_LABEL_COUNT)
        ):
            raise RuntimeError("TinyBERT label map must contain IDs 0 through 66")

        # ``tokenizers`` only ever reads the packaged file; there is no hub
        # client involved, so no offline environment flags are required.
        tokenizer_file = tokenizer_path / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Packaged tokenizer is missing: {tokenizer_file}")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        pad_id = self._resolve_pad_id(tokenizer, tokenizer_path)

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        if "logits" not in {output.name for output in session.get_outputs()}:
            raise RuntimeError("TinyBERT ONNX output named 'logits' is missing")
        self._tokenizer = tokenizer
        self._pad_id = pad_id
        self._session = session
        self._id2label = labels
        self._input_names = {inp.name for inp in session.get_inputs()}
        self.load_count += 1

    @staticmethod
    def _resolve_pad_id(tokenizer: Any, tokenizer_path: Path) -> int:
        """Find the padding token id from the tokenizer or its config sidecar."""

        pad_token = "[PAD]"
        config_file = tokenizer_path / "tokenizer_config.json"
        if config_file.is_file():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            pad_token = str(config.get("pad_token") or pad_token)
        pad_id = tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise RuntimeError(f"Tokenizer has no padding token {pad_token!r}")
        return int(pad_id)

    def _window_encodings(self, text: str) -> tuple[Any, list[dict[str, Any]]]:
        """Encode once in full, then again as overlapping context windows.

        The full encoding fixes the authoritative offsets and word grouping for
        the whole utterance. When it exceeds the model context, the text is
        re-encoded with truncation plus ``OVERFLOW_STRIDE`` overlap; the primary
        encoding and its ``overflowing`` siblings together cover every token.
        """

        assert self._tokenizer is not None
        full = self._tokenizer.encode(text)
        if len(full.ids) <= CONTEXT_LIMIT:
            return full, [self._as_window(full)]

        self._tokenizer.enable_truncation(
            max_length=CONTEXT_LIMIT, stride=OVERFLOW_STRIDE
        )
        try:
            primary = self._tokenizer.encode(text)
        finally:
            # Always restore the unrestricted state so the next call sees a
            # tokenizer in a known configuration.
            self._tokenizer.no_truncation()
        windows = [self._as_window(primary)]
        windows.extend(self._as_window(item) for item in primary.overflowing)
        return full, windows

    @staticmethod
    def _as_window(encoding: Any) -> dict[str, Any]:
        return {
            "input_ids": list(encoding.ids),
            "attention_mask": list(encoding.attention_mask),
            "offset_mapping": list(encoding.offsets),
        }

    @staticmethod
    def _word_ids(
        encoding: Any, text: str, offsets: list[tuple[int, int]]
    ) -> list[int | None]:
        ids = getattr(encoding, "word_ids", None)
        if callable(ids):
            ids = ids()
        ids = list(ids) if ids is not None else []
        if len(ids) != len(offsets):
            ids = derive_word_ids(text, offsets)
        return [
            word_id if start != end else None
            for word_id, (start, end) in zip(ids, offsets)
        ]

    def _pad_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Right-pad a batch of windows to the longest member."""

        import numpy as np

        width = max(len(window["input_ids"]) for window in batch)
        input_ids = np.full((len(batch), width), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(batch), width), dtype=np.int64)
        for row, window in enumerate(batch):
            ids = window["input_ids"]
            mask = window["attention_mask"]
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(mask)] = mask
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.zeros((len(batch), width), dtype=np.int64)
        return inputs

    @staticmethod
    def _argmax_with_confidence(logits: Any) -> tuple[Any, Any]:
        import numpy as np

        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / np.sum(exp, axis=-1, keepdims=True)
        predictions = np.argmax(probabilities, axis=-1)
        confidence = np.take_along_axis(
            probabilities, predictions[..., None], axis=-1
        )[..., 0]
        return predictions, confidence

    def mask_text(self, text: str) -> PIIResult:
        """Mask one complete buffer; callers enforce fail-closed exceptions."""

        if not isinstance(text, str):
            raise TypeError("PII input must be text")
        if not text.strip():
            return PIIResult(
                original_text=text,
                masked_text="",
                latency_seconds=0.0,
                entities=[],
                safe_to_release=True,
                privacy_status="masked_empty",
            )
        self.load()
        assert self._tokenizer is not None
        assert self._session is not None
        started = time.perf_counter()
        tinybert_started = time.perf_counter()
        full, windows = self._window_encodings(text)
        all_offsets = [tuple(map(int, item)) for item in full.offsets]
        all_word_ids = self._word_ids(full, text, all_offsets)
        offsets = [item for item in all_offsets if item[0] != item[1]]
        word_ids = [
            word_id
            for word_id, item in zip(all_word_ids, all_offsets)
            if item[0] != item[1]
        ]
        if not offsets:
            raise RuntimeError("Tokenizer produced no source-aligned text tokens")

        prediction_by_offset: dict[tuple[int, int], tuple[int, float]] = {}
        for batch_start in range(0, len(windows), BATCH_SIZE):
            batch = windows[batch_start : batch_start + BATCH_SIZE]
            inputs = self._pad_batch(batch)
            logits = self._session.run(["logits"], inputs)[0]
            expected = (
                inputs["input_ids"].shape[0],
                inputs["input_ids"].shape[1],
                EXPECTED_LABEL_COUNT,
            )
            if tuple(logits.shape) != expected:
                raise RuntimeError(
                    f"Unexpected TinyBERT output {tuple(logits.shape)}; expected {expected}"
                )
            predictions, confidences = self._argmax_with_confidence(logits)
            for row, window in enumerate(batch):
                for offset, label_id, confidence in zip(
                    window["offset_mapping"],
                    predictions[row, : len(window["offset_mapping"])],
                    confidences[row, : len(window["offset_mapping"])],
                ):
                    aligned = tuple(map(int, offset))
                    if aligned[0] == aligned[1]:
                        continue
                    previous = prediction_by_offset.get(aligned)
                    if previous is None or float(confidence) > previous[1]:
                        prediction_by_offset[aligned] = (
                            int(label_id),
                            float(confidence),
                        )

        missing = [offset for offset in offsets if offset not in prediction_by_offset]
        if missing:
            raise RuntimeError(f"Missing TinyBERT prediction for offset {missing[0]}")
        labels, repairs = _repair_bio(
            [self._id2label[prediction_by_offset[offset][0]] for offset in offsets]
        )
        confidences = [prediction_by_offset[offset][1] for offset in offsets]
        consolidated = consolidate_predictions_to_words(
            text,
            offsets,
            labels,
            confidences,
            word_ids=word_ids,
        )
        tinybert_latency = time.perf_counter() - tinybert_started
        tinybert_spans: list[SpanCandidate] = consolidated.spans
        structured_started = time.perf_counter()
        structured_spans = [
            span.as_candidate() for span in self._structured.find_spans(text)
        ]
        structured_latency = time.perf_counter() - structured_started
        merged = merge_pii_spans(text, tinybert_spans, structured_spans)
        return PIIResult(
            original_text=text,
            masked_text=merged.masked_text,
            latency_seconds=time.perf_counter() - started,
            entities=merged.entities,
            safe_to_release=True,
            privacy_status="masked",
            raw_model_output={
                "bio_repairs": repairs,
                "window_count": len(windows),
                "word_policy": PII_WORD_POLICY,
                "structured_enabled": True,
                "tinybert_latency_seconds": tinybert_latency,
                "structured_latency_seconds": structured_latency,
                "span_merger_latency_seconds": merged.latency_seconds,
                "partial_word_boundary_count": count_partial_word_boundaries(
                    text, merged.chosen_spans
                ),
                "span_trace": [
                    decision_to_dict(decision) for decision in merged.decisions
                ],
                "warnings": consolidated.warnings,
            },
        )
