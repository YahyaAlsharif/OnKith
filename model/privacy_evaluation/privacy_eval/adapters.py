"""Thin adapters around the authoritative FullPipeline_V3 privacy components."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOT = REPO_ROOT / "FullPipeline_V3"
TINYBERT_ROOT = REPO_ROOT / "model" / "TinyBERT-4" / "artifacts" / "tinybert_pii"
FP32_ONNX = TINYBERT_ROOT / "onnx_model" / "model_fp32.onnx"
INT8_ONNX = TINYBERT_ROOT / "onnx_model" / "model_int8.onnx"


def _production_imports():
    root = str(PRODUCTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from integrated_pipeline.config import PipelineConfig
    from integrated_pipeline.pii_adapter import PIIMaskerAdapter
    from integrated_pipeline.pipeline import FailClosedMasker
    from integrated_pipeline.span_merger import merge_pii_spans
    from integrated_pipeline.structured_pii import StructuredPIIRecognizer
    from integrated_pipeline.supplemental_pii import SupplementalPIIMasker, SupplementalPIIRecognizer

    return SimpleNamespace(
        PipelineConfig=PipelineConfig,
        PIIMaskerAdapter=PIIMaskerAdapter,
        FailClosedMasker=FailClosedMasker,
        merge_pii_spans=merge_pii_spans,
        StructuredPIIRecognizer=StructuredPIIRecognizer,
        SupplementalPIIMasker=SupplementalPIIMasker,
        SupplementalPIIRecognizer=SupplementalPIIRecognizer,
    )


class _NoStructuredRecognizer:
    def find_spans(self, text: str) -> list:
        return []


def build_model_only_masker(model_path: Path):
    """Use production windowing/BIO/whole-word logic with structured spans disabled."""

    api = _production_imports()
    config = api.PipelineConfig(root=PRODUCTION_ROOT)
    config.data["pii"]["artifact"] = str(Path(model_path).resolve())
    adapter = api.PIIMaskerAdapter(config)
    adapter._structured = _NoStructuredRecognizer()
    return adapter


class StructuredOnlyMasker:
    """Evaluate deterministic recognizers independently of the neural model."""

    def __init__(self, *, include_supplemental: bool):
        api = _production_imports()
        self._merge = api.merge_pii_spans
        self._base = api.StructuredPIIRecognizer()
        self._supplemental = api.SupplementalPIIRecognizer() if include_supplemental else None

    def mask_text(self, text: str):
        import time

        started = time.perf_counter()
        base = self._base.find_spans(text)
        candidates = [span.as_candidate() for span in base]
        if self._supplemental is not None:
            covered = [(span.start_char, span.end_char) for span in base]
            candidates.extend(span.as_candidate() for span in self._supplemental.find_spans(text, covered))
        merged = self._merge(text, [], candidates)
        return SimpleNamespace(
            masked_text=merged.masked_text,
            entities=merged.entities,
            latency_seconds=time.perf_counter() - started,
            safe_to_release=True,
            raw_model_output={"recognizer_only": True},
        )


def build_production_masker():
    """Return the exact current INT8 + structured + supplemental + fail-closed stack."""

    api = _production_imports()
    config = api.PipelineConfig(root=PRODUCTION_ROOT)
    adapter = api.PIIMaskerAdapter(config)
    return api.FailClosedMasker(api.SupplementalPIIMasker(adapter))


def stage_factory(stage: str):
    factories = {
        "fp32_model": lambda: build_model_only_masker(FP32_ONNX),
        "int8_model": lambda: build_model_only_masker(INT8_ONNX),
        "structured": lambda: StructuredOnlyMasker(include_supplemental=False),
        "structured_supplemental": lambda: StructuredOnlyMasker(include_supplemental=True),
        "production": build_production_masker,
    }
    try:
        return factories[stage]()
    except KeyError as exc:
        raise ValueError(f"Unknown stage {stage!r}; choose from {sorted(factories)}") from exc
