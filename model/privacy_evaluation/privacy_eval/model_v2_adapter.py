"""Model V2 masker access point for the evaluation harness.

The masker used to live here because production was pinned to TinyBERT and
could not load a 63-label SentencePiece model at all. That is no longer true:
the pipeline *is* Model V2, and the corrected decoder this module used to
define now ships inside it as ``integrated_pipeline.model_v2_masker``.

Which pipeline root that resolves to comes from
``benchmark_common.PRODUCTION_ROOT``, so the harness and the benchmarks always
agree on one copy. V4's and V5's decoders are byte-identical; the test named
below fails if they ever stop being.

Keeping only a re-export is the point. The accepted INT8 numbers in
``artifacts/model_v2/metrics/evaluation_production_path_int8_onnx.json`` were
measured through this name, and they stay meaningful precisely because the code
behind it is now the same code the deployment runs, rather than a second
implementation that has to be kept in step by hand.

The word-boundary fix (word groups derived from character offsets, never from
SentencePiece word ids) lives with the decoder; see that module's docstring.
"""

from __future__ import annotations

import sys

from .benchmark_common import PRODUCTION_ROOT, REPO_ROOT

if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

from integrated_pipeline.model_v2_masker import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_OVERFLOW_STRIDE,
    ModelV2Decoding,
    ModelV2OnnxMasker,
    build_model_v2_masker,
    repair_bio,
)

#: Where the decoder actually lives, for tests that read its source.
MASKER_SOURCE = PRODUCTION_ROOT / "integrated_pipeline" / "model_v2_masker.py"

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONTEXT_LIMIT",
    "DEFAULT_OVERFLOW_STRIDE",
    "MASKER_SOURCE",
    "ModelV2Decoding",
    "ModelV2OnnxMasker",
    "PRODUCTION_ROOT",
    "REPO_ROOT",
    "build_model_v2_masker",
    "repair_bio",
]
