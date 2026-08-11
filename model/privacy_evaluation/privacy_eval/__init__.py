"""Reproducible IID and out-of-distribution privacy-model evaluation."""

from .metrics import evaluate_predictions
from .schema import EntitySpan, EvalCase, load_cases

__all__ = ["EntitySpan", "EvalCase", "evaluate_predictions", "load_cases"]
