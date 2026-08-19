"""Vendored privacy metrics from YahyaAlsharif/OnKith.

Copied rather than reimplemented, and deliberately unmodified, so the
numbers in eval/compare_classifiers.py mean the same thing as the numbers
in Yahya's model card. Two implementations of "typed F1" that agree by
inspection are two numbers, not a comparison.

Source: model/privacy_evaluation/privacy_eval/{schema,metrics}.py
"""

from .metrics import evaluate_predictions
from .schema import EntitySpan, EvalCase, load_cases

__all__ = ["EntitySpan", "EvalCase", "evaluate_predictions", "load_cases"]
