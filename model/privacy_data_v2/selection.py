"""Targeted supplementary row selection.

External corpora are never concatenated wholesale. A row is admitted only when
it strengthens something the pinned AI4Privacy training distribution cannot
teach, and two different weaknesses are treated differently:

*volume weakness* - a label with almost no training spans (``ACCOUNTNUM`` has 1,
``IPV4`` has 2). These need rows until a span target is met.

*diversity weakness* - a label with ample volume but poor frozen-OOD behaviour
(``ORGANISATION`` has 18 spans and scores F1 0.000; ``DATE`` has 134,599 spans
and scores 0.275). Adding more near-identical rows would not help, so these are
admitted only when the row contributes a genuinely new surface group.

Once a row is selected, **every** correctly mapped span in it is kept. A row
pulled in for ``ACCOUNTNUM`` retains its ``GIVENNAME``, ``AMOUNT`` and
``CURRENCY`` annotations; legitimate PII is never demoted to ``O`` merely
because it was not the reason for selection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from .grouped_splits import (
        canonical_template,
        entity_value_factors,
        formatting_family_factors,
    )
    from .ontology import (
        BASELINE_TRAIN_SPAN_COUNTS,
        OOD_WEAK_LABELS,
        WEAK_LABELS,
    )
    from .password_rules import PasswordDecision, filter_password_spans
except ImportError:  # Direct notebook/script execution from this directory.
    from grouped_splits import (
        canonical_template,
        entity_value_factors,
        formatting_family_factors,
    )
    from ontology import BASELINE_TRAIN_SPAN_COUNTS, OOD_WEAK_LABELS, WEAK_LABELS
    from password_rules import PasswordDecision, filter_password_spans

#: Span targets for volume-weak labels, tiered by how empty the base really is.
CRITICAL_SPAN_TARGET = 10_000      # baseline < 10 spans
WEAK_SPAN_TARGET = 12_000          # baseline 10-99 spans
PASSWORD_SPAN_TARGET = 6_000       # deliberately conservative; quality over quota

#: Rows admitted per diversity-weak label, each of which must be novel.
DIVERSITY_ROW_TARGET = 6_000

#: No single source may supply more than this share of one label's spans.
MAX_SOURCE_SHARE = 0.60

REASON_VOLUME = "volume"
REASON_DIVERSITY = "diversity"


def default_span_targets() -> dict[str, int]:
    """Volume targets derived from the measured baseline, not from guesswork."""

    targets: dict[str, int] = {}
    for label in WEAK_LABELS:
        if label == "PASSWORD":
            targets[label] = PASSWORD_SPAN_TARGET
        elif BASELINE_TRAIN_SPAN_COUNTS[label] < 10:
            targets[label] = CRITICAL_SPAN_TARGET
        else:
            targets[label] = WEAK_SPAN_TARGET
    return targets


@dataclass
class SelectionConfig:
    span_targets: dict[str, int] = field(default_factory=default_span_targets)
    diversity_labels: tuple[str, ...] = tuple(sorted(OOD_WEAK_LABELS))
    diversity_row_target: int = DIVERSITY_ROW_TARGET
    max_source_share: float = MAX_SOURCE_SHARE

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_targets": dict(sorted(self.span_targets.items())),
            "diversity_labels": list(self.diversity_labels),
            "diversity_row_target": self.diversity_row_target,
            "max_source_share": self.max_source_share,
        }


@dataclass(frozen=True)
class SelectionOutcome:
    accepted: bool
    reason: str
    selection_reason: tuple[str, ...] = ()
    password_decisions: tuple[PasswordDecision, ...] = ()


def _diversity_keys(record: Mapping[str, Any], label: str) -> set[str]:
    """Surface groups that make a diversity-weak row genuinely new."""

    keys = {f"template::{canonical_template(record)}"}
    keys |= {
        f"value::{factor}"
        for factor in entity_value_factors(record)
        if factor.startswith(f"{label}:")
    }
    keys |= {
        f"format::{factor}"
        for factor in formatting_family_factors(record)
        if factor.startswith(f"{label}:")
    }
    return keys


class SupplementarySelector:
    """Decide, row by row, whether a supplementary candidate earns its place."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()
        self.span_counts: Counter[str] = Counter()
        self.label_source_spans: Counter[tuple[str, str]] = Counter()
        self.diversity_rows: Counter[str] = Counter()
        self.seen_diversity_keys: dict[str, set[str]] = defaultdict(set)
        self.rejections: Counter[str] = Counter()
        self.accepted_rows = 0
        self.accepted_by_source: Counter[str] = Counter()
        self.password_decisions: list[PasswordDecision] = []
        self.carried_spans: Counter[str] = Counter()

    # -- target bookkeeping -------------------------------------------------

    def _volume_needed(self, label: str) -> bool:
        target = self.config.span_targets.get(label, 0)
        return target > 0 and self.span_counts[label] < target

    def _source_cap_reached(self, label: str, source: str) -> bool:
        total = self.span_counts[label]
        if total < 200:  # too small for a share test to be meaningful
            return False
        share = self.label_source_spans[(label, source)] / total
        return share >= self.config.max_source_share

    def _diversity_contribution(self, record: Mapping[str, Any], label: str) -> set[str]:
        if self.diversity_rows[label] >= self.config.diversity_row_target:
            return set()
        keys = _diversity_keys(record, label)
        return keys - self.seen_diversity_keys[label]

    # -- main entry point ---------------------------------------------------

    def consider(self, record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, SelectionOutcome]:
        """Evaluate one mapped candidate row.

        Returns the row to keep (with rejected PASSWORD spans removed) together
        with an auditable outcome. The caller still owns dedup and the
        contamination firewall; this method only answers "is it useful?".
        """

        source = str(record.get("source_dataset", "unknown"))
        record, password_decisions = filter_password_spans(record)
        self.password_decisions.extend(password_decisions)

        labels = {str(span["label"]) for span in record["privacy_mask"]}
        if not labels:
            self.rejections["no_supported_spans"] += 1
            return None, SelectionOutcome(False, "no_supported_spans", (), tuple(password_decisions))

        selection_reason: set[str] = set()
        novel_keys: dict[str, set[str]] = {}

        for label in sorted(labels):
            if self._volume_needed(label) and not self._source_cap_reached(label, source):
                selection_reason.add(f"{label}:{REASON_VOLUME}")
        for label in sorted(labels & set(self.config.diversity_labels)):
            contribution = self._diversity_contribution(record, label)
            if contribution:
                novel_keys[label] = contribution
                selection_reason.add(f"{label}:{REASON_DIVERSITY}")

        if not selection_reason:
            self.rejections["no_target_need"] += 1
            return None, SelectionOutcome(False, "no_target_need", (), tuple(password_decisions))

        # Accepted: bank every supported span, not just the triggering label.
        for span in record["privacy_mask"]:
            label = str(span["label"])
            self.span_counts[label] += 1
            self.label_source_spans[(label, source)] += 1
            if not any(entry.startswith(f"{label}:") for entry in selection_reason):
                self.carried_spans[label] += 1
        for label, keys in novel_keys.items():
            self.seen_diversity_keys[label].update(keys)
            self.diversity_rows[label] += 1
        self.accepted_rows += 1
        self.accepted_by_source[source] += 1

        kept = dict(record)
        kept["selection_reason"] = sorted(selection_reason)
        return kept, SelectionOutcome(
            True, "selected", tuple(sorted(selection_reason)), tuple(password_decisions)
        )

    # -- reporting ----------------------------------------------------------

    def unmet_targets(self) -> dict[str, dict[str, int]]:
        return {
            label: {
                "target_spans": target,
                "selected_spans": self.span_counts[label],
                "shortfall": max(0, target - self.span_counts[label]),
            }
            for label, target in sorted(self.config.span_targets.items())
            if self.span_counts[label] < target
        }

    def audit(self) -> dict[str, Any]:
        by_label_source: dict[str, dict[str, int]] = defaultdict(dict)
        for (label, source), count in sorted(self.label_source_spans.items()):
            by_label_source[label][source] = count
        return {
            "config": self.config.as_dict(),
            "accepted_rows": self.accepted_rows,
            "accepted_rows_by_source": dict(sorted(self.accepted_by_source.items())),
            "rejected_rows_by_reason": dict(sorted(self.rejections.items())),
            "selected_spans_by_label": dict(sorted(self.span_counts.items())),
            "selected_spans_by_label_and_source": {
                label: dict(sorted(sources.items()))
                for label, sources in sorted(by_label_source.items())
            },
            "carried_spans_by_label": dict(sorted(self.carried_spans.items())),
            "diversity_rows_by_label": dict(sorted(self.diversity_rows.items())),
            "diversity_groups_by_label": {
                label: len(keys) for label, keys in sorted(self.seen_diversity_keys.items())
            },
            "unmet_targets": self.unmet_targets(),
            "note": (
                "carried_spans_by_label counts correctly mapped spans kept because "
                "their row was selected for a different label. Unmet targets are "
                "reported as measured evidence; data is never fabricated to close "
                "a shortfall."
            ),
        }
