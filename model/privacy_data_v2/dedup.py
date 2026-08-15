"""Deduplication and the evaluation-contamination firewall.

Two independent mechanisms, deliberately kept separate:

``Deduplicator``
    Removes repeats inside the corpus being built. Exact matching normalises
    Unicode and whitespace; near-duplicate matching uses a span-masked
    structural SimHash so "same template, different values" is caught. Ported
    from the audited implementation in
    ``model/data/OpenPII_1_5M_Unified_PII_Builder.ipynb``.

``ContaminationFirewall``
    Refuses any candidate whose text already appears in an evaluation set. This
    is the single most important safety gate in the Model V2 pipeline: it is
    built from the original AI4Privacy validation, the supplementary holdout and
    both OOD fixtures *before* any training row is written.

    Rejecting a candidate at selection time is not enough on its own: the
    AI4Privacy base split is pinned rather than selected, so it can carry an
    upstream train/validation duplicate straight into the training pool.
    ``filter_records`` is the deterministic sweep that removes those, and
    ``assert_clean`` is the proof that the sweep worked.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

try:
    from .records import normalized_text, text_fingerprint
except ImportError:  # Direct notebook/script execution from this directory.
    from records import normalized_text, text_fingerprint

NEAR_DUP_HAMMING = 2
MAX_LSH_BUCKET = 128
MIN_STRUCTURAL_WORDS = 7
LSH_BANDS = 4


def structural_text(text: str, spans: list[Mapping[str, Any]]) -> str:
    """Replace entity values with their labels to expose template reuse."""

    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor : int(span["start"])])
        parts.append(f" <{span['label']}> ")
        cursor = int(span["end"])
    parts.append(text[cursor:])
    return normalized_text("".join(parts))


def simhash64(text: str) -> int:
    tokens = re.findall(r"[\w<>]+", text, flags=re.UNICODE)
    features = tokens + [f"{left}|{right}" for left, right in zip(tokens, tokens[1:])]
    if not features:
        return 0
    vector = [0] * 64
    for feature in features:
        value = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(vector) if score >= 0)


class Deduplicator:
    """Streaming exact + structural + near-duplicate rejection."""

    def __init__(self, *, enable_near_dedup: bool = True) -> None:
        self.enable_near_dedup = enable_near_dedup
        self.exact_hashes: set[str] = set()
        self.structure_hashes: set[bytes] = set()
        self.signatures: set[int] = set()
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.counts: Counter[str] = Counter()
        self.accepted = 0

    def check_and_add(
        self,
        text: str,
        spans: list[Mapping[str, Any]],
        *,
        allow_structural_repeat: bool = False,
    ) -> tuple[bool, str | None]:
        """Return ``(keep, rejection_reason)`` and register the row when kept.

        ``allow_structural_repeat`` exempts deliberate entity-replacement rows
        from the template check - replacement is a required augmentation method
        and each parent context is consumed once - but never from exact matching.
        """

        exact = text_fingerprint(text)
        if exact in self.exact_hashes:
            self.counts["exact"] += 1
            return False, "exact"

        structure = structural_text(text, spans)
        structure_hash = hashlib.blake2b(structure.encode("utf-8"), digest_size=12).digest()
        signature: int | None = None

        if not allow_structural_repeat and self.enable_near_dedup:
            if len(structure.split()) >= MIN_STRUCTURAL_WORDS:
                if structure_hash in self.structure_hashes:
                    self.counts["same_structure"] += 1
                    return False, "same_structure"
                signature = simhash64(structure)
                candidates: set[int] = set()
                for band in range(LSH_BANDS):
                    key = (band, (signature >> (16 * band)) & 0xFFFF)
                    candidates.update(self.buckets.get(key, ()))
                if any(
                    (signature ^ other).bit_count() <= NEAR_DUP_HAMMING
                    for other in candidates
                ):
                    self.counts["near"] += 1
                    return False, "near"

        self.exact_hashes.add(exact)
        self.structure_hashes.add(structure_hash)
        if signature is not None and signature not in self.signatures:
            self.signatures.add(signature)
            for band in range(LSH_BANDS):
                key = (band, (signature >> (16 * band)) & 0xFFFF)
                bucket = self.buckets[key]
                if len(bucket) < MAX_LSH_BUCKET:
                    bucket.append(signature)
        self.accepted += 1
        return True, None

    def audit(self) -> dict[str, Any]:
        return {
            "accepted_rows": self.accepted,
            "removed_rows": sum(self.counts.values()),
            "removed_by_reason": dict(sorted(self.counts.items())),
            "near_dedup_enabled": self.enable_near_dedup,
            "near_dup_hamming": NEAR_DUP_HAMMING,
        }


class ContaminationFirewall:
    """Reject any candidate row that already exists in an evaluation set.

    Built once from every protected set, then consulted for every candidate.
    A hit is a hard rejection with the offending set named, never a warning.
    """

    def __init__(self) -> None:
        # fingerprint -> (protected set name, uid of the protected row or None)
        self._fingerprints: dict[str, tuple[str, str | None]] = {}
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)

    def add_protected(
        self, name: str, texts: Iterable[str], *, uids: Iterable[str | None] | None = None
    ) -> int:
        """Register one evaluation set. Returns the number of fingerprints held.

        ``uids`` is optional provenance: when supplied it lets a later removal
        name the exact protected row a training candidate collided with, which
        is what turns "contamination happened" into an auditable finding.
        """

        added = 0
        uid_stream = iter(uids) if uids is not None else None
        for text in texts:
            uid = next(uid_stream, None) if uid_stream is not None else None
            fingerprint = text_fingerprint(text)
            # First registration wins so the reported source stays stable.
            if self._fingerprints.setdefault(fingerprint, (name, uid))[0] == name:
                added += 1
        return added

    def add_protected_records(self, name: str, records: Iterable[Mapping[str, Any]]) -> int:
        rows = list(records)
        return self.add_protected(
            name,
            (str(record["source_text"]) for record in rows),
            uids=(str(record.get("uid")) if record.get("uid") is not None else None for record in rows),
        )

    @property
    def size(self) -> int:
        return len(self._fingerprints)

    def check(self, text: str) -> str | None:
        """Return the name of the evaluation set that already holds ``text``."""

        found = self._fingerprints.get(text_fingerprint(text))
        return found[0] if found is not None else None

    def locate(self, text: str) -> tuple[str, str | None] | None:
        """Return ``(protected set, protected uid)`` for a colliding text."""

        return self._fingerprints.get(text_fingerprint(text))

    def accept(self, text: str) -> bool:
        """Return whether the candidate is clean, recording any collision."""

        source = self.check(text)
        if source is None:
            return True
        self.counts[source] += 1
        if len(self.examples[source]) < 5:
            self.examples[source].append(str(text)[:120])
        return False

    def filter_records(
        self, records: Iterable[Mapping[str, Any]], *, label: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Drop every candidate already held by a protected set.

        Returns ``(clean rows, removals)``. Survivors keep their input order and
        removals are recorded in encounter order, so the same input always
        produces the same output - the filter is part of the reproducible build,
        not a repair step someone runs by hand.

        The protected row is never touched. When a training row and an
        evaluation row are the same text, the evaluation row is the one that
        must survive, because a validation set that quietly loses rows stops
        measuring what it claims to measure.
        """

        clean: list[dict[str, Any]] = []
        removals: list[dict[str, Any]] = []
        for record in records:
            text = str(record["source_text"])
            found = self.locate(text)
            if found is None:
                clean.append(dict(record))
                continue
            protected_set, protected_uid = found
            removals.append(
                {
                    "candidate_partition": label,
                    "uid": record.get("uid"),
                    "source_dataset": record.get("source_dataset"),
                    "source_split": record.get("source_split"),
                    "protected_set": protected_set,
                    "protected_uid": protected_uid,
                    "fingerprint": text_fingerprint(text),
                    "text_length": len(text),
                    "span_count": len(record.get("privacy_mask", ())),
                }
            )
        return clean, removals

    def audit(self) -> dict[str, Any]:
        return {
            "protected_fingerprints": self.size,
            "rejected_rows": sum(self.counts.values()),
            "rejected_by_protected_set": dict(sorted(self.counts.items())),
            "examples_by_protected_set": {
                name: list(values) for name, values in sorted(self.examples.items())
            },
        }

    def assert_clean(self, records: Iterable[Mapping[str, Any]], *, label: str) -> None:
        """Hard assertion used after a split is materialised.

        Raises on the first collision. This is what makes "no training row is an
        evaluation row" a proven property rather than an intention.
        """

        for record in records:
            source = self.check(str(record["source_text"]))
            if source is not None:
                raise AssertionError(
                    f"Contamination: {label} row {record.get('uid')!r} also appears in "
                    f"the protected set {source!r}."
                )
