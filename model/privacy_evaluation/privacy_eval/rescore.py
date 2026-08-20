"""Re-score a completed audio benchmark folder with the current definitions.

Why this exists
---------------
``audio_benchmark.py`` writes each entity's outcome into the run folder at the
time the run happens. When the scoring definition changes afterwards - as it did
when ``locate_entity`` learned to fold spoken numbers and to trim sentence
punctuation off a located region - every folder written before the change holds
outcomes computed the old way. Comparing a fresh Whisper run against a frozen
Canary folder then compares two definitions, not two recognizers.

Re-running Canary to fix that is not an option: the V3 build is archived and its
recognizer is not installed. But nothing here needs a recognizer, and nothing
here needs a privacy model either, because every run saved the two pieces of
text that a re-score needs:

``privacy_input_text``
    exactly what the masker was given.
``masked_text``
    exactly what it returned.

The predicted character spans were never written to disk, but they are
recoverable from that pair without guessing: masking replaces each span with a
``[LABEL_n]`` placeholder and copies everything else through unchanged, so
splitting the masked text on the placeholders gives literal chunks that can be
matched back into the source in order. Whatever lies between two consecutive
matches is what was replaced. ``verify_recovery`` checks the reconstruction
against each run's own recorded ``mask_coverage``, and every committed run
folder reproduces exactly.

So a Canary folder and a Whisper folder can both be scored by this module, with
one implementation, and the resulting numbers are comparable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audio_benchmark import (
    FULL_MASK_THRESHOLD,
    MANIFEST_PATH,
    PRESERVATION_THRESHOLD,
    _informative_characters,
    classify_entity,
    locate_entity,
    mask_coverage,
)
from .stt_comparison import engine_of, pipeline_of, read_predictions

#: ``[GIVENNAME_1]``, ``[TELEPHONENUM_12]`` - the shape ``span_merger`` writes.
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")

#: Outcomes in which no part of the entity reached the released text.
PROTECTED_OUTCOMES = ("masked", "stt_altered_masked")


class RecoveryFailed(RuntimeError):
    """The masked text could not be aligned back onto its input."""


def recover_predicted_spans(source: str, masked: str) -> list[tuple[int, int]]:
    """Character ranges of ``source`` that the masker replaced.

    Exact, not approximate: the literal text between placeholders is copied
    through byte for byte, so matching those literals back in order pins every
    replaced region. Adjacent placeholders - ``[PASSPORTNUM_1][TELEPHONENUM_1]``,
    which the merger does emit - produce one combined range, which is all the
    character-coverage arithmetic downstream needs.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for literal in PLACEHOLDER.split(masked):
        if not literal:
            continue
        found = source.find(literal, cursor)
        if found < 0:
            raise RecoveryFailed(
                f"masked text does not align onto its input at offset {cursor}"
            )
        if found > cursor:
            spans.append((cursor, found))
        cursor = found + len(literal)
    if cursor < len(source):
        spans.append((cursor, len(source)))
    return spans


def covered_characters(spans: Iterable[tuple[int, int]]) -> set[int]:
    covered: set[int] = set()
    for start, end in spans:
        covered.update(range(start, end))
    return covered


def gold_spans(manifest_path: Path = MANIFEST_PATH
               ) -> dict[str, list[dict[str, Any]]]:
    """Annotated entities per clip, from the committed fixture manifest.

    Only clips carrying a privacy annotation appear. The LibriSpeech subset is
    read prose full of person names with no gold spans, so an empty entry for
    one of those would read as "seven entities to protect, none of them there"
    in the hard-negative count. It measures recognition, not masking.
    """

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        clip["clip_id"]: list(clip["expected_private_spans"])
        for clip in manifest["clips"]
        if clip.get("privacy_annotated", True)
    }


def verify_recovery(directory: Path) -> dict[str, Any]:
    """Reproduce the run's own ``mask_coverage`` from the recovered spans.

    A guard, not a metric. If this does not come back clean the recovered spans
    are wrong and nothing else in this module should be believed.
    """

    checked = mismatched = 0
    for row in read_predictions(Path(directory)):
        source, masked = row.get("privacy_input_text"), row.get("masked_text")
        if source is None or masked is None:
            continue
        covered = covered_characters(recover_predicted_spans(source, masked))
        for entity in row.get("entities") or []:
            start, end = entity["hypothesis_char_start"], entity["hypothesis_char_end"]
            replayed = mask_coverage(source, start, end, covered)
            checked += 1
            if abs(replayed - entity["mask_coverage"]) > 1e-9:
                mismatched += 1
    return {"entities_checked": checked, "mismatched": mismatched,
            "exact": mismatched == 0}


def rescore_run(directory: Path,
                manifest_path: Path = MANIFEST_PATH) -> dict[str, Any] | None:
    """Recompute one saved run's end-to-end privacy outcome, current definitions.

    Returns ``None`` for a folder with no usable predictions - a run that never
    reached the recognizer, or a text benchmark folder.
    """

    directory = Path(directory)
    config_path = directory / "benchmark_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [
        row for row in read_predictions(directory)
        if row.get("privacy_input_text") is not None
        and row.get("masked_text") is not None
    ]
    if not rows:
        return None

    gold = gold_spans(manifest_path)
    entities: list[dict[str, Any]] = []
    scored_clips: list[str] = []
    negatives = negative_hits = 0

    for row in rows:
        clip_id = row["clip_id"]
        source = row["privacy_input_text"]
        expected = gold.get(clip_id, [])
        if row.get("expected_entities") == 0 and clip_id in gold:
            negatives += 1
            if row.get("predicted_entities"):
                negative_hits += 1
        if not expected:
            continue
        covered = covered_characters(
            recover_predicted_spans(source, row["masked_text"])
        )
        scored_clips.append(clip_id)
        for span in expected:
            location = locate_entity(row["reference_transcript"], span, source)
            coverage = mask_coverage(
                source, location["hypothesis_char_start"],
                location["hypothesis_char_end"], covered,
            )
            entities.append({
                "clip_id": clip_id,
                "label": span["label"],
                "reference_text": span["text"],
                "outcome": classify_entity(location, coverage),
                "mask_coverage": coverage,
                **location,
            })

    outcomes = {name: 0 for name in
                ("masked", "partially_masked", "leaked_privacy_model",
                 "stt_altered_masked", "stt_altered_leaked")}
    for entity in entities:
        outcomes[entity["outcome"]] += 1

    by_clip: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        by_clip.setdefault(entity["clip_id"], []).append(entity)
    sanitized = sum(
        1 for group in by_clip.values()
        if all(entity["outcome"] in PROTECTED_OUTCOMES for entity in group)
    )

    covered_total = sum(
        entity["mask_coverage"] * _informative_characters(entity["hypothesis_surface"])
        for entity in entities
    )
    informative_total = sum(
        _informative_characters(entity["hypothesis_surface"]) for entity in entities
    )
    preserved = sum(
        1 for entity in entities
        if entity["preservation_ratio"] >= PRESERVATION_THRESHOLD
    )
    protected = outcomes["masked"] + outcomes["stt_altered_masked"]

    by_label: dict[str, dict[str, int]] = {}
    for entity in entities:
        bucket = by_label.setdefault(
            entity["label"], {"expected": 0, "preserved_by_stt": 0,
                              "masked": 0, "protected": 0}
        )
        bucket["expected"] += 1
        bucket["preserved_by_stt"] += int(
            entity["preservation_ratio"] >= PRESERVATION_THRESHOLD
        )
        bucket["masked"] += int(entity["outcome"] == "masked")
        bucket["protected"] += int(entity["outcome"] in PROTECTED_OUTCOMES)

    return {
        "run_id": directory.name,
        "directory": str(directory),
        "engine": engine_of(config),
        "pipeline_version": pipeline_of(config),
        "device_label": config.get("device_label"),
        "board_model": _board_of(directory),
        "mode": config.get("mode"),
        "scoring": "rescored",
        "thresholds": {
            "preservation_ratio": PRESERVATION_THRESHOLD,
            "full_mask_coverage": FULL_MASK_THRESHOLD,
        },
        "expected_entities": len(entities),
        "entities_preserved_by_stt": preserved,
        "stt_preservation_rate": preserved / len(entities) if entities else 0.0,
        "entities_fully_masked": outcomes["masked"],
        # The claim the deployment actually makes. An entity the recognizer
        # rewrote and the masker covered anyway did not reach the released text,
        # so it is protected; it is simply not evidence about the model's
        # accuracy on the entity as spoken, which is why both are reported.
        "entities_protected": protected,
        "entity_protection_rate": protected / len(entities) if entities else 0.0,
        "character_recall_after_stt": (
            covered_total / informative_total if informative_total else 0.0
        ),
        "character_leakage_after_stt": (
            1.0 - covered_total / informative_total if informative_total else 0.0
        ),
        "outcomes": outcomes,
        "clips_scored": len(scored_clips),
        "fully_sanitized_clips": sanitized,
        "fully_sanitized_clip_rate": (
            sanitized / len(scored_clips) if scored_clips else 0.0
        ),
        "hard_negative_clips": negatives,
        "hard_negative_false_positive_clips": negative_hits,
        "by_entity_type": dict(sorted(by_label.items())),
        "entities": entities,
        "recovery": verify_recovery(directory),
    }


def _board_of(directory: Path) -> str | None:
    path = Path(directory) / "system_info.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("board_model")
    except (OSError, json.JSONDecodeError):
        return None


def rescore_runs(directories: Iterable[Path],
                 manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    scored = [rescore_run(Path(item), manifest_path) for item in directories]
    return [item for item in scored if item]


def device_of(scored: Mapping[str, Any]) -> str:
    """A short, unambiguous device name for a table column."""

    board = scored.get("board_model")
    if board:
        return board.replace("Raspberry Pi", "Pi").strip()
    return str(scored.get("device_label") or "unknown device")
