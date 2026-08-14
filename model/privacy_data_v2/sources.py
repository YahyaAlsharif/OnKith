"""Pinned source registry and span-normalising adapters.

Every source is declared once, with its licence, its split names, its field
names and the revision actually used. Annotation handling is ported from the
audited implementations in ``model/data/OpenPII_1_5M_Unified_PII_Builder.ipynb``:

* offsets are validated against ``text[start:end]`` and repaired only when the
  repair is unambiguous;
* Gretel publishes entity *strings* rather than offsets, so values are anchored
  longest-first into non-overlapping positions;
* overlapping spans are dropped rather than merged;
* non-Latin script is rejected because the deployment is English-only.

Nothing here decides whether a row is *useful*; that is ``selection``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from .label_mapping import mapping_decision
    from .ontology import ENTITIES
    from .records import make_record
except ImportError:  # Direct notebook/script execution from this directory.
    from label_mapping import mapping_decision
    from ontology import ENTITIES
    from records import make_record

#: The base dataset. Revision matches ``data_audit.DATASET_REVISION`` so the
#: historical split can be reproduced exactly.
AI4PRIVACY = {
    "name": "ai4privacy/pii-masking-openpii-1.5m",
    "revision": "a785eb528e28be2693c3718a27e066970de5dadb",
    "licence": "CC-BY-4.0",
    "splits": ("train", "validation"),
    "text_field": "source_text",
    "annotation_field": "privacy_mask",
    "language_field": "language",
    "role": "base",
}

#: Supplementary gap-fillers. ``revision`` is resolved at build time and written
#: into the manifest; pin it here once a build has been accepted.
SUPPLEMENTARY_SOURCES: dict[str, dict[str, Any]] = {
    "gretel": {
        "name": "gretelai/gretel-pii-masking-en-v1",
        "revision": None,
        "licence": "Apache-2.0",
        "splits": ("train", "validation", "test"),
        "text_field": "text",
        "annotation_field": "entities",
        "annotation_style": "entity_strings",
        "language_field": None,
        "role": "supplementary",
    },
    "nemotron": {
        "name": "nvidia/Nemotron-PII",
        "revision": None,
        "licence": "CC-BY-4.0",
        "splits": ("train", "test"),
        "text_field": "text",
        "annotation_field": "spans",
        "annotation_style": "offsets",
        "language_field": None,
        "role": "supplementary",
        "optional": True,
        "note": "May be gated. If unavailable the build continues without it and records the failure.",
    },
    "kiji": {
        "name": "DataikuNLP/kiji-pii-training-data",
        "revision": None,
        "licence": "Apache-2.0",
        "splits": ("train", "test"),
        "text_field": "text",
        "annotation_field": "privacy_mask",
        "annotation_style": "offsets",
        "language_field": "language",
        "role": "supplementary",
    },
    "saudi_mena": {
        "name": "model/data/saudi_mena_pii_dataset.csv",
        "revision": None,
        "licence": "team-generated; see saudi_PII_Builder.ipynb resource_audit",
        "splits": ("generated",),
        "text_field": "text",
        "annotation_field": "spans",
        "annotation_style": "builder_csv",
        "language_field": None,
        "role": "supplementary",
        "local": True,
    },
}

#: Code-point ranges for Greek, Cyrillic, Hebrew, Arabic, Indic, Thai/Tibetan,
#: Hangul Jamo, Kana, CJK and Hangul syllables. Declared as integers so this
#: module never depends on how the file itself is encoded.
_DISALLOWED_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek
    (0x0400, 0x052F),  # Cyrillic
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x08FF),  # Arabic and supplements
    (0x0900, 0x0DFF),  # Indic
    (0x0E00, 0x0FFF),  # Thai, Lao, Tibetan
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # Hangul syllables
)


def contains_non_latin_script(text: str) -> bool:
    """Reject non-Latin scripts while retaining Latin diacritics."""

    for character in text:
        code_point = ord(character)
        if code_point < 128:
            continue
        if any(low <= code_point <= high for low, high in _DISALLOWED_SCRIPT_RANGES):
            return True
        if unicodedata.category(character).startswith("L") and "LATIN" not in unicodedata.name(
            character, ""
        ):
            return True
    return False


def parse_serialized(value: Any) -> list:
    """Hugging Face exposes some annotations as JSON or Python-literal strings."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(value.strip())
        except Exception:
            continue
        if isinstance(result, list):
            return result
    return []


def repair_offset(text: str, start: Any, end: Any, value: Any) -> tuple[int, int, str] | None:
    """Validate an offset, repairing only when the repair is unambiguous."""

    try:
        start_index, end_index = int(start), int(end)
    except (TypeError, ValueError):
        return None
    expected = "" if value is None else str(value)
    if 0 <= start_index < end_index <= len(text):
        actual = text[start_index:end_index]
        if not expected or actual == expected:
            return start_index, end_index, actual
    if expected and 0 <= start_index <= end_index < len(text):
        if text[start_index : end_index + 1] == expected:
            return start_index, end_index + 1, expected
    if expected:
        low = max(0, start_index - 12)
        high = min(len(text), end_index + 12 + len(expected))
        matches = [match.start() for match in re.finditer(re.escape(expected), text[low:high])]
        if len(matches) == 1:
            fixed_start = low + matches[0]
            return fixed_start, fixed_start + len(expected), expected
    return None


def drop_overlaps(spans: list[dict[str, Any]], audit: Counter) -> list[dict[str, Any]]:
    """Keep the earliest, longest span; drop anything that overlaps it."""

    kept: list[dict[str, Any]] = []
    last_end = -1
    for span in sorted(spans, key=lambda item: (item["start"], -(item["end"] - item["start"]))):
        if span["start"] < last_end:
            audit["dropped_overlap"] += 1
            continue
        kept.append(span)
        last_end = span["end"]
    return kept


def normalize_offset_spans(
    text: str, raw: Any, source: str, audit: Counter
) -> list[dict[str, Any]]:
    """Normalise sources that publish explicit character offsets."""

    output: list[dict[str, Any]] = []
    for item in parse_serialized(raw):
        if not isinstance(item, Mapping):
            continue
        native = item.get("label", item.get("type"))
        decision = mapping_decision(source, native)
        if not decision.target:
            audit[f"dropped::{native}::{decision.reason}"] += 1
            continue
        fixed = repair_offset(
            text, item.get("start"), item.get("end"), item.get("value", item.get("text"))
        )
        if not fixed:
            audit[f"dropped_invalid_offset::{native}"] += 1
            continue
        start, end, actual = fixed
        audit[f"mapped::{native}->{decision.target}"] += 1
        output.append({"start": start, "end": end, "label": decision.target, "value": actual})
    return drop_overlaps(output, audit)


def normalize_entity_string_spans(
    text: str, raw: Any, source: str, audit: Counter
) -> list[dict[str, Any]]:
    """Normalise Gretel-style annotations that give values but no offsets."""

    candidates: list[tuple[str, str]] = []
    for item in parse_serialized(raw):
        if not isinstance(item, Mapping):
            continue
        value = str(item.get("entity") or item.get("value") or "")
        natives = item.get("types") or item.get("type") or []
        if isinstance(natives, str):
            natives = [natives]
        decisions = [(native, mapping_decision(source, native)) for native in natives]
        targets = {decision.target for _, decision in decisions if decision.target}
        if len(targets) != 1:
            audit[
                "dropped_type_conflict" if len(targets) > 1 else "dropped_unmapped_type"
            ] += 1
            continue
        target = next(iter(targets))
        native = next(name for name, decision in decisions if decision.target == target)
        if not value:
            audit[f"dropped_missing_value::{native}"] += 1
            continue
        candidates.append((value, target))

    occupied: list[tuple[int, int]] = []
    output: list[dict[str, Any]] = []
    for value, target in sorted(candidates, key=lambda item: -len(item[0])):
        accepted = 0
        for match in re.finditer(re.escape(value), text):
            start, end = match.span()
            if any(start < other_end and end > other_start for other_start, other_end in occupied):
                continue
            output.append(
                {"start": start, "end": end, "label": target, "value": text[start:end]}
            )
            occupied.append((start, end))
            accepted += 1
        if accepted:
            audit[f"mapped::{target}"] += accepted
        else:
            audit[f"dropped_value_not_found::{target}"] += 1
    return drop_overlaps(output, audit)


def normalize_row(
    *,
    source: str,
    row: Mapping[str, Any],
    partition: str,
    audit: Counter,
    uid: str | None = None,
) -> dict[str, Any] | None:
    """Convert one raw source row into a validated canonical record."""

    spec = AI4PRIVACY if source == "ai4privacy" else SUPPLEMENTARY_SOURCES[source]
    text = str(row.get(spec["text_field"]) or "")
    if not text.strip():
        audit["filtered_empty_text"] += 1
        return None

    language_field = spec.get("language_field")
    if language_field:
        language = str(row.get(language_field, "")).strip().casefold()
        if language and language not in {"en", "english"}:
            audit["filtered_non_english"] += 1
            return None
    if contains_non_latin_script(text):
        audit["filtered_non_latin_script"] += 1
        return None

    style = spec.get("annotation_style", "offsets")
    raw = row.get(spec["annotation_field"])
    if style == "entity_strings":
        spans = normalize_entity_string_spans(text, raw, source, audit)
    else:
        spans = normalize_offset_spans(text, raw, source, audit)
    if not spans:
        audit["filtered_no_supported_spans"] += 1
        return None

    identity = uid or row.get("uid") or row.get("id")
    if identity in (None, ""):
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    builder_metadata: dict[str, Any] = {}
    if source == "saudi_mena":
        builder_metadata = {
            "generation_type": str(row.get("generation_type") or "original"),
            "entity_source": str(row.get("entity_source") or "") or None,
            "builder_final_split": str(row.get("final_split") or "") or None,
            "is_saudi": str(row.get("is_saudi") or "").casefold() == "true",
        }
    return make_record(
        uid=f"{source}:{identity}",
        source_text=text,
        privacy_mask=spans,
        source_dataset=source,
        partition=partition,
        source_split=str(row.get("__split__") or "") or None,
        region=str(row.get("region") or "") or None,
        **builder_metadata,
    )


def iter_source_rows(
    source: str,
    split: str,
    *,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream one upstream split, tagging each row with its split name."""

    spec = AI4PRIVACY if source == "ai4privacy" else SUPPLEMENTARY_SOURCES[source]
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if hf_token:
        kwargs["token"] = hf_token
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(spec["name"], **kwargs)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            return
        item = dict(row)
        item["__split__"] = split
        yield item


def iter_saudi_rows(csv_path: str) -> Iterator[dict[str, Any]]:
    """Stream the locally generated Saudi/MENA CSV.

    The builder's own ``final_split`` column is deliberately ignored: the
    centralised Model V2 pipeline owns every split decision.
    """

    import csv as _csv

    with open(csv_path, "r", encoding="utf-8", newline="") as stream:
        for row in _csv.DictReader(stream):
            item = dict(row)
            item["__split__"] = "generated"
            item["spans"] = row.get("spans")
            yield item


def audit_saudi_csv(csv_path: str | Path) -> dict[str, Any]:
    """Validate the local Saudi builder artifact before expensive data work.

    This is intentionally strict: every span must be source-aligned, every
    label must belong to Model V2, every PASSWORD span must be a credential
    disclosure, and row-level provenance must be present.
    """

    import csv as _csv

    try:
        from .password_rules import classify_password_span
    except ImportError:
        from password_rules import classify_password_span

    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Saudi dataset is missing: {path}. "
            "Run model/data/saudi_PII_Builder.ipynb first."
        )

    required_columns = {
        "id", "text", "spans", "labels_present", "source_dataset",
        "source_split", "final_split", "is_saudi", "generation_type",
        "entity_source",
    }
    digest = hashlib.sha256()
    with path.open("rb") as raw_stream:
        for chunk in iter(lambda: raw_stream.read(1 << 20), b""):
            digest.update(chunk)

    row_count = span_count = password_rows = password_spans = 0
    labels: Counter[str] = Counter()
    sources_seen: Counter[str] = Counter()
    generation_types: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    columns: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = _csv.DictReader(stream)
        columns = list(reader.fieldnames or ())
        missing = sorted(required_columns - set(columns))
        if missing:
            raise ValueError(f"Saudi dataset is missing required columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            uid = str(row["id"])
            text = str(row["text"])
            if not uid or uid in seen_ids:
                raise ValueError(f"Saudi row {line_number} has a missing or duplicate id: {uid!r}")
            seen_ids.add(uid)
            normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
            if not normalized or normalized in seen_texts:
                raise ValueError(f"Saudi row {line_number} has empty or duplicate text")
            seen_texts.add(normalized)
            if not str(row["entity_source"]).strip():
                raise ValueError(f"Saudi row {line_number} is missing entity_source provenance")

            try:
                spans = json.loads(row["spans"])
            except Exception as exc:
                raise ValueError(f"Saudi row {line_number} has invalid spans JSON") from exc
            if not isinstance(spans, list) or not spans:
                raise ValueError(f"Saudi row {line_number} must contain at least one span")

            previous_end = -1
            row_has_password = False
            labels_in_spans: set[str] = set()
            for span_index, span in enumerate(spans):
                label = str(span.get("label") or "")
                start, end = int(span["start"]), int(span["end"])
                value = str(span.get("value") or "")
                if label not in set(ENTITIES):
                    raise ValueError(
                        f"Saudi row {line_number} span {span_index} uses unsupported label {label!r}"
                    )
                if not (0 <= start < end <= len(text)) or text[start:end] != value:
                    raise ValueError(
                        f"Saudi row {line_number} span {span_index} is not source-aligned"
                    )
                if start < previous_end:
                    raise ValueError(f"Saudi row {line_number} contains overlapping spans")
                previous_end = end
                labels[label] += 1
                labels_in_spans.add(label)
                span_count += 1
                if label == "PASSWORD":
                    decision = classify_password_span(text, start, end)
                    if not decision.accept:
                        raise ValueError(
                            f"Saudi row {line_number} has a non-disclosure PASSWORD span: "
                            f"{decision.category}: {decision.reason}"
                        )
                    password_spans += 1
                    row_has_password = True

            declared = set(json.loads(row["labels_present"]))
            if declared != labels_in_spans:
                raise ValueError(f"Saudi row {line_number} labels_present does not match spans")
            password_rows += int(row_has_password)
            sources_seen[str(row["source_dataset"])] += 1
            generation_types[str(row["generation_type"])] += 1

    if row_count == 0:
        raise ValueError("Saudi dataset contains no rows")
    if password_spans == 0:
        raise ValueError("Saudi dataset contains no PASSWORD credential disclosures")

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": row_count,
        "spans": span_count,
        "columns": columns,
        "labels": dict(sorted(labels.items())),
        "password_rows": password_rows,
        "password_spans": password_spans,
        "rows_by_source_dataset": dict(sorted(sources_seen.items())),
        "rows_by_generation_type": dict(sorted(generation_types.items())),
        "provenance_present": True,
    }


def resolve_revision(source: str, *, hf_token: str | None = None) -> str | None:
    """Best-effort resolution of the revision actually used, for the manifest."""

    spec = AI4PRIVACY if source == "ai4privacy" else SUPPLEMENTARY_SOURCES[source]
    if spec.get("revision"):
        return str(spec["revision"])
    if spec.get("local"):
        return None
    try:
        from huggingface_hub import dataset_info

        return dataset_info(spec["name"], token=hf_token).sha
    except Exception:
        return None


def source_registry_payload(sources: Iterable[str] | None = None) -> dict[str, Any]:
    """The committed record of what the build read."""

    names = list(sources or SUPPLEMENTARY_SOURCES)
    return {
        "base": dict(AI4PRIVACY),
        "supplementary": {name: dict(SUPPLEMENTARY_SOURCES[name]) for name in names},
        "ontology_note": (
            f"All annotations are mapped into the {len(ENTITIES)} Model V2 entity "
            "types; unmapped source labels become O and are counted in the mapping audit."
        ),
    }
