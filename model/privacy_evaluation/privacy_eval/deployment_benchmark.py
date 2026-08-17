"""The fixed held-out subset the CPU deployment benchmark scores on every device.

``Full_Pipeline_CPU_Benchmark.ipynb`` runs on a development workstation and,
later, on the project's 4 GB Raspberry Pi. Comparing those two runs only means
something if both scored the *same rows*, and the Pi will never hold the
materialized corpora: ``artifacts/model_v2/data/`` is gitignored and rebuilding
it needs multi-gigabyte downloads. So the subset is selected once, here, and
committed as a small fixture with a manifest that pins its SHA-256.

Provenance
----------

Rows come from ``model_v2_test.jsonl``, the held-out test split. It is not
training data and it is not the validation set the checkpoint was selected on.
The frozen OOD final fixture is never touched: it buys one number, it has been
spent, and a deployment benchmark is not a reason to re-open it.

Selection
---------

Uniform sampling would give roughly zero rows for ``CURRENCY``, ``AMOUNT`` and
``BANKNAME``, which appear in one to seven rows out of 19,136. Balancing BIO
token counts would be worse: it would distort the length and context
distribution the latency measurement depends on. So selection is
rare-label-first and then natural:

1. every zero-PII row, because the hard-negative false-positive rate has no
   denominator without them;
2. walking entity types from rarest to most common, take rows until each type
   has at least ``MIN_ROWS_PER_LABEL`` rows, or until its rows run out;
3. fill the remaining budget by uniform sampling, so common labels keep their
   natural frequency and the text-length distribution stays realistic.

Every step walks one seeded shuffle of the candidates, ordered by content hash
rather than file order, so the result depends on the seed and the corpus - not
on how the corpus happened to be written out.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "model" / "privacy_evaluation" / "fixtures" / "deployment_benchmark"
SUBSET_PATH = FIXTURE_DIR / "deployment_benchmark.jsonl"
MANIFEST_PATH = FIXTURE_DIR / "deployment_benchmark.manifest.json"
SOURCE_PATH = (
    REPO_ROOT
    / "model"
    / "DeBERTa-v3-xsmall"
    / "artifacts"
    / "model_v2"
    / "data"
    / "model_v2_test.jsonl"
)

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "rare-label-first-then-uniform.v1"
DEFAULT_SEED = 20260817
DEFAULT_TARGET_ROWS = 2000
MIN_ROWS_PER_LABEL = 12
SOURCE_VIEW = "test"

#: Fields carried into the fixture. Everything else in a corpus row is build
#: bookkeeping that the evaluation never reads.
KEPT_FIELDS = ("uid", "source_dataset", "generation_type")


class SubsetRejected(RuntimeError):
    """The committed subset is missing or no longer matches its manifest."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: str | Path) -> str:
    """Repo-relative POSIX path where possible; tests build outside the repo."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def row_id(record: Mapping[str, Any]) -> str:
    """Content address of one row: the text and its annotation, nothing else.

    ``uid`` is not usable as the identity - the test split holds two duplicate
    uids - and a positional index would change the moment the corpus is rebuilt.
    """

    spans = [
        {"start": int(s["start"]), "end": int(s["end"]), "label": str(s["label"])}
        for s in record["privacy_mask"]
    ]
    spans.sort(key=lambda s: (s["start"], s["end"], s["label"]))
    return sha256_bytes(_canonical({"text": record["source_text"], "spans": spans}))[:16]


def project(record: Mapping[str, Any]) -> dict[str, Any]:
    """Trim a corpus row to what the benchmark actually scores."""

    projected: dict[str, Any] = {
        "row_id": row_id(record),
        "source_text": str(record["source_text"]),
        "privacy_mask": [
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "label": str(span["label"]),
                "value": str(span["value"]),
            }
            for span in sorted(
                record["privacy_mask"], key=lambda s: (int(s["start"]), int(s["end"]))
            )
        ],
    }
    for field in KEPT_FIELDS:
        if record.get(field) is not None:
            projected[field] = record[field]
    return projected


def load_source(path: str | Path = SOURCE_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. It is a gitignored build artifact; rebuild it with "
            "model/DeBERTa-v3-xsmall/model_v2_deberta_v3_xsmall.ipynb. The committed "
            "benchmark fixture exists so this file is not needed to *run* the benchmark."
        )
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select(
    records: Sequence[Mapping[str, Any]],
    *,
    target_rows: int = DEFAULT_TARGET_ROWS,
    seed: int = DEFAULT_SEED,
    min_rows_per_label: int = MIN_ROWS_PER_LABEL,
) -> list[dict[str, Any]]:
    """Choose the benchmark rows. Same inputs and seed, same output, everywhere."""

    projected: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate = project(record)
        # Byte-identical rows would give two fixture entries that can never be
        # told apart in the parity file; keep the first in content order.
        projected.setdefault(candidate["row_id"], candidate)
    candidates = [projected[key] for key in sorted(projected)]

    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)

    rows_by_label: defaultdict[str, list[int]] = defaultdict(list)
    for index in order:
        for label in {span["label"] for span in candidates[index]["privacy_mask"]}:
            rows_by_label[label].append(index)

    chosen: list[int] = []
    taken: set[int] = set()

    def take(index: int) -> None:
        if index not in taken:
            taken.add(index)
            chosen.append(index)

    # 1. Hard negatives first: without them there is no false-positive denominator.
    for index in order:
        if not candidates[index]["privacy_mask"]:
            take(index)

    # 2. Rarest entity types first, so a floor for CURRENCY does not get eaten by
    #    rows that were going to be sampled for DATE anyway.
    label_frequency = {label: len(indices) for label, indices in rows_by_label.items()}
    for label in sorted(label_frequency, key=lambda name: (label_frequency[name], name)):
        have = sum(
            1
            for index in taken
            if any(span["label"] == label for span in candidates[index]["privacy_mask"])
        )
        for index in rows_by_label[label]:
            if have >= min_rows_per_label or len(taken) >= target_rows:
                break
            if index in taken:
                continue
            take(index)
            have += 1

    # 3. Natural fill.
    for index in order:
        if len(taken) >= target_rows:
            break
        take(index)

    return [candidates[index] for index in sorted(chosen, key=lambda i: candidates[i]["row_id"])]


def coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Entity/span-level coverage statistics recorded in the manifest."""

    spans_by_label: Counter[str] = Counter()
    rows_by_label: Counter[str] = Counter()
    lengths = []
    for row in rows:
        lengths.append(len(row["source_text"]))
        seen = set()
        for span in row["privacy_mask"]:
            spans_by_label[span["label"]] += 1
            seen.add(span["label"])
        for label in seen:
            rows_by_label[label] += 1
    lengths.sort()
    empty = sum(1 for row in rows if not row["privacy_mask"])
    return {
        "rows": len(rows),
        "rows_without_pii": empty,
        "rows_with_pii": len(rows) - empty,
        "total_spans": int(sum(spans_by_label.values())),
        "entity_types_present": len(spans_by_label),
        "spans_by_entity": dict(sorted(spans_by_label.items())),
        "rows_by_entity": dict(sorted(rows_by_label.items())),
        "character_length": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max": lengths[-1] if lengths else 0,
        },
        "source_datasets": dict(sorted(Counter(str(r.get("source_dataset")) for r in rows).items())),
    }


def write_subset(rows: Sequence[Mapping[str, Any]], path: str | Path = SUBSET_PATH) -> str:
    """Write the fixture with LF endings, whatever the host platform is.

    ``.gitattributes`` marks this file ``-text`` so the bytes survive checkout
    unconverted; writing LF here is the other half of that contract, and the
    manifest SHA-256 is only portable because of both.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_file(path)


def build(
    *,
    source_path: str | Path = SOURCE_PATH,
    subset_path: str | Path = SUBSET_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    target_rows: int = DEFAULT_TARGET_ROWS,
    seed: int = DEFAULT_SEED,
    min_rows_per_label: int = MIN_ROWS_PER_LABEL,
) -> dict[str, Any]:
    """Select, write and describe the benchmark subset. Run once; commit both files."""

    source_path = Path(source_path)
    records = load_source(source_path)
    rows = select(
        records, target_rows=target_rows, seed=seed, min_rows_per_label=min_rows_per_label
    )
    digest = write_subset(rows, subset_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "deployment_benchmark_subset",
        "purpose": (
            "Fixed held-out rows scored by Full_Pipeline_CPU_Benchmark.ipynb on every "
            "device, so workstation and Raspberry Pi results are comparable."
        ),
        "fixture_file": Path(subset_path).name,
        "fixture_sha256": digest,
        "fixture_bytes": Path(subset_path).stat().st_size,
        "row_ids_sha256": sha256_bytes(
            "\n".join(row["row_id"] for row in rows).encode("utf-8")
        ),
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "target_rows": target_rows,
            "min_rows_per_label": min_rows_per_label,
            "row_identity": "sha256(source_text + typed spans)[:16]",
            "order": "candidates sorted by row_id, then one seeded shuffle",
        },
        "source": {
            "view": SOURCE_VIEW,
            "path": _relative(source_path),
            "sha256": sha256_file(source_path),
            "rows": len(records),
            "note": (
                "Held-out test split. Not training data, not the validation set used for "
                "checkpoint selection, and not the frozen OOD final fixture."
            ),
        },
        "coverage": coverage(rows),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_subset(
    *,
    subset_path: str | Path = SUBSET_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    verify: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the committed subset and prove it is the one the manifest describes."""

    subset_path, manifest_path = Path(subset_path), Path(manifest_path)
    if not subset_path.is_file():
        raise SubsetRejected(
            f"benchmark subset is missing: {subset_path}. Rebuild it with "
            "`python -m privacy_eval.deployment_benchmark --build` on a machine that "
            "holds the corpora, then commit both files."
        )
    if not manifest_path.is_file():
        raise SubsetRejected(f"benchmark subset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        digest = sha256_file(subset_path)
        if digest != manifest.get("fixture_sha256"):
            raise SubsetRejected(
                f"benchmark subset SHA-256 {digest} does not match the manifest "
                f"{manifest.get('fixture_sha256')}. On Windows this is usually an "
                "end-of-line conversion; .gitattributes marks the fixture -text."
            )
    with subset_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if verify and len(rows) != manifest.get("coverage", {}).get("rows"):
        raise SubsetRejected(
            f"benchmark subset holds {len(rows)} rows, manifest says "
            f"{manifest.get('coverage', {}).get('rows')}"
        )
    return rows, manifest


def take(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """A deterministic prefix, for smoke runs.

    Rows are stored in ``row_id`` order, so the first *n* are the same *n* on
    every machine without any further sampling decision.
    """

    return [dict(row) for row in rows[: max(0, int(count))]]


def _print_coverage(manifest: Mapping[str, Any]) -> None:
    stats = manifest["coverage"]
    print(f"rows                : {stats['rows']:,}")
    print(f"rows without PII    : {stats['rows_without_pii']}")
    print(f"total spans         : {stats['total_spans']:,}")
    print(f"entity types present: {stats['entity_types_present']}")
    print(f"character length    : {stats['character_length']}")
    print("\nrows per entity type (rarest first):")
    for label, count in sorted(stats["rows_by_entity"].items(), key=lambda item: (item[1], item[0])):
        print(f"  {label:20} {count:>5}  ({stats['spans_by_entity'][label]:,} spans)")


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="reselect and rewrite the fixture")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=DEFAULT_TARGET_ROWS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.build:
        manifest = build(target_rows=args.rows, seed=args.seed)
        print(f"wrote {SUBSET_PATH.relative_to(REPO_ROOT).as_posix()}")
        print(f"  sha256 {manifest['fixture_sha256']}")
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()}\n")
        _print_coverage(manifest)
        return 0

    _, manifest = load_subset()
    print(f"{SUBSET_PATH.relative_to(REPO_ROOT).as_posix()} verified")
    print(f"  sha256 {manifest['fixture_sha256']}\n")
    _print_coverage(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
