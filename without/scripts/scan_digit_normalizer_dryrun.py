#!/usr/bin/env python3
"""Dry run: apply the digit normaliser to stored transcripts, report impact.

Read-only against the database - opens it in SQLite read-only mode and never
writes. Runs ``normalize_spoken_digits`` (app/digit_normalizer.py) over every
``raw_transcript`` and ``masked_transcript`` in ``transcript_batches``, and
reports how many fields changed plus sample before/after pairs. This is the
only check that shows whether the digit-normalisation rules fire on real
speech, since every unit test uses hand-written sentences.

Usage
-----
    python scripts/scan_digit_normalizer_dryrun.py
    python scripts/scan_digit_normalizer_dryrun.py --db /path/to/session_transcript.db
    python scripts/scan_digit_normalizer_dryrun.py --samples 20
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.digit_normalizer import normalize_spoken_digits


def _rows(db_path: Path) -> list[tuple[str, str | None, str | None]]:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT batch_id, raw_transcript, masked_transcript "
            "FROM transcript_batches ORDER BY batch_id"
        )
        return cursor.fetchall()
    finally:
        conn.close()


def scan(db_path: Path, sample_limit: int) -> None:
    rows = _rows(db_path)

    total_fields = 0
    changed: list[tuple[str, str, str, str]] = []
    for batch_id, raw, masked in rows:
        for field_name, text in (
            ("raw_transcript", raw),
            ("masked_transcript", masked),
        ):
            if text is None:
                continue
            total_fields += 1
            normalised = normalize_spoken_digits(text)
            if normalised != text:
                changed.append((batch_id, field_name, text, normalised))

    print(f"Database: {db_path}")
    print(f"Rows scanned: {len(rows)}")
    print(f"Fields scanned (raw_transcript + masked_transcript): {total_fields}")
    print(f"Fields changed: {len(changed)}")
    print()
    for batch_id, field_name, before, after in changed[:sample_limit]:
        print(f"--- {batch_id} [{field_name}] ---")
        print(f"BEFORE: {before}")
        print(f"AFTER : {after}")
        print()
    if len(changed) > sample_limit:
        print(f"... {len(changed) - sample_limit} more changed field(s) not shown "
              f"(raise --samples to see more)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "session_transcript.db",
        help="Path to session_transcript.db (default: repo root copy)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Maximum number of before/after pairs to print (default: 20)",
    )
    args = parser.parse_args()
    if not args.db.exists():
        parser.error(f"database not found: {args.db}")
    scan(args.db, args.samples)


if __name__ == "__main__":
    main()
