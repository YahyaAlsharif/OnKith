"""Audit long fixtures with the exact packaged TinyBERT tokenizer/window policy."""

from __future__ import annotations

import json
from pathlib import Path

from .adapters import PRODUCTION_ROOT
from .schema import load_cases


CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128


def audit(root: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PRODUCTION_ROOT / "models" / "tinybert_pii" / "tokenizer",
        local_files_only=True,
        use_fast=True,
    )
    rows = []
    for partition in ("dev", "final"):
        cases = load_cases(root / "fixtures" / f"ood_{partition}.jsonl")
        for case in cases:
            if case.category != "long_window":
                continue
            full = tokenizer(case.text, truncation=False, return_offsets_mapping=True)
            overflow = tokenizer(
                case.text,
                truncation=True,
                max_length=CONTEXT_LIMIT,
                stride=OVERFLOW_STRIDE,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
            )
            entity = case.entities[0]
            covering_windows = 0
            for offsets in overflow["offset_mapping"]:
                starts = [int(start) for start, end in offsets if start != end]
                ends = [int(end) for start, end in offsets if start != end]
                if starts and min(starts) <= entity.start and max(ends) >= entity.end:
                    covering_windows += 1
            entity_token_indices = [
                index
                for index, (start, end) in enumerate(full["offset_mapping"])
                if int(start) < entity.end and entity.start < int(end)
            ]
            rows.append(
                {
                    "case_id": case.case_id,
                    "partition": partition,
                    "subcategory": case.subcategory,
                    "tags": list(case.tags),
                    "full_token_count": len(full["input_ids"]),
                    "window_count": len(overflow["input_ids"]),
                    "entity_first_token_index": min(entity_token_indices),
                    "entity_last_token_index": max(entity_token_indices),
                    "entity_covering_window_count": covering_windows,
                }
            )
    return {
        "schema_version": 1,
        "tokenizer": "packaged TinyBERT-4 fast tokenizer",
        "context_limit": CONTEXT_LIMIT,
        "overflow_stride": OVERFLOW_STRIDE,
        "case_count": len(rows),
        "minimum_full_token_count": min(row["full_token_count"] for row in rows),
        "maximum_full_token_count": max(row["full_token_count"] for row in rows),
        "minimum_window_count": min(row["window_count"] for row in rows),
        "maximum_window_count": max(row["window_count"] for row in rows),
        "all_entities_covered": all(row["entity_covering_window_count"] >= 1 for row in rows),
        "cases": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "baselines" / "tinybert_long_window_audit.json"
    payload = audit(root)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{payload['case_count']} cases, tokens {payload['minimum_full_token_count']}-"
        f"{payload['maximum_full_token_count']}, windows {payload['minimum_window_count']}-"
        f"{payload['maximum_window_count']}, covered={payload['all_entities_covered']}"
    )


if __name__ == "__main__":
    main()
