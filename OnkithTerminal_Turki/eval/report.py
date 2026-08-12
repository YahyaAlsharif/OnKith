"""Turn the run history into a table that can go straight into the report.

`results/history.jsonl` is written for machines: every run, every engine,
every split, in full. This reads it back and prints the part a person has
to defend in writing, as markdown.

    .venv/bin/python eval/report.py                 latest run, recorded half
    .venv/bin/python eval/report.py --split all     every split
    .venv/bin/python eval/report.py --history wake  one metric over time

The recorded half is the default because it is the only half that is
evidence. A table of synthetic scores in a report is a table of how well
each recogniser reads back a machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HISTORY = HERE / "results" / "history.jsonl"

# Name, key, whether larger is better, and how to render it.
METRICS = [
    ("Wake recall", "wake_recall", True, "pct0"),
    ("False wake", "false_wake_rate", False, "pct0"),
    ("Intent", "intent_accuracy", True, "pct0"),
    ("Digits kept", "entities_kept", True, "pct0"),
    ("WER (name)", "wer_wake", False, "pct1"),
    ("WER (quiet)", "wer_quiet", False, "pct1"),
    ("RTF", "rtf", False, "rtf"),
]


def render(value: float, style: str) -> str:
    if style == "pct0":
        return f"{100 * value:.0f}%"
    if style == "pct1":
        return f"{100 * value:.1f}%"
    return f"{value:.2f}x"


def entries() -> list[dict]:
    if not HISTORY.exists():
        raise SystemExit(
            f"No runs recorded yet. Run: .venv/bin/python {Path('eval/run_eval.py')}"
        )
    found = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            found.append(json.loads(line))
    if not found:
        raise SystemExit("history.jsonl is empty.")
    return found


def latest_by_engine(runs: list[dict], split: str) -> dict[str, dict]:
    """The most recent score each engine has on this split.

    Engines are keyed separately so an engine that was not in the last run
    still shows its last real number rather than vanishing from the report.
    Its date is printed, so a stale row is visible as stale.
    """

    seen: dict[str, dict] = {}
    for run in runs:
        for row in run.get("results", []):
            # Runs recorded before the split existed covered the whole
            # corpus, which is the overall number under a different name.
            row_split = row.get("split", "overall")
            if row_split != split:
                continue
            seen[row["engine"]] = {**row, "at": run["at"], "revision": run["revision"]}
    return seen


def markdown_table(rows: dict[str, dict], caption: str) -> str:
    if not rows:
        return f"_No runs recorded for {caption}._\n"
    engines = list(rows)
    best: dict[str, float] = {}
    for _, key, larger_is_better, _style in METRICS:
        values = [rows[e].get(key, 0.0) for e in engines]
        best[key] = max(values) if larger_is_better else min(values)

    lines = [f"| Metric | {' | '.join(rows[e]['detail'] for e in engines)} |"]
    lines.append("|" + "---|" * (len(engines) + 1))
    for name, key, _larger, style in METRICS:
        cells = []
        for engine in engines:
            value = rows[engine].get(key, 0.0)
            text = render(value, style)
            # Bold the winner so the reader does not have to work out the
            # direction of each metric to see who took it.
            cells.append(f"**{text}**" if value == best[key] else text)
        lines.append(f"| {name} | {' | '.join(cells)} |")
    counts = [rows[e].get("counts", {}) for e in engines]
    if any(counts):
        lines.append(
            "| _woke/called_ | "
            + " | ".join(c.get("woke_when_called", "") for c in counts)
            + " |"
        )
    # An engine keeps its last real score even when it was not in the most
    # recent run, so the column can be older than the one beside it. When
    # they do not all come from the same commit, each one says where it
    # came from rather than sitting under one heading that speaks for all.
    if len({(rows[e]["at"][:10], rows[e]["revision"]) for e in engines}) > 1:
        lines.append(
            "| _scored_ | "
            + " | ".join(
                f"{rows[e]['at'][:10]} `{rows[e]['revision']}`" for e in engines
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def one_revision(rows: dict[str, dict]) -> str | None:
    """The date and revision every column shares, or None if they differ."""

    stamps = {(row["at"][:10], row["revision"]) for row in rows.values()}
    if len(stamps) != 1:
        return None
    day, revision = stamps.pop()
    return f"{day} at `{revision}`"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="recorded",
        choices=["recorded", "synthetic", "overall", "all"],
        help="which half of the corpus to report (default: recorded)",
    )
    parser.add_argument(
        "--history",
        help="print one metric over every run instead, e.g. wake_recall",
    )
    args = parser.parse_args()

    runs = entries()

    if args.history:
        key = args.history
        if key not in {metric[1] for metric in METRICS}:
            known = ", ".join(metric[1] for metric in METRICS)
            raise SystemExit(f"Unknown metric {key!r}. Have: {known}")
        style = next(metric[3] for metric in METRICS if metric[1] == key)
        print(f"### {key} over time, recorded half\n")
        print("| Date | Revision | Engine | Value | Recorded clips |")
        print("|---|---|---|---|---|")
        for run in runs:
            for row in run.get("results", []):
                if row.get("split", "overall") != "recorded":
                    continue
                print(
                    f"| {run['at'][:10]} | `{run['revision']}` | {row['detail']} "
                    f"| {render(row.get(key, 0.0), style)} "
                    f"| {run.get('corpus_recorded', 0)} |"
                )
        return 0

    wanted = (
        ["recorded", "synthetic", "overall"] if args.split == "all" else [args.split]
    )
    captions = {
        "recorded": "Recorded voice, real room. This is the evidence.",
        "synthetic": "Synthetic speech. A floor on difficulty, not a result.",
        "overall": "Both halves together.",
    }
    for split in wanted:
        rows = latest_by_engine(runs, split)
        print(f"### {captions[split]}\n")
        shared = one_revision(rows) if rows else None
        if shared:
            print(f"_Scored {shared}._\n")
        print(markdown_table(rows, split))

    print("Higher is better for wake, intent and digits. Lower is better for")
    print("false wake, WER and RTF. RTF is meaningless off the target machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
