#!/usr/bin/env python3
"""Run both trained classifiers over the same corpus and print the difference.

    .venv/bin/python eval/compare_classifiers.py
    .venv/bin/python eval/compare_classifiers.py --json results.json

Why this exists rather than quoting the model card. Yahya's held-out
numbers were measured on his machine, through his harness, over the IID
validation split, with the model-only decoder and no structured regex
layer. What this device actually runs is a different thing on all four
counts, and the question the report has to answer is what happens to
*OnKith* if the classifier changes. So both models are run here, over one
corpus, through the same span merger the Pi will use - each in its own
process, so the memory figure belongs to one model rather than to
whichever ran second.

Two arms per model, because this device deliberately releases some of
what the corpus marks private:

    deployment  exactly what the device does, never_mask applied. CITY,
                COUNTRY and TIMEZONE are released on purpose, so the 40
                CITY entities in this corpus count as misses. That is
                the honest number for the deployed system and it is
                lower than any number in anybody's model card.
    model       the same decode with that policy lifted, which is the
                number that compares the two classifiers without our
                product decisions in the way.

The corpus is Yahya's ood_dev split: 360 speech-like rows, spoken emails
("team dot bot at deepfield dot engineering"), hard negatives, and
numeric structures. It is the closest fixture to what a microphone
actually hands this pipeline. Its sibling ood_final.jsonl is a one-shot
frozen holdout that has already been spent once, and is deliberately not
vendored here: re-running it would quietly turn a holdout into a dev set.

Latency here is Mac latency and is only useful as a ratio. The Pi 5
numbers have to be measured on the Pi.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.privacy_eval import evaluate_predictions, load_cases
from integrated_pipeline.config import PII_ENGINE_VARIABLE, PipelineConfig

DEFAULT_CORPUS = ROOT / "eval" / "fixtures" / "ood_dev.jsonl"
#: Longest text this device can actually hand the masker. Capture stops at
#: 8.5 seconds of speech, and the buffer releases one utterance at a time,
#: so a few hundred characters is the ceiling. The corpus also holds
#: 5,000-character documents to stress the 512-token windowing, and those
#: land exactly on the p95 and make it a number about a document rather
#: than about anything this device will ever be asked to mask.
UTTERANCE_CHARACTER_LIMIT = 400
ENGINES = ("tinybert", "deberta")
ENGINE_TITLES = {
    "tinybert": "Model V1  TinyBERT-4 INT8",
    "deberta": "Model V2  DeBERTa-v3-xsmall INT8",
}


def peak_rss_bytes() -> int:
    """Peak resident size of this process so far.

    ru_maxrss is bytes on Darwin and kilobytes on Linux, which is a
    difference that silently reports a Pi as using a thousandth of the
    memory a Mac does.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def measure(engine: str, cases: list[Any], *, apply_policy: bool) -> dict[str, Any]:
    """Load one engine, run every case, and return predictions and timings."""

    previous = os.environ.get(PII_ENGINE_VARIABLE)
    os.environ[PII_ENGINE_VARIABLE] = engine
    try:
        # Imported inside so the engine variable is already set when the
        # config reads it, and re-imported per arm so nothing is carried
        # over from the previous model.
        from integrated_pipeline.pii_adapter import PIIMaskerAdapter

        config = PipelineConfig()
        if not apply_policy:
            # The categories this device keeps on purpose. Lifting them
            # is what makes the two models comparable to each other
            # rather than to our product decisions.
            config.pii["never_mask"] = []
        adapter = PIIMaskerAdapter(config)

        gc.collect()
        rss_before = peak_rss_bytes()
        load_started = time.perf_counter()
        adapter.load()
        load_seconds = time.perf_counter() - load_started

        predictions: dict[str, dict[str, Any]] = {}
        model_latencies: list[float] = []
        utterance_latencies: list[float] = []
        total_latencies: list[float] = []
        for case in cases:
            result = adapter.mask_text(case.text)
            details = result.raw_model_output or {}
            model_seconds = float(details.get("pii_model_latency_seconds", 0.0))
            model_latencies.append(model_seconds)
            if len(case.text) <= UTTERANCE_CHARACTER_LIMIT:
                utterance_latencies.append(model_seconds)
            total_latencies.append(float(result.latency_seconds))
            predictions[case.case_id] = {
                "entities": result.entities,
                "safe_to_release": result.safe_to_release,
                "latency_seconds": result.latency_seconds,
                "raw_model_output": {"bio_repairs": details.get("bio_repairs", 0)},
            }
        rss_after = peak_rss_bytes()

        return {
            "engine": engine,
            "policy_applied": apply_policy,
            "predictions": predictions,
            "label_count": config.pii_label_count,
            "model_bytes": config.pii_model_path.stat().st_size,
            "load_seconds": load_seconds,
            "peak_rss_bytes": rss_after,
            "rss_growth_bytes": max(0, rss_after - rss_before),
            "model_latency": distribution(model_latencies),
            "utterance_latency": distribution(utterance_latencies),
            "total_latency": distribution(total_latencies),
        }
    finally:
        if previous is None:
            os.environ.pop(PII_ENGINE_VARIABLE, None)
        else:
            os.environ[PII_ENGINE_VARIABLE] = previous


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
    }


def summarise(run: dict[str, Any], cases: list[Any]) -> dict[str, Any]:
    scored = evaluate_predictions(cases, run["predictions"])
    overall = scored["overall"]
    return {
        "engine": run["engine"],
        "policy_applied": run["policy_applied"],
        "label_count": run["label_count"],
        "model_bytes": run["model_bytes"],
        "load_seconds": run["load_seconds"],
        "peak_rss_bytes": run["peak_rss_bytes"],
        "rss_growth_bytes": run["rss_growth_bytes"],
        "model_latency": run["model_latency"],
        "utterance_latency": run["utterance_latency"],
        "total_latency": run["total_latency"],
        "typed_f1": overall["typed_entity"]["f1"],
        "typed_precision": overall["typed_entity"]["precision"],
        "typed_recall": overall["typed_entity"]["recall"],
        "leakage": overall["privacy_character"]["private_character_leakage_rate"],
        "private_character_recall": overall["privacy_character"][
            "private_character_recall"
        ],
        "overmasking": overall["privacy_character"][
            "non_private_character_overmasking_rate"
        ],
        "exact_case_rate": overall["case_level"]["exact_typed_case_rate"],
        "hard_negative_fp_rate": overall["case_level"][
            "hard_negative_false_positive_case_rate"
        ],
        "hard_negative_cases": overall["case_level"]["hard_negative_cases"],
        "per_entity_type": overall["per_entity_type"],
        "by_category": {
            name: {
                "typed_f1": slice_["typed_entity"]["f1"],
                "leakage": slice_["privacy_character"][
                    "private_character_leakage_rate"
                ],
                "cases": slice_["case_count"],
                "gold_entities": (
                    slice_["typed_entity"]["tp"] + slice_["typed_entity"]["fn"]
                ),
                "hard_negative_fp_rate": slice_["case_level"][
                    "hard_negative_false_positive_case_rate"
                ],
            }
            for name, slice_ in scored.get("by_category", {}).items()
        },
    }


def run_isolated(engine: str, *, apply_policy: bool, corpus: Path) -> dict[str, Any]:
    """Score one arm in a process of its own.

    Peak RSS is a high-water mark for the whole process and it never comes
    back down, so two models measured in one process report the second one
    as costing whatever the first one did plus its own growth. Since the
    memory question is the one the Pi actually turns on, each arm gets a
    clean process and reports its own peak.
    """

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--corpus",
            str(corpus),
            "--measure-one",
            engine,
            "--policy",
            "1" if apply_policy else "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine} arm failed:\n{completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def _megabytes(value: int) -> str:
    return f"{value / (1024 * 1024):.0f} MB"


def _row(label: str, left: str, right: str, note: str = "") -> str:
    return f"  {label:<26}{left:>14}{right:>14}   {note}"


def _delta(new: float, old: float, *, higher_is_better: bool) -> str:
    difference = new - old
    if abs(difference) < 5e-5:
        return "no change"
    better = (difference > 0) == higher_is_better
    return f"{difference:+.4f}  {'better' if better else 'WORSE'}"


def render(arms: dict[str, dict[str, Any]], corpus: Path, rows: int) -> str:
    lines = [
        "Classifier comparison  Model V1 (TinyBERT-4) vs Model V2 (DeBERTa-v3-xsmall)",
        "",
        f"  corpus                    {corpus.name}, {rows} rows",
        f"  platform                  {platform.system()} {platform.machine()}, "
        f"Python {platform.python_version()}",
        "",
        "Both are INT8 ONNX on CPUExecutionProvider, decoded through the same",
        "span merger and the same structured regex layer.",
    ]

    for policy, heading, blurb in (
        (
            True,
            "As deployed",
            "What the device actually does. CITY, COUNTRY and TIMEZONE are\n"
            "released on purpose, and this corpus marks 40 cities private, so\n"
            "both models are penalised identically for a decision that is ours.",
        ),
        (
            False,
            "Model only",
            "The same decode with that policy lifted, which is the number that\n"
            "compares the two classifiers rather than our product decisions.",
        ),
    ):
        old = arms[("tinybert", policy)]
        new = arms[("deberta", policy)]
        lines.extend(["", "=" * 78, heading, "", blurb, ""])
        lines.append(_row("", "Model V1", "Model V2", ""))
        for label, key, higher in (
            ("typed entity F1", "typed_f1", True),
            ("typed precision", "typed_precision", True),
            ("typed recall", "typed_recall", True),
            ("private char recall", "private_character_recall", True),
            ("private char leakage", "leakage", False),
            ("over-masking rate", "overmasking", False),
            ("exact-case rate", "exact_case_rate", True),
            ("hard-negative FP rate", "hard_negative_fp_rate", False),
        ):
            lines.append(
                _row(
                    label,
                    f"{old[key]:.4f}",
                    f"{new[key]:.4f}",
                    _delta(new[key], old[key], higher_is_better=higher),
                )
            )

    old = arms[("tinybert", True)]
    new = arms[("deberta", True)]
    lines.extend(["", "=" * 78, "Cost", ""])
    lines.append(_row("", "Model V1", "Model V2", ""))
    lines.append(
        _row(
            "model on disk",
            _megabytes(old["model_bytes"]),
            _megabytes(new["model_bytes"]),
            f"{new['model_bytes'] / old['model_bytes']:.1f}x",
        )
    )
    lines.append(
        _row(
            "labels",
            str(old["label_count"]),
            str(new["label_count"]),
            "",
        )
    )
    lines.append(
        _row(
            "load time",
            f"{old['load_seconds']:.2f}s",
            f"{new['load_seconds']:.2f}s",
            f"{new['load_seconds'] / max(old['load_seconds'], 1e-9):.1f}x",
        )
    )
    utterances = new["utterance_latency"]["count"]
    lines.append(
        _row("", "", "", f"utterance-sized rows only ({utterances} of {rows})")
    )
    for label, key in (("classify p50", "p50"), ("classify p95", "p95"), ("classify max", "max")):
        left = old["utterance_latency"][key] * 1000
        right = new["utterance_latency"][key] * 1000
        lines.append(
            _row(
                label,
                f"{left:.1f} ms",
                f"{right:.1f} ms",
                f"{right / max(left, 1e-9):.1f}x",
            )
        )
    lines.append(_row("", "", "", "every row, including 5,000-char documents"))
    for label, key in (("classify p95", "p95"), ("classify max", "max")):
        left = old["model_latency"][key] * 1000
        right = new["model_latency"][key] * 1000
        lines.append(
            _row(
                label,
                f"{left:.1f} ms",
                f"{right:.1f} ms",
                f"{right / max(left, 1e-9):.1f}x",
            )
        )
    lines.append(
        _row(
            "peak RSS",
            _megabytes(old["peak_rss_bytes"]),
            _megabytes(new["peak_rss_bytes"]),
            "process peak, one model each",
        )
    )

    lines.extend(
        [
            "",
            "=" * 78,
            "Where they differ most, by category (model-only typed F1)",
            "",
        ]
    )
    old_cat = arms[("tinybert", False)]["by_category"]
    new_cat = arms[("deberta", False)]["by_category"]
    # A category with no gold entities has no typed F1 - the score is 0/0
    # for everybody, and printing 0.0000 in both columns reads as two
    # models failing completely at the one thing they both do best. Those
    # categories are scored on the only question they can answer: how
    # often a sentence with nothing private in it was masked anyway.
    scored, negatives = [], []
    for name in new_cat:
        if name not in old_cat:
            continue
        (negatives if not new_cat[name]["gold_entities"] else scored).append(name)
    lines.append(_row("", "Model V1", "Model V2", ""))
    for name in sorted(
        scored, key=lambda n: new_cat[n]["typed_f1"] - old_cat[n]["typed_f1"]
    ):
        lines.append(
            _row(
                f"{name} ({new_cat[name]['cases']})",
                f"{old_cat[name]['typed_f1']:.4f}",
                f"{new_cat[name]['typed_f1']:.4f}",
                _delta(
                    new_cat[name]["typed_f1"],
                    old_cat[name]["typed_f1"],
                    higher_is_better=True,
                ),
            )
        )
    for name in sorted(negatives):
        lines.extend(["", f"  {name} ({new_cat[name]['cases']}) holds no entities, "
                          "so it is scored on false positives:"])
        lines.append(
            _row(
                "  masked anyway",
                f"{old_cat[name]['hard_negative_fp_rate']:.4f}",
                f"{new_cat[name]['hard_negative_fp_rate']:.4f}",
                _delta(
                    new_cat[name]["hard_negative_fp_rate"],
                    old_cat[name]["hard_negative_fp_rate"],
                    higher_is_better=False,
                ),
            )
        )

    lines.extend(
        [
            "",
            "=" * 78,
            "Read this before quoting it",
            "",
            "Latency and memory above are this machine's, and are only useful as",
            "a ratio. Model V2 is the larger model on a board nothing has been",
            "measured on yet, so the Pi 5 numbers have to come from the Pi.",
            "",
            "Quote the utterance-sized latency, not the all-rows one. Capture",
            f"stops at 8.5 seconds of speech, so nothing over {UTTERANCE_CHARACTER_LIMIT} characters can",
            "reach the masker on this device. The corpus holds 24 documents of",
            "about 5,000 characters to stress the 512-token windowing, they are",
            "6.7% of the rows, and they land exactly on the p95.",
            "",
            "This corpus is 360 speech-like rows. It is the right shape for this",
            "device and it is small: a category holding 24 cases moves visibly on",
            "one row changing.",
            "",
            "The email category is the one line here that must not be read off the",
            "table. Written addresses score 1.0000 on both models, caught by the",
            "structured regex layer rather than by either classifier. Spoken ones",
            '- "team dot bot at deepfield dot engineering" - score 0.0000 on both:',
            "neither model produces an EMAIL span for them at all. Model V2 scores",
            "lower there only because it guesses more, and its wrong guesses still",
            "cover most of the address, which is why its character leakage is a",
            "third of Model V1's. Spoken email is what a microphone actually",
            "produces, and it is unsolved by both.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--json", type=Path, help="also write the full result here")
    parser.add_argument("--measure-one", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--policy", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cases = load_cases(args.corpus)
    if args.measure_one:
        # One arm, one process. See run_isolated.
        run = measure(args.measure_one, cases, apply_policy=bool(args.policy))
        print(json.dumps(summarise(run, cases)))
        return 0

    arms: dict[Any, dict[str, Any]] = {}
    for policy in (True, False):
        for engine in ENGINES:
            label = "deployment" if policy else "model-only"
            print(
                f"running {engine} ({label}) over {len(cases)} cases...",
                file=sys.stderr,
            )
            arms[(engine, policy)] = run_isolated(
                engine, apply_policy=policy, corpus=args.corpus
            )

    report = render(arms, args.corpus, len(cases))
    print(report)

    if args.json:
        payload = {
            "corpus": str(args.corpus),
            "rows": len(cases),
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "arms": {
                f"{engine}_{'deployment' if policy else 'model_only'}": value
                for (engine, policy), value in arms.items()
            },
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
