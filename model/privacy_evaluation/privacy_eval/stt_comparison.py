"""Score two speech recognizers against each other from saved result folders.

The Canary runs - workstation and Raspberry Pi - are a frozen historical
baseline. They are **never rerun**: everything here reads
``audio_predictions.jsonl`` out of a completed run folder, so a Canary number in
a comparison is the number that run produced, recomputed by the same code that
scores the Whisper run rather than copied from a differently-defined field.

Why this module exists
----------------------
The audio benchmark's ``aggregate_wer`` is the standard definition: Levenshtein
over lowercased, punctuation-stripped words. That was a fair number when the
recognizer was Canary, which ran with ``pnc: no`` and wrote spoken numbers as
words - the same form the reference transcripts are written in.

Whisper writes numerals. On this corpus the reference says "sixty three" and
"five five five, zero one eight eight"; Whisper writes "63" and "55550188".
Those are correct transcriptions, and the standard WER charges eight
substitutions for the second one. Comparing Canary's raw WER against Whisper's
raw WER therefore measures orthography, not recognition.

So both numbers are reported:

``aggregate_wer``
    Unchanged, exactly as each run recorded it. Comparable across devices for
    one engine; **not** comparable across engines.

``aggregate_wer_number_normalised``
    The same Levenshtein, after both the reference and the hypothesis have had
    spoken numbers folded to numerals. Identical treatment on both sides and for
    both engines, which is what makes an engine-to-engine comparison mean
    something.

Neither one is a speech-recognition benchmark. This is a few dozen clips.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Number normalisation
# --------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "oh": 0, "o": 0, "nought": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
#: Ordinals the corpus actually uses, mapped to the cardinal a numeral carries.
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}
_DECADES = {
    "twenties": "20s", "thirties": "30s", "forties": "40s", "fifties": "50s",
    "sixties": "60s", "seventies": "70s", "eighties": "80s", "nineties": "90s",
}
#: Written by one engine and not the other; dropped from both sides so a comma
#: read aloud does not count as a word.
_FILLER = {"and"}

_ORDINAL_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th)$")


def _fold_numbers(tokens: Sequence[str]) -> list[str]:
    """Fold spoken number words into numerals, leaving everything else alone.

    Three shapes, which is all this corpus contains:

    * a run of single digits - "five five five zero one eight eight" - becomes
      one numeral, "55550188", because that is what a recognizer writing
      numerals produces for a telephone number read aloud;
    * a tens word followed by a unit - "sixty three" - becomes "63";
    * a teen, a tens word or an ordinal on its own becomes its numeral.

    "and" is dropped between number words only: "five five five and zero one" is
    one number read with a pause, not two numbers joined by a conjunction.
    """

    output: list[str] = []
    digit_run: list[str] = []

    def flush() -> None:
        if digit_run:
            output.append("".join(digit_run))
            digit_run.clear()

    index = 0
    while index < len(tokens):
        token = tokens[index]

        # A numeral the recognizer already wrote. "19th" -> "19".
        ordinal = _ORDINAL_SUFFIX.match(token)
        if ordinal:
            token = ordinal.group(1)
        if token.isdigit():
            digit_run.append(token)
            index += 1
            continue

        if token in _DECADES:
            flush()
            output.append(_DECADES[token])
            index += 1
            continue

        if token in _UNITS:
            digit_run.append(str(_UNITS[token]))
            index += 1
            continue

        if token in _TENS:
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following in _UNITS and _UNITS[following] != 0:
                flush()
                output.append(str(_TENS[token] + _UNITS[following]))
                index += 2
            else:
                flush()
                output.append(str(_TENS[token]))
                index += 1
            continue

        if token in _TEENS:
            flush()
            output.append(str(_TEENS[token]))
            index += 1
            continue

        if token in _ORDINALS:
            flush()
            output.append(str(_ORDINALS[token]))
            index += 1
            continue

        if token in _FILLER and digit_run:
            # Only inside a number. A dangling "and" at the end of a run is a
            # real word and is restored below.
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following in _UNITS or following.isdigit():
                index += 1
                continue

        flush()
        output.append(token)
        index += 1

    flush()
    return output


def normalise_numbers(text: str) -> str:
    """Lowercase, punctuation-free text with spoken numbers folded to numerals."""

    from .audio_benchmark import normalise_text

    return " ".join(_fold_numbers(normalise_text(text).split()))


# --------------------------------------------------------------------------
# Metrics computed from stored transcripts
# --------------------------------------------------------------------------


def _levenshtein_words(reference: str, hypothesis: str) -> dict[str, Any]:
    import difflib

    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return {"reference_words": 0, "errors": len(hyp),
                "wer": float(len(hyp) > 0)}
    substitutions = deletions = insertions = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, ref, hyp, autojunk=False
    ).get_opcodes():
        if tag == "replace":
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "insert":
            insertions += j2 - j1
    errors = substitutions + deletions + insertions
    return {
        "reference_words": len(ref),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "wer": errors / len(ref),
    }


def number_normalised_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """WER after folding spoken numbers to numerals on **both** sides."""

    return _levenshtein_words(
        normalise_numbers(reference), normalise_numbers(hypothesis)
    )


#: A run of this many digits is an identifier, not a quantity. Same threshold
#: the deployment's structured recognizer uses (``phone_min_digits``).
IDENTIFIER_DIGITS = 7


def identifier_digits(text: str) -> list[str]:
    """Every long digit run in the number-normalised text, in order."""

    return [
        run for run in re.findall(r"\d+", normalise_numbers(text))
        if len(run) >= IDENTIFIER_DIGITS
    ]


def digit_preservation(reference: str, hypothesis: str) -> dict[str, Any] | None:
    """Did the long numbers in this clip come through digit for digit?

    The one thing this device must never round off. ``None`` when the clip
    carries no identifier-scale number, so clips without one do not dilute the
    rate.
    """

    expected = identifier_digits(reference)
    if not expected:
        return None
    produced = identifier_digits(hypothesis)
    remaining = list(produced)
    exact = 0
    for value in expected:
        if value in remaining:
            remaining.remove(value)
            exact += 1
    return {
        "expected": expected,
        "produced": produced,
        "identifiers": len(expected),
        "exact": exact,
        "accuracy": exact / len(expected),
    }


# --------------------------------------------------------------------------
# Run folders
# --------------------------------------------------------------------------


def engine_of(config: Mapping[str, Any]) -> str:
    """Which recognizer produced a run.

    Runs written before V4 carry no ``stt_engine`` key. Every one of them is a
    Canary run - the key was added in the same change that replaced Canary - so
    its absence is the identification, not a guess.
    """

    return str(config.get("stt_engine", "canary")).lower()


def pipeline_of(config: Mapping[str, Any]) -> str:
    return str(config.get("pipeline_version", "V3"))


def read_predictions(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / "audio_predictions.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _recorded_model(directory: Path) -> str | None:
    path = Path(directory) / "audio_stt_metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model")
    except (OSError, json.JSONDecodeError):
        return None


def score_run(directory: Path) -> dict[str, Any] | None:
    """Recompute the engine-comparable metrics for one saved run folder.

    Reads only. Nothing here loads a model, which is what makes it safe to point
    at the frozen Canary folders.
    """

    directory = Path(directory)
    config_path = directory / "benchmark_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [
        row for row in read_predictions(directory)
        if row.get("stt_transcript") and row.get("reference_transcript")
    ]
    if not rows:
        return None

    per_clip: list[dict[str, Any]] = []
    for row in rows:
        reference = row["reference_transcript"]
        hypothesis = row["stt_transcript"]
        normalised = number_normalised_wer(reference, hypothesis)
        digits = digit_preservation(reference, hypothesis)
        per_clip.append({
            "clip_id": row["clip_id"],
            "subset": row.get("subset"),
            "audio_duration_s": row.get("audio_duration_s"),
            "raw_wer": row.get("wer"),
            "normalised_wer": normalised["wer"],
            "normalised_errors": normalised["errors"],
            "normalised_reference_words": normalised["reference_words"],
            "digit_identifiers": None if digits is None else digits["identifiers"],
            "digit_exact": None if digits is None else digits["exact"],
            "stt_latency_ms": row.get("stt_latency_ms"),
            "stt_real_time_factor": row.get("stt_real_time_factor"),
            "pipeline_real_time_factor": row.get("pipeline_real_time_factor"),
            "end_to_end_latency_ms": row.get("end_to_end_latency_ms"),
        })

    def aggregate(selected: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        selected = list(selected)
        if not selected:
            return {"clips": 0}
        errors = sum(item["normalised_errors"] for item in selected)
        words = sum(item["normalised_reference_words"] for item in selected)
        raw = [item["raw_wer"] for item in selected if item["raw_wer"] is not None]
        return {
            "clips": len(selected),
            "reference_words": words,
            "aggregate_wer_number_normalised": errors / words if words else 0.0,
            "mean_wer_number_normalised": statistics.fmean(
                item["normalised_wer"] for item in selected
            ),
            "median_wer_number_normalised": statistics.median(
                item["normalised_wer"] for item in selected
            ),
            "aggregate_wer_raw": (
                sum(item["raw_wer"] * item["normalised_reference_words"]
                    for item in selected) / words if words and raw else None
            ),
            "mean_wer_raw": statistics.fmean(raw) if raw else None,
        }

    digit_clips = [item for item in per_clip if item["digit_identifiers"]]
    identifiers = sum(item["digit_identifiers"] for item in digit_clips)
    exact = sum(item["digit_exact"] for item in digit_clips)

    def summarise(values: Iterable[float | None]) -> dict[str, Any] | None:
        numbers = [float(value) for value in values if value is not None]
        if not numbers:
            return None
        ordered = sorted(numbers)
        return {
            "count": len(ordered),
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "min": ordered[0],
            "max": ordered[-1],
        }

    subsets = sorted({item["subset"] for item in per_clip if item["subset"]})
    return {
        "run_id": directory.name,
        "directory": str(directory),
        "engine": engine_of(config),
        "pipeline_version": pipeline_of(config),
        # V4 records the model in the config. The frozen Canary runs predate that
        # key and name their model in the STT metrics instead, so the table shows
        # what each run actually used rather than a blank for the older ones.
        "stt_model": config.get("stt_model") or _recorded_model(directory),
        "device_label": config.get("device_label"),
        "mode": config.get("mode"),
        "clips": len(per_clip),
        "overall": aggregate(per_clip),
        "by_subset": {
            name: aggregate(item for item in per_clip if item["subset"] == name)
            for name in subsets
        },
        "digit_preservation": {
            "clips_with_identifiers": len(digit_clips),
            "identifiers": identifiers,
            "exact": exact,
            "accuracy": exact / identifiers if identifiers else None,
            "note": (
                "A long number transcribed digit for digit. Scored after folding "
                "spoken number words to numerals, so an engine writing "
                "'five five five' and one writing '555' are judged on the value, "
                "not the spelling."
            ),
        },
        # Neither engine can be scored on wake-word recognition here: no clip in
        # the committed corpus says the device's name. Stated rather than
        # silently omitted, so nobody reads its absence as a zero.
        "wake_word": {
            "measurable": False,
            "reason": (
                "no clip in the audio benchmark corpus addresses the device by "
                "name, so wake-word recognition is not measurable from these "
                "artifacts for either engine"
            ),
        },
        "stt_latency_ms": summarise(item["stt_latency_ms"] for item in per_clip),
        "stt_real_time_factor": summarise(
            item["stt_real_time_factor"] for item in per_clip
        ),
        "pipeline_real_time_factor": summarise(
            item["pipeline_real_time_factor"] for item in per_clip
        ),
        "end_to_end_latency_ms": summarise(
            item["end_to_end_latency_ms"] for item in per_clip
        ),
        "per_clip": per_clip,
    }


def score_runs(directories: Iterable[Path]) -> list[dict[str, Any]]:
    scored = [score_run(Path(directory)) for directory in directories]
    return [item for item in scored if item]


def paired_clip_rows(left: Mapping[str, Any], right: Mapping[str, Any]
                     ) -> list[dict[str, Any]]:
    """Clips both runs scored, so a per-clip comparison has no ragged edges."""

    right_by_clip = {item["clip_id"]: item for item in right["per_clip"]}
    rows = []
    for item in left["per_clip"]:
        other = right_by_clip.get(item["clip_id"])
        if other is None:
            continue
        rows.append({
            "clip_id": item["clip_id"],
            "subset": item["subset"],
            "audio_duration_s": item["audio_duration_s"],
            f"{left['engine']}_wer": item["normalised_wer"],
            f"{right['engine']}_wer": other["normalised_wer"],
            f"{left['engine']}_stt_rtf": item["stt_real_time_factor"],
            f"{right['engine']}_stt_rtf": other["stt_real_time_factor"],
            f"{left['engine']}_stt_latency_ms": item["stt_latency_ms"],
            f"{right['engine']}_stt_latency_ms": other["stt_latency_ms"],
        })
    return rows


# --------------------------------------------------------------------------
# The one end-to-end privacy signal that survives a change of recognizer
# --------------------------------------------------------------------------
#
# The benchmark's ``entity_recall_after_stt`` family follows each gold entity
# from the reference transcript into the recognized one by aligning words. That
# alignment is sound when both sides spell numbers the same way and it breaks
# when they do not: Whisper writes "555-0147." where the reference says "five
# five five, zero one four seven", the seven reference tokens match nothing, and
# the entity is booked as "altered by the recognizer" even in clips where the
# released text shows it was masked perfectly. Those figures are therefore valid
# within one engine and are **not** comparable across engines on this corpus.
#
# Recomputing that family for the frozen Canary runs is not possible from what
# they saved: the predicted character spans were never written to disk, only the
# per-entity outcomes derived from them, and rerunning Canary is out of the
# question.
#
# What every run did save is the text the device would actually have released.
# Asking whether an identifier-scale number is still legible in it needs no
# alignment, no gold offsets and no model - and it is the question the product
# claim rests on.


def released_identifier_leak(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Is an identifier-scale number still readable in this clip's output?

    Scored on ``final_masked_output``, which is what the deployment puts on
    screen and into storage - after masking *and* after the spoken-digit
    renderer, which is the last stage able to put a number back.

    Both spellings count. A number left standing as digit words is folded to
    numerals first, so "five five five zero one four seven" surviving the masker
    is the same leak as "5550147" surviving it.

    ``None`` for a clip whose reference carries no identifier-scale number,
    which keeps clips with nothing to protect out of the denominator.
    """

    reference = row.get("reference_transcript") or ""
    released = row.get("final_masked_output")
    if released is None:
        return None
    expected = identifier_digits(reference)
    if not expected:
        return None
    leaked = identifier_digits(released)
    return {
        "clip_id": row.get("clip_id"),
        "expected": expected,
        "leaked": leaked,
        "clean": not leaked,
    }


def released_identifier_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [item for item in (released_identifier_leak(row) for row in rows) if item]
    clean = sum(1 for item in scored if item["clean"])
    return {
        "definition": (
            "clips whose reference carries a 7+ digit identifier, scored on "
            "whether any identifier-scale number is still legible in the text "
            "the device released - after masking and after the spoken-digit "
            "renderer. Needs no alignment against the reference, so it is "
            "comparable across recognizers."
        ),
        "clips_with_identifiers": len(scored),
        "clips_released_clean": clean,
        "clips_leaking_an_identifier": len(scored) - clean,
        "clean_rate": clean / len(scored) if scored else None,
        "leaks": [item for item in scored if not item["clean"]],
    }
