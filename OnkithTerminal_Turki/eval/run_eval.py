"""Score a recogniser on the things this device actually needs from one.

Word error rate alone would have said Moonshine was fine. Measured on
ordinary speech it scores 1.9%, which is excellent, and the device still
never once answered to its own name. The metrics here are the failure
modes that were actually observed, in the order they matter:

  wake recall     of the clips that called it, how many woke it
  false wake      of the clips that did not, how many woke it anyway
  intent accuracy did it route to the lane the words asked for
  entities kept   did every digit of a phone number survive verbatim
  WER             ordinary transcription quality, split by wake and quiet
  RTF             seconds of compute per second of audio

Wake recall and false wake are separate on purpose. One number hides the
trade: a recogniser can take either to 100% by ruining the other.

Usage:
    .venv/bin/python eval/run_eval.py                      the two shipped engines
    .venv/bin/python eval/run_eval.py --engine whisper     just one
    .venv/bin/python eval/run_eval.py --no-record          do not append to history

Canary is asked for by name and never runs by default. It needs torch and
NeMo, which this project does not install, so it is scored from a separate
interpreter:

    ~/.onkith-canary-eval/.venv/bin/python eval/run_eval.py --engine canary
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "corpus" / "manifest.json"
HISTORY = HERE / "results" / "history.jsonl"


def normalise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over words. Insertions count, so it can exceed 1.0."""

    reference_words, hypothesis_words = normalise(reference), normalise(hypothesis)
    if not reference_words:
        return 0.0
    previous = list(range(len(hypothesis_words) + 1))
    for i, want in enumerate(reference_words, start=1):
        current = [i]
        for j, got in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (want != got),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


@dataclass
class Tally:
    woke_when_called: int = 0
    called: int = 0
    woke_uncalled: int = 0
    uncalled: int = 0
    intent_right: int = 0
    intent_total: int = 0
    entities_kept: int = 0
    entities_total: int = 0
    wake_wer: list[float] = field(default_factory=list)
    quiet_wer: list[float] = field(default_factory=list)
    audio_seconds: float = 0.0
    compute_seconds: float = 0.0
    misses: list[str] = field(default_factory=list)

    def plus(self, other: "Tally") -> "Tally":
        """Add two splits together without scoring the audio twice."""

        return Tally(
            woke_when_called=self.woke_when_called + other.woke_when_called,
            called=self.called + other.called,
            woke_uncalled=self.woke_uncalled + other.woke_uncalled,
            uncalled=self.uncalled + other.uncalled,
            intent_right=self.intent_right + other.intent_right,
            intent_total=self.intent_total + other.intent_total,
            entities_kept=self.entities_kept + other.entities_kept,
            entities_total=self.entities_total + other.entities_total,
            wake_wer=self.wake_wer + other.wake_wer,
            quiet_wer=self.quiet_wer + other.quiet_wer,
            audio_seconds=self.audio_seconds + other.audio_seconds,
            compute_seconds=self.compute_seconds + other.compute_seconds,
            misses=self.misses + other.misses,
        )

    def as_row(self) -> dict:
        def rate(hit: int, total: int) -> float:
            return hit / total if total else 0.0

        mean = lambda values: sum(values) / len(values) if values else 0.0
        return {
            "wake_recall": rate(self.woke_when_called, self.called),
            "false_wake_rate": rate(self.woke_uncalled, self.uncalled),
            "intent_accuracy": rate(self.intent_right, self.intent_total),
            "entities_kept": rate(self.entities_kept, self.entities_total),
            "wer_wake": mean(self.wake_wer),
            "wer_quiet": mean(self.quiet_wer),
            "rtf": (
                self.compute_seconds / self.audio_seconds if self.audio_seconds else 0.0
            ),
            "counts": {
                "woke_when_called": f"{self.woke_when_called}/{self.called}",
                "woke_uncalled": f"{self.woke_uncalled}/{self.uncalled}",
                "intent_right": f"{self.intent_right}/{self.intent_total}",
                "entities_kept": f"{self.entities_kept}/{self.entities_total}",
            },
        }


def read_clip(path: Path):
    import numpy as np

    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def engines_for(names: list[str] | None):
    """Build only the engines that were asked for.

    Every adapter is imported inside its own factory rather than at the
    top. The engines do not share a dependency set: Canary needs torch and
    NeMo, which this project refuses to install, so it is scored from a
    separate interpreter. Importing all three eagerly meant the one engine
    that could not load took the other two down with it.
    """

    from integrated_pipeline.config import PipelineConfig

    config = PipelineConfig()

    def moonshine():
        from integrated_pipeline.moonshine_adapter import MoonshineAdapter

        return MoonshineAdapter(config)

    def whisper():
        from integrated_pipeline.whisper_adapter import WhisperAdapter

        return WhisperAdapter(config)

    def canary():
        from eval.canary_engine import CanaryEngine

        return CanaryEngine(config)

    available = {"moonshine": moonshine, "whisper": whisper, "canary": canary}
    # Canary is left out of a bare run on purpose. It cannot load in this
    # interpreter, and a default that always reports one engine as broken
    # trains everybody to ignore the output.
    chosen = names or ["moonshine", "whisper"]
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise SystemExit(f"Unknown engine {unknown[0]}. Have: {', '.join(available)}")
    return config, [(name, available[name]) for name in chosen]


def split_of(clip: dict) -> str:
    return "synthetic" if clip.get("source") == "synthetic" else "recorded"


def score(adapter, clips: list[dict], corpus: Path, config) -> dict[str, Tally]:
    """Score every clip once, kept apart by where the audio came from.

    The two halves are not the same evidence and averaging them together
    hides which one carried the result. One machine voice with no room is
    a floor on difficulty; the recorded half is what the device will
    actually be asked to do.
    """

    from app.onkith_music import (
        WakeWords,
        configure_wake_words,
        intercept_onkith_command,
    )
    from integrated_pipeline.schemas import RecordedUtterance

    configure_wake_words(WakeWords.from_config(config.wake))
    tallies = {"synthetic": Tally(), "recorded": Tally()}
    for index, clip in enumerate(clips, start=1):
        tally = tallies[split_of(clip)]
        samples, rate = read_clip(corpus / clip["file"])
        recording = RecordedUtterance(
            session_id="eval",
            batch_id=clip["id"],
            sequence_number=index,
            samples=samples,
            sample_rate=rate,
            audio_started_monotonic=0.0,
            audio_ended_monotonic=0.0,
            audio_started_at_utc="",
            audio_ended_at_utc="",
            captured_at_utc="",
            recording_closed_monotonic=time.monotonic(),
            endpoint_reason="endpoint_silence",
            reliable=True,
            dropped_audio_chunks=0,
            captured_audio_seconds=len(samples) / rate,
        )
        started = time.perf_counter()
        heard = (adapter.transcribe_recording(recording).final_text or "").strip()
        tally.compute_seconds += time.perf_counter() - started
        tally.audio_seconds += len(samples) / rate

        result = intercept_onkith_command(heard)
        error = word_error_rate(clip["transcript"], heard)

        if clip["expect_wake"]:
            tally.called += 1
            tally.wake_wer.append(error)
            if result.command_detected:
                tally.woke_when_called += 1
                if clip["expect_intent"]:
                    tally.intent_total += 1
                    if result.intent == clip["expect_intent"]:
                        tally.intent_right += 1
                    else:
                        tally.misses.append(
                            f"{clip['id']}: wanted {clip['expect_intent']}, "
                            f"got {result.intent} from {heard!r}"
                        )
            else:
                # Not woken at all, so the lane it would have taken is
                # counted as wrong rather than quietly left out.
                if clip["expect_intent"]:
                    tally.intent_total += 1
                tally.misses.append(f"{clip['id']}: never woke, heard {heard!r}")
        else:
            tally.uncalled += 1
            tally.quiet_wer.append(error)
            if result.command_detected:
                tally.woke_uncalled += 1
                tally.misses.append(f"{clip['id']}: woke uncalled, heard {heard!r}")

        for wanted in clip.get("must_contain", []):
            tally.entities_total += 1
            if kept_verbatim(wanted, heard):
                tally.entities_kept += 1
            else:
                tally.misses.append(f"{clip['id']}: lost {wanted!r} in {heard!r}")
    return tallies


def kept_verbatim(wanted: str, heard: str) -> bool:
    """Did every digit survive, whatever the recogniser did to the spacing.

    Both engines write long numbers with separators, so a plain substring
    test fails 1,098,765,432 against 1098765432 and calls a correct
    transcript a loss. The separators are a rendering choice and the
    masker sees through them. An inserted or dropped digit is not, and
    that is the failure this exists to catch: 0501234567 came back once
    as 05001234567, which is a phone number nobody has.
    """

    if wanted.lower() in heard.lower():
        return True
    digits = re.sub(r"\D", "", wanted)
    return bool(digits) and digits in re.sub(r"\D", "", heard)


def describe(name: str, adapter, config) -> str:
    if name == "canary":
        from eval.canary_engine import DECODER, boost_alpha

        alpha = boost_alpha()
        boost = f"/boost{alpha:g}" if alpha else ""
        return f"180m-flash/fp32/beam{DECODER['beam_size']}{boost}"
    if name == "whisper":
        settings = config.stt.get("whisper", {})
        return (
            f"{settings.get('model', 'base.en')}/"
            f"{settings.get('compute_type', 'int8')}/"
            f"beam{settings.get('beam_size', 5)}"
        )
    return f"{config.moonshine['model_arch']}/{config.moonshine['package_version']}"


TABLE_HEADER = (
    f"{'engine':<34} {'wake':>7} {'false':>7} {'intent':>7} "
    f"{'digits':>7} {'WER/w':>7} {'WER/q':>7} {'RTF':>6}"
)


def format_row(label: str, row: dict) -> str:
    return (
        f"{label:<34} "
        f"{100 * row['wake_recall']:6.0f}% "
        f"{100 * row['false_wake_rate']:6.0f}% "
        f"{100 * row['intent_accuracy']:6.0f}% "
        f"{100 * row['entities_kept']:6.0f}% "
        f"{100 * row['wer_wake']:6.1f}% "
        f"{100 * row['wer_quiet']:6.1f}% "
        f"{row['rtf']:5.2f}x"
    )


def git_revision() -> str:
    """The commit these numbers came from, marked if it was not only that.

    A score is only reproducible if the revision beside it is the code that
    produced it. The best measurement in this file so far was taken with
    the change still uncommitted, and the row says the commit before it,
    which reads as that commit being better than it was. `+dirty` says the
    working tree had something in it that the commit does not.

    Only this project is looked at, and the results directory is left out
    of the check. A run appends to the history file, so counting that as a
    change would mark every run after the first one dirty, and a flag that
    is always on is the same as no flag.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        revision = git("rev-parse", "--short", "HEAD")
    except Exception:
        return "unknown"
    try:
        changed = git("status", "--porcelain", "--", ".", ":!eval/results")
    except Exception:
        return revision
    return f"{revision}+dirty" if changed else revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="append")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    corpus = MANIFEST.parent
    config, engines = engines_for(args.engine)

    recorded = sum(1 for clip in clips if clip.get("source") != "synthetic")
    print(
        f"corpus: {len(clips)} clips "
        f"({len(clips) - recorded} synthetic, {recorded} recorded)\n"
    )

    scored = []
    for name, build in engines:
        adapter = build()
        adapter.load()
        tallies = score(adapter, clips, corpus, config)
        adapter.close()
        scored.append((f"{name} {describe(name, adapter, config)}", name, tallies))

    # Recorded first and on its own, because it is the half that decides.
    # Printing one blended average would let 24 clips of clean machine
    # speech carry a recogniser that cannot hear a person in a room.
    captions = {
        "recorded": f"recorded ({recorded} clips, real voice, real room) DECIDES",
        "synthetic": (
            f"synthetic ({len(clips) - recorded} clips, one machine voice, "
            "no room) floor on difficulty"
        ),
        "overall": f"overall ({len(clips)} clips)",
    }
    rows = []
    for split in ("recorded", "synthetic", "overall"):
        present = [
            (
                label,
                name,
                (
                    tallies["recorded"].plus(tallies["synthetic"])
                    if split == "overall"
                    else tallies[split]
                ),
            )
            for label, name, tallies in scored
        ]
        if not any(t.called or t.uncalled for _, _, t in present):
            continue
        print(captions[split])
        print(TABLE_HEADER)
        print("-" * len(TABLE_HEADER))
        for label, name, tally in present:
            row = tally.as_row()
            print(format_row(label, row))
            rows.append(
                {
                    "engine": name,
                    "detail": label,
                    "split": split,
                    **row,
                    "misses": tally.misses,
                }
            )
            if args.verbose and split != "overall":
                for miss in tally.misses:
                    print(f"    {miss}")
        print()

    print("wake/intent/digits: higher is better. false/WER/RTF: lower is better.")

    if not args.no_record:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "revision": git_revision(),
            "corpus_clips": len(clips),
            "corpus_recorded": recorded,
            "platform": f"{sys.platform}",
            "results": rows,
        }
        with HISTORY.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"appended to {HISTORY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
