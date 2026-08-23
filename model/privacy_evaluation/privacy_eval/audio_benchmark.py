"""End-to-end deployment benchmark: audio in, sanitized text out.

The large text evaluations answer "is the privacy model good?". This one answers
a different question, and it is the question a deployed device actually poses:

    Does private information stay protected after imperfect speech recognition?

So it starts from committed WAV files and runs the real production stack -

    audio -> Silero VAD -> Parakeet STT -> transcript merge / wake-word
          -> inactivity buffer -> DeBERTa-v3-xsmall INT8 -> masking
          -> spoken-digit rendering -> final sanitized text

- and measures every stage. Nothing is mocked. If a component cannot run on this
machine it is reported as unavailable, with the reason and the command that
fixes it; no stage is ever simulated to fill a gap.

Three arms, scored on the same clips
------------------------------------

``reference_text``
    The reference transcript, exactly as written, straight into the production
    masker. A text-side control with exact gold offsets, so it is directly
    comparable to the text benchmark's numbers.

``reference_uncased``
    The same transcript lowercased with punctuation blanked out. Under V3 this
    was what a perfect recognizer would hand over, because Canary ran with
    ``pnc: no``. Whisper punctuates and capitalises, so under V4 this arm is no
    longer the recognizer's own output shape - it is kept unchanged because it
    is the arm every Canary run was scored against, and dropping it would end
    the comparison. The gap between it and ``reference_text`` is still the cost
    of losing case and punctuation alone, with no recognition error involved.

``stt``
    The real recognized transcript through the real pipeline. The gap between
    this arm and ``reference_uncased`` is the cost of recognition errors alone.

That decomposition is what lets a failure be attributed to the speech recognizer
or to the privacy model instead of being reported as one undifferentiated
end-to-end number.

Scope
-----

At most a few dozen clips. This is a **controlled deployment benchmark and a
Raspberry Pi feasibility test**, not a comprehensive speech-recognition
benchmark, and it does not replace the held-out test set, the combined
validation, ``ood_dev`` or the frozen ``ood_final``. Summaries generated here
say so.

Transcripts *are* written to the result files, unlike the text benchmark. The
privacy clips contain only fictional information generated for this project and
the public clips are LibriSpeech dev-clean, so there is nothing to withhold -
and "what did the recognizer actually produce" is one of the questions.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import benchmark_common as common
from .benchmark_common import RunDirectory, WarningLog, print_summary, summarise

common.ensure_import_paths()

FIXTURE_DIR = common.EVALUATION_ROOT / "fixtures" / "audio_benchmark"
MANIFEST_PATH = FIXTURE_DIR / "audio_benchmark.manifest.json"

#: A clip's entity counts as surviving recognition when at least this share of
#: its words came through. Below it, the surface the listener would read is not
#: the entity any more, and the failure belongs to the recognizer.
PRESERVATION_THRESHOLD = 0.5
#: Character coverage at or above this counts as fully masked. Exactly 1.0 would
#: make a single trailing space a failure.
FULL_MASK_THRESHOLD = 0.999

#: Which definition of the per-entity outcomes a run folder holds, stamped into
#: ``benchmark_config.json``.
#:
#: ``2026-08-20.number-folded-trimmed``
#:     ``locate_entity`` aligns both sides on number-folded tokens and trims the
#:     located region to its alphanumeric core. Before this, a reference reading
#:     "twenty seven" matched nothing in a transcript reading "27", and a masker
#:     that left the full stop of "Carter." standing was scored at 6/7. Both
#:     misattributed Whisper output; Canary emitted neither numerals nor
#:     punctuation, so V3's figures barely move under it.
#:
#: A folder written before the fix carries no key at all. ``privacy_eval.rescore``
#: recomputes one to this definition from what the folder saved.
ENTITY_SCORING_VERSION = "2026-08-20.number-folded-trimmed"

# Identifier scale, borrowed from the deployment's own structured recognizer
# (``StructuredPIIConfig.phone_min_digits``): a run of this many digits is a
# telephone number, an account or a card, not a quantity.
IDENTIFIER_DIGITS = 7

WARMUP_TEXT = (
    "Hi, my name is Sarah Walusiak, my number is 555 0147 and my email is "
    "sarah.walusiak@example.org. Remind me about the appointment on 12 March."
)

STT_UNAVAILABLE_HELP = """\
The audio benchmark needs the production speech recognizer. Install it, then run
this script again - nothing else changes:

    pip install -r FullPipeline_V5/requirements-windows.txt   (or -rpi.txt)

V5 runs NVIDIA Parakeet 110M through NeMo's native TDT/RNNT branch on torch CPU,
so the install is torch + nemo_toolkit[asr]. The pinned 438 MiB checkpoint is
fetched and byte-verified once by the platform setup script, or directly with:

    python -c "from integrated_pipeline.parakeet_adapter import \
cache_parakeet_checkpoint; print(cache_parakeet_checkpoint())"

Until then the control arms below still run, so the fixtures, the privacy model
and the memory profile are all validated - only the recognizer is missing."""


class FixtureRejected(RuntimeError):
    """A committed clip is missing or does not match its manifest hash."""


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FixtureRejected(
            f"audio benchmark manifest is missing: {path}. Rebuild the fixtures with "
            "`python model/privacy_evaluation/scripts/build_audio_benchmark.py` on a "
            "workstation, then commit the clips and the manifest."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_clip(clip: Mapping[str, Any], *, fixture_dir: Path = FIXTURE_DIR) -> Path:
    """Prove one WAV is the file the manifest describes before it is measured."""

    path = fixture_dir / clip["audio_path"]
    if not path.is_file():
        raise FixtureRejected(f"clip {clip['clip_id']} is missing: {path}")
    digest = common.sha256_file(path)
    if digest != clip["sha256"]:
        raise FixtureRejected(
            f"clip {clip['clip_id']} SHA-256 {digest} does not match the manifest "
            f"{clip['sha256']}. WAV files are marked binary in .gitattributes; a "
            "mismatch means the file was rewritten, not merely checked out."
        )
    return path


def read_wav(path: Path):
    """One 16 kHz mono float32 waveform. stdlib only, so the Pi needs no codec."""

    import numpy as np

    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise ValueError(f"{path.name} is not mono")
        if stream.getsampwidth() != 2:
            raise ValueError(f"{path.name} is not 16-bit PCM")
        if stream.getframerate() != common.SAMPLE_RATE:
            raise ValueError(
                f"{path.name} is {stream.getframerate()} Hz; the pipeline requires "
                f"exactly {common.SAMPLE_RATE} Hz"
            )
        raw = stream.readframes(stream.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------
# Text normalisation, word error rate, entity alignment
# --------------------------------------------------------------------------

_APOSTROPHES = "'‘’ʼ"
_TOKEN = re.compile(r"[0-9a-z]+")


def normalise_text(text: str) -> str:
    """Casing and punctuation removed, the way both sides of a WER are compared.

    Apostrophes are deleted rather than blanked, so ``quilter's`` and
    ``quilters`` are one token either way; every other non-alphanumeric
    character becomes a space.
    """

    lowered = text.casefold()
    for mark in _APOSTROPHES:
        lowered = lowered.replace(mark, "")
    return " ".join(_TOKEN.findall(lowered))


def uncased_unpunctuated(text: str) -> str:
    """Lowercase, punctuation blanked to spaces, **length preserved**.

    Preserving length is the point: the gold spans measured against the original
    transcript stay valid character-for-character, so this arm can be scored by
    the same ``evaluate_spans`` the text benchmark uses.
    """

    return "".join(
        character.casefold() if character.isalnum() or character.isspace() else " "
        for character in text
    )


def tokenise_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """``(normalised token, start, end)`` with offsets into the original string."""

    tokens: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S+", text):
        normalised = normalise_text(match.group())
        if normalised:
            tokens.append((normalised, match.start(), match.end()))
    return tokens


def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Levenshtein over normalised words: the standard definition, nothing else."""

    ref = normalise_text(reference).split()
    hyp = normalise_text(hypothesis).split()
    if not ref:
        return {"reference_words": 0, "hypothesis_words": len(hyp), "substitutions": 0,
                "deletions": 0, "insertions": len(hyp), "errors": len(hyp),
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
        "hypothesis_words": len(hyp),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "wer": errors / len(ref),
    }


def _spoken_form_rules() -> tuple[str, ...]:
    """The rewrite rules the deployment applies before detection, by name.

    Imported here rather than at module scope: the pipeline package is only on
    ``sys.path`` once ``benchmark_common.ensure_import_paths`` has run.
    """

    from integrated_pipeline.spoken_form_normalizer import rules

    return tuple(rules())


def _covered_characters(spans: Iterable[Mapping[str, Any]]) -> set[int]:
    covered: set[int] = set()
    for span in spans:
        covered.update(range(int(span["start"]), int(span["end"])))
    return covered


def _informative_characters(text: str) -> int:
    """How many characters of a region can carry private information.

    The single weighting rule for every character-level number in this
    benchmark. ``mask_coverage`` already excludes whitespace from each entity's
    coverage, so the aggregate that averages those coverages has to weight by
    the same characters, or the control arms and the end-to-end arm end up
    averaging the same per-entity values against different denominators.
    """

    return sum(1 for character in text if not character.isspace())


def reconstructed_digit_runs(masked_text: str, final_text: str,
                             *, minimum_digits: int = IDENTIFIER_DIGITS
                             ) -> list[dict[str, Any]]:
    """Numerals the render step built out of text the masker left standing.

    The deployment renders spoken numbers as numerals for display
    (``app/digit_normalizer.py``), and that step runs *after* masking by
    design. When the privacy layer masks the number, the placeholder has no
    digit words left to render and nothing happens here. When it misses one,
    the renderer turns "five five five zero one four seven" into "5550147" -
    the same leaked information, now in the compact form an identifier is
    normally read, copied and matched in.

    Only runs absent from the masked text are reported, so a number the
    recognizer already produced as digits is never counted: the question is
    what postprocessing *added*. ``minimum_digits`` follows the deployment's
    own notion of identifier scale (``StructuredPIIConfig.phone_min_digits``).
    """

    existing = set(re.findall(r"\d+", masked_text))
    runs = []
    for match in re.finditer(r"\d+", final_text):
        value = match.group(0)
        if len(value) >= minimum_digits and value not in existing:
            runs.append({
                "value": value,
                "digits": len(value),
                "start": match.start(),
                "end": match.end(),
            })
    return runs


def _fold_tokens(tokens: Sequence[tuple[str, int, int]]
                 ) -> list[tuple[str, int, int]]:
    """``tokenise_with_offsets`` output with spoken numbers folded to numerals.

    The character offsets are carried across the fold, so a folded token still
    points at the text that produced it: "five five five zero one four seven"
    becomes one ``5550147`` token spanning all seven words.
    """

    from .stt_comparison import fold_numbers_with_indices

    return [
        (value, tokens[first][1], tokens[last - 1][2])
        for value, first, last in fold_numbers_with_indices(
            [token for token, _, _ in tokens]
        )
    ]


def informative_bounds(text: str, start: int | None, end: int | None
                       ) -> tuple[int | None, int | None]:
    """Shrink a located region to the characters that can carry information.

    ``tokenise_with_offsets`` splits on whitespace, so its tokens carry whatever
    punctuation was attached to the word: the region located for ``Carter`` in
    "My name is Alex Carter." is ``Carter.``, full stop included. A masker
    replaces the name and leaves the sentence punctuation exactly where it was,
    which is correct, and measuring coverage over the untrimmed region booked
    that as a partial failure at six characters out of seven. Canary ran with
    ``pnc: no`` and emitted no punctuation at all, so the defect only became
    visible under Whisper.

    Only the edges are trimmed. Punctuation *inside* an entity belongs to it -
    the dots in ``alex.carter@example.com``, the hyphen in ``555-0147`` - and
    stays in the region and in the denominator.
    """

    if start is None or end is None or end <= start:
        return start, end
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return (start, end) if end > start else (None, None)


def locate_entity(reference: str, span: Mapping[str, Any], hypothesis: str) -> dict[str, Any]:
    """Follow one gold entity from the reference transcript into the recognized one.

    A recognizer inserts, drops and substitutes words, so an entity's offsets do
    not survive. The two token sequences are aligned once, and the entity's
    reference tokens are carried across that alignment. Tokens the alignment
    matched are the part of the entity that survived; where nothing matched, the
    region the recognizer produced *instead* is returned, because that is the
    text the privacy model actually had to mask.

    Both sides are aligned on number-folded tokens. The reference transcripts
    write numbers the way they are spoken and Whisper writes numerals, so
    without the fold "twenty seven" and "27" share no token, an entity that came
    through the recognizer intact is booked as destroyed by it, and the region
    located for the entity excludes the very numeral it became. Canary wrote
    numbers as words, so the fold changes almost nothing for the V3 runs; it is
    what makes the two engines comparable. The located region is then trimmed to
    its informative core - see ``informative_bounds``.
    """

    ref_tokens = _fold_tokens(tokenise_with_offsets(reference))
    hyp_tokens = _fold_tokens(tokenise_with_offsets(hypothesis))
    entity_indices = [
        index for index, (_, start, end) in enumerate(ref_tokens)
        if start < int(span["end"]) and end > int(span["start"])
    ]
    if not entity_indices:
        return {"entity_tokens": 0, "preserved_tokens": 0, "preservation_ratio": 0.0,
                "hypothesis_char_start": None, "hypothesis_char_end": None,
                "hypothesis_surface": "", "alignment": "no_reference_tokens"}

    opcodes = difflib.SequenceMatcher(
        None, [token for token, _, _ in ref_tokens], [token for token, _, _ in hyp_tokens],
        autojunk=False,
    ).get_opcodes()

    mapped: list[int] = []
    fallback: list[int] = []
    for tag, i1, i2, j1, j2 in opcodes:
        overlap = [index for index in entity_indices if i1 <= index < i2]
        if not overlap:
            continue
        if tag == "equal":
            mapped.extend(j1 + (index - i1) for index in overlap)
        else:
            # 'replace' hands back what the recognizer produced instead;
            # 'delete' contributes an empty region at the join.
            fallback.extend(range(j1, j2))

    chosen = mapped or fallback
    if chosen:
        first, last = min(chosen), max(chosen)
        char_start, char_end = informative_bounds(
            hypothesis, hyp_tokens[first][1], hyp_tokens[last][2]
        )
    else:
        char_start = char_end = None

    return {
        "entity_tokens": len(entity_indices),
        "preserved_tokens": len(mapped),
        "preservation_ratio": len(mapped) / len(entity_indices),
        "hypothesis_char_start": char_start,
        "hypothesis_char_end": char_end,
        "hypothesis_surface": (
            hypothesis[char_start:char_end] if char_start is not None else ""
        ),
        "alignment": "matched" if mapped else ("substituted" if fallback else "deleted"),
    }


def mask_coverage(text: str, start: int | None, end: int | None,
                  covered: set[int]) -> float:
    """Share of an entity's *informative* characters that a prediction covers.

    Whitespace is excluded. When the model emits two adjacent spans - "ten past"
    as TIME and "eleven" as DATE - the space between them belongs to no span,
    and counting it as leaked would turn a correct masking into a partial
    failure. No private information lives in a space.
    """

    if start is None or end is None or end <= start:
        return 0.0
    informative = {index for index in range(start, end) if not text[index].isspace()}
    if not informative:
        return 0.0
    return len(informative & covered) / len(informative)


def classify_entity(location: Mapping[str, Any], coverage: float) -> str:
    """One of five outcomes, named for the component responsible.

    ``leaked_privacy_model`` and ``partially_masked`` are privacy-model defects:
    the entity was recognizable in the transcript and was not (fully) masked.
    The two ``stt_altered_`` outcomes mean the recognizer changed the entity
    beyond recognition first, which is a different failure with a different fix.
    """

    preserved = location["preservation_ratio"] >= PRESERVATION_THRESHOLD
    if preserved and coverage >= FULL_MASK_THRESHOLD:
        return "masked"
    if preserved and coverage > 0.0:
        return "partially_masked"
    if preserved:
        return "leaked_privacy_model"      # the entity was there, unmasked
    if coverage >= FULL_MASK_THRESHOLD:
        return "stt_altered_masked"        # recognizer changed it, masked anyway
    return "stt_altered_leaked"            # recognizer changed it, residual left over


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass
class AudioBenchmarkOptions:
    smoke_test: bool = False
    clips: tuple[str, ...] = ()
    subset: str = "all"
    device_label: str = ""
    run_stt: bool = True
    warmup_runs: int = 1
    results_root: Path | None = None
    manifest_path: Path = MANIFEST_PATH

    mode: str = field(init=False, default="full")

    def __post_init__(self) -> None:
        self.mode = "smoke" if self.smoke_test else "full"


def select_clips(manifest: Mapping[str, Any], options: AudioBenchmarkOptions
                 ) -> list[dict[str, Any]]:
    """Which clips this run scores. Deterministic under every combination."""

    clips = list(manifest["clips"])
    if options.subset != "all":
        clips = [clip for clip in clips if clip["subset"] == options.subset]
    if options.clips:
        wanted = set(options.clips)
        chosen = [clip for clip in clips if clip["clip_id"] in wanted]
        missing = wanted - {clip["clip_id"] for clip in chosen}
        if missing:
            raise FixtureRejected(f"unknown clip id(s): {sorted(missing)}")
        return chosen
    if not options.smoke_test:
        return clips

    # Smoke mode: the shortest clip of each kind, plus the densest privacy clip,
    # so every code path runs - hard negatives, multi-entity decode, both
    # subsets - in well under a minute.
    def shortest(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda clip: (clip["duration_seconds"], clip["clip_id"]))
        return [dict(ordered[0])] if ordered else []

    privacy = [c for c in clips if c["subset"] == "synthetic_privacy" and c["expected_private_spans"]]
    negatives = [c for c in clips if c["subset"] == "synthetic_privacy" and not c["expected_private_spans"]]
    public = [c for c in clips if c["subset"] == "public"]
    densest = sorted(
        privacy, key=lambda clip: (-len(clip["expected_private_spans"]), clip["clip_id"])
    )[:1]

    selected: dict[str, dict[str, Any]] = {}
    for group in (shortest(privacy), densest, shortest(negatives), shortest(public)):
        for clip in group:
            selected[clip["clip_id"]] = dict(clip)
    return [clip for clip in clips if clip["clip_id"] in selected]


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------


class AudioBenchmark:
    def __init__(self, options: AudioBenchmarkOptions) -> None:
        self.options = options
        self.warnings = WarningLog()
        self.completed: dict[str, bool] = {}
        self.started_utc = common.utc_now()
        # The engine is part of the run id, so a Parakeet run can never land
        # in - or be mistaken for - a Whisper or a Canary folder. The historical
        # Canary runs carry no engine token and are identified by its absence.
        self.run_id = common.make_run_id(
            self.started_utc, options.mode, options.device_label,
            prefix="audio_parakeet",
        )
        self.run = RunDirectory(self.run_id, options.results_root)

        self.manifest: dict[str, Any] = {}
        self.clips: list[dict[str, Any]] = []
        self.system_info: dict[str, Any] = {}
        self.artifact: dict[str, Any] = {}
        self.baseline_rss: int | None = None
        self.stage_rss: dict[str, int | None] = {}
        self.peak_rss = 0
        self.load_seconds: dict[str, float] = {}

        self.vad = None
        self.parakeet = None
        self.adapter = None
        self.stt_available = False
        self.stt_reason = ""
        self.stt_backend: dict[str, Any] | None = None

        self.clip_results: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        # Every spoken address form the pre-detection view rewrote, so a run
        # folder records what the detectors were shown and not only what the
        # recognizer produced.
        self.spoken_form_edits: list[dict[str, Any]] = []
        self.spoken_form_clips: list[str] = []
        self.failures: list[dict[str, Any]] = []
        self.control_metrics: dict[str, Any] = {}
        self.privacy_scored_clips: list[str] = []
        self.latency_summary: dict[str, Any] = {}
        self.stt_metrics: dict[str, Any] = {}
        self.privacy_metrics: dict[str, Any] = {}
        self.memory: dict[str, Any] = {}

    # -- setup -------------------------------------------------------------

    def announce(self) -> None:
        self.manifest = load_manifest(self.options.manifest_path)
        self.clips = select_clips(self.manifest, self.options)
        if not self.clips:
            raise FixtureRejected("no clips selected")

        self.system_info = common.collect_system_info(self.run_id, self.started_utc)
        self.baseline_rss = self.system_info["baseline_process_rss_bytes"]
        self.peak_rss = self.baseline_rss or 0
        self.stage_rss["baseline"] = self.baseline_rss

        from integrated_pipeline.config import PipelineConfig

        self.config = PipelineConfig(root=common.PRODUCTION_ROOT)
        self.artifact = common.verify_privacy_artifacts(
            self.config, self.run_id, self.warnings
        )

        total = sum(clip["duration_seconds"] for clip in self.clips)
        info = self.system_info
        print()
        print("OnKith Full Audio Pipeline Benchmark")
        print("====================================")
        print()
        print(f"Device            : {info['board_model'] or info['platform']}")
        print(f"CPU               : {info['cpu_model']}")
        print(f"Architecture      : {info['architecture']} ({info['logical_cpus']} logical CPUs)")
        print(f"RAM               : {common.gib(info['total_ram_bytes'])}")
        print(f"Python            : {info['python']}   ONNX Runtime {info['onnxruntime']}")
        print(f"STT model         : {self.config.parakeet['model']} "
              f"({self.config.parakeet['runtime']}, "
              f"{self.config.parakeet['device']}, "
              f"{self.config.parakeet['precision']}, "
              f"{self.config.parakeet['decoder_family']} "
              f"{self.config.parakeet['decoder_strategy']})")
        print("Privacy model     : DeBERTa-v3-xsmall INT8")
        print(f"Privacy model SHA : {self.artifact['measured_sha256']}")
        print(f"Audio clips       : {len(self.clips)} "
              f"({total:.1f} s total, {self.options.mode} mode)")
        print(f"Run id            : {self.run_id}")
        print(f"Output            : {self.run.relative_path}")
        print()

        if not self.artifact["ok"]:
            self.warnings.note(
                "privacy artifact identity failed; the privacy sections will be skipped"
            )

    def verify_fixtures(self) -> None:
        print("Verifying fixtures ...")
        for clip in self.clips:
            clip["_path"] = verify_clip(clip, fixture_dir=self.options.manifest_path.parent)
        print(f"  {len(self.clips)} clip(s) match their manifest SHA-256")
        self.completed["fixture_verification"] = True

    def load_models(self) -> None:
        print("\nLoading pipeline ...")
        self._load_vad()
        self._load_stt()
        self._load_privacy()
        print()

    def _load_vad(self) -> None:
        try:
            from integrated_pipeline.silero_vad import SileroOnnxVAD

            started = time.perf_counter()
            self.vad = SileroOnnxVAD(self.config)
            self.vad.load()
            self.load_seconds["vad"] = time.perf_counter() - started
            self.stage_rss["vad_loaded"] = common.rss_bytes()
            print(f"  VAD      Silero v6 ONNX          loaded in {self.load_seconds['vad']:.2f} s")
        except Exception as exc:
            self.warnings.note(f"VAD unavailable: {type(exc).__name__}: {exc}")
            self.vad = None

    def _load_stt(self) -> None:
        if not self.options.run_stt:
            self.stt_reason = "disabled with --skip-stt"
            print("  STT      skipped (--skip-stt)")
            return
        try:
            from integrated_pipeline.parakeet_adapter import (
                ParakeetAdapter,
                describe_parakeet_backend,
            )

            backend = describe_parakeet_backend()
            if not backend.get("available"):
                raise RuntimeError(
                    "NeMo/torch Parakeet backend unavailable: "
                    + "; ".join(backend.get("errors", ["unknown"]))
                )
            common.quieten_stt()
            started = time.perf_counter()
            self.parakeet = ParakeetAdapter(self.config)
            # Loading includes the one-second warmup decode, exactly as the live
            # deployment does it, so "STT loaded" means "ready to transcribe".
            # It also proves the TDT branch is the active decoder and refuses
            # the CTC one, so a run folder cannot silently record the wrong
            # decoder for this checkpoint.
            self.parakeet.load()
            self.load_seconds["stt"] = time.perf_counter() - started
            self.stage_rss["stt_loaded"] = common.rss_bytes()
            self.stt_available = True
            self.stt_backend = {**backend, **self.parakeet.audit()}
            print(f"  STT      parakeet {self.config.parakeet['model']:15} "
                  f"loaded in {self.load_seconds['stt']:.2f} s")
        except Exception as exc:
            self.stt_reason = f"{type(exc).__name__}: {exc}"
            self.warnings.note(f"STT unavailable - {self.stt_reason}")
            print("  STT      UNAVAILABLE")

    def _load_privacy(self) -> None:
        if not self.artifact["ok"]:
            return
        from integrated_pipeline.pii_adapter import PIIMaskerAdapter

        started = time.perf_counter()
        self.adapter = PIIMaskerAdapter(self.config)
        self.adapter.load()
        self.load_seconds["privacy"] = time.perf_counter() - started
        self.stage_rss["privacy_loaded"] = common.rss_bytes()
        for _ in range(self.options.warmup_runs):
            self.adapter.mask_text(WARMUP_TEXT)
        self.stage_rss["all_models_loaded"] = common.rss_bytes()
        print(f"  Privacy  DeBERTa-v3-xsmall INT8    loaded in "
              f"{self.load_seconds['privacy']:.2f} s")

    # -- per-clip -----------------------------------------------------------

    def _transcribe(self, audio, clip_id: str) -> tuple[str, float]:
        segment = common.make_canary_segment(
            audio,
            session_id="audio-benchmark",
            utterance_id=f"audio-benchmark-{clip_id}",
            segment_id=f"audio-benchmark-{clip_id}-0",
        )
        started = time.perf_counter()
        transcript = self.parakeet.transcribe_segment(segment)
        return transcript.text, time.perf_counter() - started

    def _run_vad(self, audio) -> tuple[float, dict[str, Any]]:
        frame_samples = int(self.config.silero["frame_samples"])
        threshold = float(self.config.silero["threshold"])
        self.vad.reset()
        started = time.perf_counter()
        probabilities = [
            self.vad.probability(audio[start:start + frame_samples])
            for start in range(0, len(audio) - frame_samples + 1, frame_samples)
        ]
        elapsed = time.perf_counter() - started
        speech = sum(1 for value in probabilities if value >= threshold)
        return elapsed, {
            "vad_frames": len(probabilities),
            "vad_speech_frames": speech,
            "vad_speech_frame_ratio": speech / len(probabilities) if probabilities else 0.0,
        }

    def _mask(self, text: str) -> tuple[Any, float]:
        started = time.perf_counter()
        result = self.adapter.mask_text(text)
        return result, time.perf_counter() - started

    @staticmethod
    def _spans(result) -> list[dict[str, Any]]:
        return [
            {"start": int(entity["start"]), "end": int(entity["end"]),
             "label": str(entity["entity_type"]),
             "source": str(entity.get("source", ""))}
            for entity in result.entities
        ]

    def process_clip(self, clip: Mapping[str, Any], index: int, total: int) -> dict[str, Any]:
        from app.digit_normalizer import normalize_spoken_digits
        from app.wake_word_normalizer import normalize_wake_word
        from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
        from integrated_pipeline.schemas import TranscriptFragment
        from integrated_pipeline.transcript_merger import merge_segment_transcripts

        clip_id = clip["clip_id"]
        print(f"[{index:02d}/{total:02d}] {Path(clip['audio_path']).name:<22} "
              f"{clip['duration_seconds']:5.2f}s", end="", flush=True)

        record: dict[str, Any] = {
            "clip_id": clip_id,
            "subset": clip["subset"],
            "source": clip["source"],
            "audio_duration_s": clip["duration_seconds"],
            "duration_bucket": clip["duration_bucket"],
            "expected_entities": len(clip["expected_private_spans"]),
            "reference_transcript": clip["reference_transcript"],
        }

        started_total = time.perf_counter()
        started = time.perf_counter()
        audio = read_wav(clip["_path"])
        record["audio_load_latency_ms"] = (time.perf_counter() - started) * 1000.0

        # ---- VAD ----------------------------------------------------------
        if self.vad is not None:
            elapsed, stats = self._run_vad(audio)
            record["vad_latency_ms"] = elapsed * 1000.0
            record.update(stats)
        else:
            record["vad_latency_ms"] = None

        # ---- STT ----------------------------------------------------------
        stt_text = None
        if self.stt_available:
            stt_text, elapsed = self._transcribe(audio, clip_id)
            record["stt_latency_ms"] = elapsed * 1000.0
            record["stt_real_time_factor"] = elapsed / clip["duration_seconds"]
        else:
            record["stt_latency_ms"] = None
            record["stt_real_time_factor"] = None
        record["stt_transcript"] = stt_text

        del audio  # a 4 GB device keeps exactly one waveform alive at a time

        # ---- transcript normalisation / merging ----------------------------
        privacy_input = None
        if stt_text is not None:
            started = time.perf_counter()
            merged = merge_segment_transcripts([stt_text])
            wake_text, _variant = normalize_wake_word(merged.text)
            buffer = InactivityTranscriptBuffer(
                inactivity_seconds=self.config.buffer_inactivity_seconds,
                maximum_tokens=self.config.maximum_buffer_tokens,
                maximum_duration_seconds=self.config.maximum_buffer_duration_seconds,
            )
            buffer.add_fragment(TranscriptFragment(
                fragment_id=f"{clip_id}-0", text=wake_text, arrival_time_seconds=0.0,
                source_audio_id=clip_id, session_id="audio-benchmark",
                batch_id=clip_id, sequence_number=0,
            ))
            flushed = buffer.flush(0.0, reason="manual")
            privacy_input = flushed.text if flushed else wake_text
            record["normalization_latency_ms"] = (time.perf_counter() - started) * 1000.0
        else:
            record["normalization_latency_ms"] = None
        record["privacy_input_text"] = privacy_input

        # ---- privacy + postprocessing --------------------------------------
        stt_spans: list[dict[str, Any]] = []
        if privacy_input is not None and self.adapter is not None:
            result, elapsed = self._mask(privacy_input)
            record["privacy_latency_ms"] = elapsed * 1000.0
            stt_spans = self._spans(result)
            record["predicted_entities"] = len(stt_spans)
            masked_text = result.masked_text

            edits = list((result.raw_model_output or {}).get("spoken_form_edits") or [])
            record["spoken_form_edits"] = edits
            if edits:
                self.spoken_form_clips.append(clip_id)
                self.spoken_form_edits.extend({"clip_id": clip_id, **edit}
                                              for edit in edits)

            started = time.perf_counter()
            final_text = normalize_spoken_digits(masked_text)
            record["postprocessing_latency_ms"] = (time.perf_counter() - started) * 1000.0
            record["masked_text"] = masked_text
            record["final_masked_output"] = final_text
            # What the render step added to the released text, if anything.
            record["reconstructed_identifiers"] = reconstructed_digit_runs(
                masked_text, final_text
            )
        else:
            record["privacy_latency_ms"] = None
            record["postprocessing_latency_ms"] = None
            record["predicted_entities"] = None
            record["spoken_form_edits"] = []
            record["masked_text"] = None
            record["final_masked_output"] = None
            record["reconstructed_identifiers"] = []

        record["end_to_end_latency_ms"] = (time.perf_counter() - started_total) * 1000.0
        record["pipeline_real_time_factor"] = (
            record["end_to_end_latency_ms"] / 1000.0 / clip["duration_seconds"]
        )

        # ---- control arms ---------------------------------------------------
        control_spans: dict[str, list[dict[str, Any]]] = {}
        if self.adapter is not None:
            reference = clip["reference_transcript"]
            result, elapsed = self._mask(reference)
            record["reference_privacy_latency_ms"] = elapsed * 1000.0
            control_spans["reference_text"] = self._spans(result)
            record["reference_masked_output"] = result.masked_text

            uncased = uncased_unpunctuated(reference)
            result, _ = self._mask(uncased)
            control_spans["reference_uncased"] = self._spans(result)
            record["reference_uncased_text"] = uncased
            record["reference_uncased_masked_output"] = result.masked_text

        # ---- word error rate -------------------------------------------------
        if stt_text is not None:
            record["wer_detail"] = word_error_rate(clip["reference_transcript"], stt_text)
            record["wer"] = record["wer_detail"]["wer"]
        else:
            record["wer_detail"] = None
            record["wer"] = None

        # ---- entity-level end-to-end outcomes --------------------------------
        entity_records = []
        if privacy_input is not None:
            covered = _covered_characters(stt_spans)
            for span in clip["expected_private_spans"]:
                location = locate_entity(clip["reference_transcript"], span, privacy_input)
                coverage = mask_coverage(
                    privacy_input,
                    location["hypothesis_char_start"],
                    location["hypothesis_char_end"],
                    covered,
                )
                entity_records.append({
                    "clip_id": clip_id,
                    "label": span["label"],
                    "reference_text": span["text"],
                    "outcome": classify_entity(location, coverage),
                    "mask_coverage": coverage,
                    **location,
                })
        record["entities"] = entity_records

        current = common.rss_bytes()
        if current:
            self.peak_rss = max(self.peak_rss, current)
        record["peak_rss_mb"] = self.peak_rss / 1024 ** 2 if self.peak_rss else None

        outcomes = [entry["outcome"] for entry in entity_records]
        leaked = sum(1 for outcome in outcomes
                     if outcome in ("leaked_privacy_model", "stt_altered_leaked",
                                    "partially_masked"))
        if self.stt_available:
            print(f"  WER {record['wer']:5.1%}  RTF {record['stt_real_time_factor']:5.2f}  "
                  f"{record['end_to_end_latency_ms']:7.0f} ms"
                  + (f"  {len(outcomes) - leaked}/{len(outcomes)} masked" if outcomes else ""))
        else:
            print("  (STT unavailable; control arms only)")

        record["_control_spans"] = control_spans
        return record

    def run_clips(self) -> None:
        print("Processing clips")
        print("-" * 72)
        total = len(self.clips)
        for index, clip in enumerate(self.clips, start=1):
            try:
                self.clip_results.append(self.process_clip(clip, index, total))
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                self.failures.append(
                    {"clip_id": clip["clip_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
        print("-" * 72)
        self.completed["clip_processing"] = bool(self.clip_results) and not self.failures
        if self.failures:
            self.warnings.note(f"{len(self.failures)} clip(s) raised; see the summary")

    # -- aggregation ---------------------------------------------------------

    def summarise_latency(self) -> None:
        def column(name: str) -> list[float]:
            return [
                record[name] / 1000.0 for record in self.clip_results
                if record.get(name) is not None
            ]

        stages = {
            name: summarise(column(f"{name}_latency_ms"))
            for name in ("audio_load", "vad", "stt", "normalization", "privacy",
                         "postprocessing", "end_to_end")
        }
        stt_rtf = [r["stt_real_time_factor"] for r in self.clip_results
                   if r.get("stt_real_time_factor") is not None]
        pipeline_rtf = [r["pipeline_real_time_factor"] for r in self.clip_results
                        if r.get("pipeline_real_time_factor") is not None]

        by_bucket: dict[str, list[float]] = {}
        for record in self.clip_results:
            by_bucket.setdefault(record["duration_bucket"], []).append(
                record["end_to_end_latency_ms"] / 1000.0
            )

        self.latency_summary = {
            "run_id": self.run_id,
            "mode": self.options.mode,
            "clips": len(self.clip_results),
            "stt_available": self.stt_available,
            "model_load_seconds": dict(self.load_seconds),
            "stages": stages,
            "real_time_factor": {
                "definition": (
                    "processing seconds divided by audio seconds; RTF < 1 means the "
                    "device processes audio faster than it arrives"
                ),
                "stt": summarise(stt_rtf, unit_scale=1.0),
                "pipeline": summarise(pipeline_rtf, unit_scale=1.0),
            },
            "end_to_end_by_duration_bucket": {
                bucket: summarise(values) for bucket, values in sorted(by_bucket.items())
            },
            "stage_share_of_end_to_end": self._stage_shares(stages),
        }

        print("\nLatency")
        for name in ("audio_load", "vad", "stt", "normalization", "privacy",
                     "postprocessing", "end_to_end"):
            print_summary(name, stages[name])
        if stt_rtf:
            rtf = self.latency_summary["real_time_factor"]
            print(f"\nSTT real-time factor      mean {rtf['stt']['mean_ms']:.3f}  "
                  f"p50 {rtf['stt']['p50_ms']:.3f}  p90 {rtf['stt']['p90_ms']:.3f}  "
                  f"p95 {rtf['stt']['p95_ms']:.3f}")
            print(f"pipeline real-time factor mean {rtf['pipeline']['mean_ms']:.3f}  "
                  f"p50 {rtf['pipeline']['p50_ms']:.3f}  p90 {rtf['pipeline']['p90_ms']:.3f}  "
                  f"p95 {rtf['pipeline']['p95_ms']:.3f}")

    @staticmethod
    def _stage_shares(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
        """Each stage's share of total measured time. Answers "what dominates?"."""

        total = sum(
            stages[name].get("total_ms", 0.0)
            for name in ("audio_load", "vad", "stt", "normalization", "privacy",
                         "postprocessing")
        )
        if not total:
            return {}
        return {
            name: stages[name].get("total_ms", 0.0) / total
            for name in ("audio_load", "vad", "stt", "normalization", "privacy",
                         "postprocessing")
        }

    def summarise_stt(self) -> None:
        scored = [record for record in self.clip_results if record.get("wer") is not None]
        if not scored:
            self.stt_metrics = {
                "run_id": self.run_id, "available": False, "reason": self.stt_reason,
                "how_to_enable": STT_UNAVAILABLE_HELP,
            }
            return

        def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            if not records:
                return {"clips": 0}
            errors = sum(r["wer_detail"]["errors"] for r in records)
            words = sum(r["wer_detail"]["reference_words"] for r in records)
            per_clip = sorted(r["wer"] for r in records)
            return {
                "clips": len(records),
                "reference_words": words,
                "aggregate_wer": errors / words if words else 0.0,
                "mean_wer": statistics.fmean(per_clip),
                "median_wer": statistics.median(per_clip),
                "min_wer": per_clip[0],
                "max_wer": per_clip[-1],
                "substitutions": sum(r["wer_detail"]["substitutions"] for r in records),
                "deletions": sum(r["wer_detail"]["deletions"] for r in records),
                "insertions": sum(r["wer_detail"]["insertions"] for r in records),
            }

        subsets = sorted({record["subset"] for record in scored})
        self.stt_metrics = {
            "run_id": self.run_id,
            "available": True,
            "engine": "parakeet",
            "model": self.config.parakeet["model"],
            "model_id": self.config.parakeet["model_id"],
            "revision": self.config.parakeet["revision"],
            "checkpoint_sha256": self.config.parakeet["sha256"],
            "backend": (
                f"{self.config.parakeet['runtime']}_"
                f"{self.config.parakeet['device']}_"
                f"{self.config.parakeet['precision']}"
            ),
            "runtime": self.config.parakeet["runtime"],
            "runtime_version": (self.stt_backend or {}).get("nemo_toolkit_version"),
            "torch_version": (self.stt_backend or {}).get("torch_version"),
            "device": self.config.parakeet["device"],
            "compute_type": self.config.parakeet["precision"],
            # Read back off the loaded model rather than off the config, so this
            # records the decoder that actually ran.
            "decoding": {
                "strategy": self.config.parakeet["decoder_strategy"],
                "decoder_family": self.config.parakeet["decoder_family"],
                "greedy_max_symbols": self.config.parakeet["greedy_max_symbols"],
                "temperature": self.config.parakeet["temperature"],
                "active_decoder": (
                    self.parakeet.active_decoder if self.parakeet else {}
                ),
            },
            "note": (
                "A controlled deployment check on a few dozen clips, not a "
                "speech-recognition benchmark. Both sides of every WER are "
                "lowercased and stripped of punctuation before scoring, which is "
                "the same definition the Canary and Whisper runs used - Canary "
                "produced unpunctuated lowercase text (pnc=no) while Whisper and "
                "Parakeet punctuate, so normalising both sides is what keeps the "
                "three comparable."
            ),
            "overall": aggregate(scored),
            "by_subset": {
                name: aggregate([r for r in scored if r["subset"] == name])
                for name in subsets
            },
        }

        # The engine-comparable number, computed by the module that also scores
        # the frozen Canary folders from their stored transcripts, so both
        # engines are judged by one implementation. See stt_comparison.py for
        # why the raw WER above cannot be compared across engines.
        from .stt_comparison import digit_preservation, number_normalised_wer

        normalised = [
            number_normalised_wer(r["reference_transcript"], r["stt_transcript"])
            for r in scored
        ]
        normalised_words = sum(item["reference_words"] for item in normalised)
        normalised_errors = sum(item["errors"] for item in normalised)
        digit_rows = [
            item for item in (
                digit_preservation(r["reference_transcript"], r["stt_transcript"])
                for r in scored
            ) if item
        ]
        identifiers = sum(item["identifiers"] for item in digit_rows)
        exact_identifiers = sum(item["exact"] for item in digit_rows)
        self.stt_metrics["number_normalised"] = {
            "definition": (
                "the same Levenshtein word error rate, after spoken number words "
                "have been folded to numerals on both the reference and the "
                "hypothesis. Identical treatment on both sides and for every "
                "engine, which is what makes Canary, Whisper and Parakeet "
                "comparable; the raw figure above cannot be, because Canary wrote "
                "numbers as words and Whisper and Parakeet write numerals."
            ),
            "clips": len(normalised),
            "reference_words": normalised_words,
            "aggregate_wer": (
                normalised_errors / normalised_words if normalised_words else 0.0
            ),
            "mean_wer": statistics.fmean(item["wer"] for item in normalised),
            "median_wer": statistics.median(item["wer"] for item in normalised),
        }
        self.stt_metrics["digit_preservation"] = {
            "definition": (
                "long numbers (7+ digits) transcribed digit for digit, scored on "
                "the value rather than the spelling"
            ),
            "clips_with_identifiers": len(digit_rows),
            "identifiers": identifiers,
            "exact": exact_identifiers,
            "accuracy": exact_identifiers / identifiers if identifiers else None,
        }
        self.stt_metrics["wake_word"] = {
            "measurable": False,
            "reason": (
                "no clip in the committed corpus addresses the device by name, so "
                "wake-word recognition is not measurable here for any engine"
            ),
        }

        print("\nSpeech recognition")
        overall = self.stt_metrics["overall"]
        print(f"  aggregate WER {overall['aggregate_wer']:.4f}   "
              f"mean {overall['mean_wer']:.4f}   median {overall['median_wer']:.4f}   "
              f"({overall['clips']} clips, {overall['reference_words']} reference words)")
        for name, entry in self.stt_metrics["by_subset"].items():
            if entry["clips"]:
                print(f"    {name:20} aggregate {entry['aggregate_wer']:.4f}  "
                      f"median {entry['median_wer']:.4f}  ({entry['clips']} clips)")
        engine_comparable = self.stt_metrics["number_normalised"]
        print(f"  number-normalised WER (engine-comparable) "
              f"{engine_comparable['aggregate_wer']:.4f}   "
              f"mean {engine_comparable['mean_wer']:.4f}   "
              f"median {engine_comparable['median_wer']:.4f}")
        digits = self.stt_metrics["digit_preservation"]
        if digits["identifiers"]:
            print(f"  long numbers transcribed exactly  {digits['exact']}/"
                  f"{digits['identifiers']} ({digits['accuracy']:.1%}) over "
                  f"{digits['clips_with_identifiers']} clips")

    @staticmethod
    def _comparable_arm(pairs) -> dict[str, Any]:
        """Score a reference-text arm the way the end-to-end arm is scored.

        Each gold entity is measured against the predicted spans over its own
        characters, whitespace excluded, and counts as protected only when it is
        fully covered. Identical definition, identical denominator - which is
        what makes "reference text against audio pipeline" a fair comparison
        rather than two numbers that merely share a name.
        """

        entities = fully_masked = characters = 0
        covered_total = 0.0
        for clip, spans in pairs:
            covered = _covered_characters(spans)
            reference = clip["reference_transcript"]
            for span in clip["expected_private_spans"]:
                coverage = mask_coverage(reference, span["start"], span["end"], covered)
                informative = _informative_characters(
                    reference[span["start"]:span["end"]]
                )
                entities += 1
                fully_masked += int(coverage >= FULL_MASK_THRESHOLD)
                covered_total += coverage * informative
                characters += informative
        recall = covered_total / characters if characters else 0.0
        return {
            "expected_entities": entities,
            "entities_fully_masked": fully_masked,
            "entity_recall": fully_masked / entities if entities else 0.0,
            "character_recall": recall,
            "character_leakage": 1.0 - recall if characters else 0.0,
            "definition": (
                "per-entity character coverage, whitespace excluded; identical to "
                "the end-to-end arm, so the two are directly comparable"
            ),
        }

    def summarise_privacy(self) -> None:
        from privacy_eval.dataset_metrics import evaluate_spans

        # ---- control arms, exact gold offsets --------------------------------
        # Only clips annotated for privacy may produce a privacy number. The
        # LibriSpeech clips are read prose full of person names carrying no gold
        # annotation; scoring them would count correct detections as false
        # positives. They measure recognition, not masking.
        by_id = {clip["clip_id"]: clip for clip in self.clips}
        scored = [
            (by_id[record["clip_id"]], record)
            for record in self.clip_results
            if record["_control_spans"]
            and by_id.get(record["clip_id"], {}).get("privacy_annotated", True)
        ]
        gold = [
            {"privacy_mask": [
                {"start": span["start"], "end": span["end"], "label": span["label"],
                 "value": span["text"]}
                for span in clip["expected_private_spans"]
            ], "source_text": clip["reference_transcript"]}
            for clip, _ in scored
        ]
        for arm in ("reference_text", "reference_uncased"):
            predictions = [record["_control_spans"].get(arm, []) for _, record in scored]
            if not predictions:
                continue
            self.control_metrics[arm] = evaluate_spans(gold, predictions)
            # ``evaluate_spans`` counts every gold character, including the spaces
            # inside a multi-word entity. The end-to-end arm cannot use it at all,
            # because recognized text has no gold offsets. So each arm also gets
            # the end-to-end definition, and that is the one the figures plot.
            self.control_metrics[arm]["comparable"] = self._comparable_arm(
                [(clip, record["_control_spans"].get(arm, [])) for clip, record in scored]
            )
        self.privacy_scored_clips = [clip["clip_id"] for clip, _ in scored]

        # ---- the end-to-end arm ----------------------------------------------
        entities = [entry for record in self.clip_results for entry in record["entities"]]
        outcomes = {name: 0 for name in ("masked", "partially_masked", "leaked_privacy_model",
                                         "stt_altered_masked", "stt_altered_leaked")}
        for entry in entities:
            outcomes[entry["outcome"]] += 1

        preserved = [e for e in entities if e["preservation_ratio"] >= PRESERVATION_THRESHOLD]
        # Weighted by informative characters, exactly as ``_comparable_arm``
        # weights the control arms. Weighting by the raw region length instead
        # would count the spaces inside a multi-word entity, which no mask ever
        # covers, and the two arms would no longer share a denominator.
        covered_characters = sum(
            e["mask_coverage"] * _informative_characters(e["hypothesis_surface"])
            for e in entities
        )
        total_characters = sum(
            _informative_characters(e["hypothesis_surface"]) for e in entities
        )

        scored_clips = [
            record for record in self.clip_results if record["entities"]
        ]
        sanitized = sum(
            1 for record in scored_clips
            if all(entry["outcome"] in ("masked", "stt_altered_masked")
                   for entry in record["entities"])
        )
        leaking = sum(
            1 for record in scored_clips
            if any(entry["outcome"] in ("leaked_privacy_model", "stt_altered_leaked",
                                        "partially_masked")
                   for entry in record["entities"])
        )
        annotated_ids = {
            clip["clip_id"] for clip in self.clips if clip.get("privacy_annotated", True)
        }
        negatives = [
            record for record in self.clip_results
            if record["clip_id"] in annotated_ids
            and record["expected_entities"] == 0
            and record.get("predicted_entities") is not None
        ]
        negative_hits = sum(1 for record in negatives if record["predicted_entities"])

        # ---- what postprocessing added after the privacy decision ------------
        # A separate question from "did the model mask it". The renderer only
        # ever sees text the masker already released, so anything it builds here
        # is downstream of every privacy decision in the pipeline.
        amplified = []
        for record in self.clip_results:
            for run in record.get("reconstructed_identifiers") or []:
                leaked = [
                    entry["label"] for entry in record["entities"]
                    if entry["outcome"] != "masked"
                ]
                amplified.append({
                    "clip_id": record["clip_id"],
                    "subset": record["subset"],
                    "rendered_value": run["value"],
                    "digits": run["digits"],
                    "unmasked_entity_labels": leaked,
                    "privacy_input_text": record["privacy_input_text"],
                    "final_masked_output": record["final_masked_output"],
                })

        by_label: dict[str, dict[str, int]] = {}
        for entry in entities:
            bucket = by_label.setdefault(
                entry["label"], {"expected": 0, "preserved_by_stt": 0, "masked": 0}
            )
            bucket["expected"] += 1
            bucket["preserved_by_stt"] += int(
                entry["preservation_ratio"] >= PRESERVATION_THRESHOLD
            )
            bucket["masked"] += int(entry["outcome"] == "masked")

        self.privacy_metrics = {
            "run_id": self.run_id,
            "benchmark": "controlled end-to-end deployment benchmark",
            "not_a_replacement_for": [
                "model_v2 held-out test", "combined validation", "ood_dev",
                "frozen ood_final",
            ],
            "clips_scored": len(self.clip_results),
            "stt_available": self.stt_available,
            "thresholds": {
                "preservation_ratio": PRESERVATION_THRESHOLD,
                "full_mask_coverage": FULL_MASK_THRESHOLD,
            },
            "control_arms": self.control_metrics,
            "privacy_scored_clips": self.privacy_scored_clips,
            "privacy_scoring_note": (
                "Only clips annotated for private information are scored here. The "
                "LibriSpeech subset is read prose containing person names with no "
                "gold annotation, so its detections are neither credited nor "
                "counted against the model; it measures recognition only."
            ),
            "end_to_end": {
                "expected_entities": len(entities),
                "entities_preserved_by_stt": len(preserved),
                "stt_preservation_rate": len(preserved) / len(entities) if entities else 0.0,
                "entities_fully_masked": outcomes["masked"],
                "entity_recall_after_stt": (
                    outcomes["masked"] / len(entities) if entities else 0.0
                ),
                "entity_recall_among_preserved": (
                    outcomes["masked"] / len(preserved) if preserved else 0.0
                ),
                "character_recall_after_stt": (
                    covered_characters / total_characters if total_characters else 0.0
                ),
                "character_leakage_after_stt": (
                    1.0 - covered_characters / total_characters if total_characters else 0.0
                ),
                "outcomes": outcomes,
                "fully_sanitized_clips": sanitized,
                "fully_sanitized_clip_rate": (
                    sanitized / len(scored_clips) if scored_clips else 0.0
                ),
                "clips_with_any_leakage": leaking,
                "clips_with_any_leakage_rate": (
                    leaking / len(scored_clips) if scored_clips else 0.0
                ),
                "hard_negative_clips": len(negatives),
                "hard_negative_false_positive_clips": negative_hits,
                "hard_negative_false_positive_rate": (
                    negative_hits / len(negatives) if negatives else 0.0
                ),
                "by_entity_type": dict(sorted(by_label.items())),
            },
            "postprocessing_amplification": {
                "identifier_digits": IDENTIFIER_DIGITS,
                "clips_affected": len({entry["clip_id"] for entry in amplified}),
                "identifiers_reconstructed": len(amplified),
                "cases": amplified,
                "mechanism": (
                    "A spoken number can arrive as digit "
                    "words. The privacy model misses some of those - it is the same "
                    "leak the entity outcomes already count - and the display "
                    "renderer, which by design runs after masking, then rewrites the "
                    "surviving words into a numeral. No new information escapes; the "
                    "information that escaped becomes machine readable."
                ),
                "scope": (
                    "Display only. app/digit_normalizer.py is documented as a "
                    "render-time transform: storage, Onkith and the preview's own "
                    "arithmetic keep the original text, so the reconstruction reaches "
                    "the terminal and the JSON display_text field, not the database."
                ),
            },
            "failure_attribution": {
                "stt_caused": outcomes["stt_altered_leaked"],
                "stt_caused_but_still_masked": outcomes["stt_altered_masked"],
                "privacy_model_caused": (
                    outcomes["leaked_privacy_model"] + outcomes["partially_masked"]
                ),
                "protected": outcomes["masked"],
                "postprocessing_amplified": len(amplified),
                "interpretation": (
                    "stt_caused: the entity did not survive recognition and nothing "
                    "was masked where it went. privacy_model_caused: the entity was "
                    "recognizable in the transcript and the model did not mask it, or "
                    "masked only part of it. Both are end-to-end leaks; only the "
                    "second is a privacy-model defect. postprocessing_amplified "
                    "counts a third thing, and it is not a fourth kind of leak: a "
                    "leak already counted above that the render step rewrote into a "
                    "numeral, which is why it is reported separately rather than "
                    "added to the totals."
                ),
            },
        }

        print("\nPrivacy, end to end")
        end = self.privacy_metrics["end_to_end"]
        if self.stt_available and entities:
            print(f"  expected entities                 {end['expected_entities']}")
            print(f"  survived speech recognition       {end['entities_preserved_by_stt']} "
                  f"({end['stt_preservation_rate']:.1%})")
            print(f"  fully masked end to end           {end['entities_fully_masked']} "
                  f"({end['entity_recall_after_stt']:.1%})")
            print(f"  character recall after STT        {end['character_recall_after_stt']:.4f}")
            print(f"  character leakage after STT       {end['character_leakage_after_stt']:.4f}")
            print(f"  fully sanitized clips             {end['fully_sanitized_clips']}"
                  f"/{len(scored_clips)} ({end['fully_sanitized_clip_rate']:.1%})")
            print(f"  hard-negative false positives     "
                  f"{end['hard_negative_false_positive_clips']}/{end['hard_negative_clips']}")
            attribution = self.privacy_metrics["failure_attribution"]
            print(f"\n  attribution: {attribution['protected']} protected, "
                  f"{attribution['privacy_model_caused']} privacy-model failures, "
                  f"{attribution['stt_caused']} STT-caused, "
                  f"{attribution['stt_caused_but_still_masked']} STT-altered but masked")
            if amplified:
                print(f"\n  postprocessing rebuilt {len(amplified)} spoken number(s) "
                      f"into digits after masking:")
                for entry in amplified:
                    print(f"    {entry['clip_id']:<14} -> {entry['rendered_value']} "
                          f"({entry['digits']} digits, "
                          f"{', '.join(entry['unmasked_entity_labels']) or 'no gold entity'})")
        for arm, metrics in self.control_metrics.items():
            comparable = metrics["comparable"]
            print(f"  control [{arm:18}] typed F1 {metrics['typed_entity']['f1']:.4f}  "
                  f"entity recall {comparable['entity_recall']:.4f}  "
                  f"character recall {comparable['character_recall']:.4f}")
        self.completed["privacy_evaluation"] = bool(self.control_metrics)

    def collect_memory(self) -> None:
        self.memory = common.collect_memory(
            self.run_id, baseline_rss=self.baseline_rss, peak_rss=self.peak_rss,
            stages=self.stage_rss,
        )
        self.memory["model_load_seconds"] = dict(self.load_seconds)
        self.memory["stage_rss_note"] = (
            "RSS sampled after each model finished loading, then the peak sampled "
            "after every clip. The target device has about 4 GB; nothing here "
            "establishes that the whole system fits until this benchmark has been "
            "run on the Raspberry Pi itself."
        )
        if common.psutil:
            print("\nMemory")
            for name in ("baseline", "vad_loaded", "stt_loaded", "privacy_loaded",
                         "all_models_loaded"):
                value = self.stage_rss.get(name)
                if value:
                    print(f"  {name:20} {value / 1024 ** 2:>8.1f} MiB")
            print(f"  {'peak during run':20} "
                  f"{self.memory['process_peak_rss_bytes'] / 1024 ** 2:>8.1f} MiB")

    # -- output ---------------------------------------------------------------

    def save(self) -> None:
        finished_utc = common.utc_now()
        totals = self.manifest["totals"]
        config_payload = {
            "run_id": self.run_id,
            "benchmark": "audio_pipeline",
            # Recorded so the figures notebook can separate this run from the
            # frozen Canary runs, which predate these keys and default to canary.
            "pipeline_version": "V5",
            "pipeline_root": common.relative(common.PRODUCTION_ROOT),
            "stt_engine": "parakeet",
            "stt_model": self.config.parakeet["model"],
            "stt_model_id": self.config.parakeet["model_id"],
            "stt_model_revision": self.config.parakeet["revision"],
            "stt_checkpoint_sha256": self.config.parakeet["sha256"],
            "stt_runtime": self.config.parakeet["runtime"],
            "stt_runtime_version": (self.stt_backend or {}).get(
                "nemo_toolkit_version"
            ),
            "stt_torch_version": (self.stt_backend or {}).get("torch_version"),
            "stt_device": self.config.parakeet["device"],
            "stt_compute_type": self.config.parakeet["precision"],
            "stt_decoder_family": self.config.parakeet["decoder_family"],
            "stt_decoder_strategy": self.config.parakeet["decoder_strategy"],
            "mode": self.options.mode,
            "smoke_test": self.options.smoke_test,
            "device_label": self.options.device_label or None,
            "entry_point": "Full_Pipeline_Audio_Benchmark.py",
            # Which definition produced the per-entity outcomes in this folder.
            # A run written before the alignment fix carries no such key, and
            # its ``entities`` blocks are not comparable with these; use
            # ``privacy_eval.rescore`` to put an older folder on this footing.
            "entity_scoring": ENTITY_SCORING_VERSION,
            "spoken_form_normalisation": {
                "stage": "pre-detection view inside PIIMaskerAdapter.mask_text",
                "rules": list(_spoken_form_rules()),
                "clips_rewritten": self.spoken_form_clips,
                "edits": self.spoken_form_edits,
            },
            "clips_selected": [clip["clip_id"] for clip in self.clips],
            "clips_scored": len(self.clip_results),
            "subset_filter": self.options.subset,
            "run_stt": self.options.run_stt,
            "stt_available": self.stt_available,
            "stt_unavailable_reason": self.stt_reason or None,
            "warmup_runs": self.options.warmup_runs,
            "fixtures": {
                "manifest": common.relative(self.options.manifest_path),
                "manifest_sha256": common.sha256_file(self.options.manifest_path),
                "schema_version": self.manifest["schema_version"],
                "generated_utc": self.manifest["generated_utc"],
                "available_clips": totals["clips"],
                "available_duration_seconds": totals["total_duration_seconds"],
                "subsets": {
                    name: {key: value for key, value in entry.items() if key != "voices"}
                    for name, entry in self.manifest["subsets"].items()
                },
            },
            "selected_clip_sha256": {
                clip["clip_id"]: clip["sha256"] for clip in self.clips
            },
            "started_utc": self.started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "duration_seconds": (finished_utc - self.started_utc).total_seconds(),
        }

        self.run.json("system_info.json", self.system_info)
        self.run.json("benchmark_config.json", config_payload)
        self.run.json("artifact_manifest.json", common.artifact_manifest_for_disk(self.artifact))
        self.run.json("audio_latency_summary.json", self.latency_summary)
        self.run.json("audio_stt_metrics.json", self.stt_metrics)
        self.run.json("audio_privacy_metrics.json", self.privacy_metrics)
        self.run.json("audio_memory_metrics.json", self.memory)

        self.run.csv(
            "audio_clip_results.csv",
            [
                {key: record.get(key) for key in CLIP_CSV_FIELDS}
                for record in self.clip_results
            ],
            CLIP_CSV_FIELDS,
        )
        self.run.jsonl(
            "audio_predictions.jsonl",
            [
                {key: value for key, value in record.items() if key != "_control_spans"}
                for record in self.clip_results
            ],
        )

        failure_rows = [
            {
                "clip_id": entry["clip_id"],
                "label": entry["label"],
                "outcome": entry["outcome"],
                "attribution": (
                    "privacy_model" if entry["outcome"] in
                    ("leaked_privacy_model", "partially_masked")
                    else "speech_recognition"
                ),
                "reference_entity": entry["reference_text"],
                "stt_surface": entry["hypothesis_surface"],
                "preservation_ratio": entry["preservation_ratio"],
                "mask_coverage": entry["mask_coverage"],
                "reference_transcript": record["reference_transcript"],
                "stt_transcript": record["stt_transcript"],
                "final_masked_output": record["final_masked_output"],
            }
            for record in self.clip_results
            for entry in record["entities"]
            if entry["outcome"] != "masked"
        ]
        # Successful masking despite a changed surface is evidence too, so a few
        # are kept: a file holding only failures would be a cherry-picked file.
        survivors = [
            {
                "clip_id": entry["clip_id"], "label": entry["label"],
                "outcome": entry["outcome"], "attribution": "protected",
                "reference_entity": entry["reference_text"],
                "stt_surface": entry["hypothesis_surface"],
                "preservation_ratio": entry["preservation_ratio"],
                "mask_coverage": entry["mask_coverage"],
                "reference_transcript": record["reference_transcript"],
                "stt_transcript": record["stt_transcript"],
                "final_masked_output": record["final_masked_output"],
            }
            for record in self.clip_results
            for entry in record["entities"]
            if entry["outcome"] == "masked" and entry["preservation_ratio"] < 1.0
        ]
        amplification_rows = [
            {
                "clip_id": entry["clip_id"],
                "label": ", ".join(entry["unmasked_entity_labels"]) or None,
                "outcome": "identifier_rebuilt_after_masking",
                "attribution": "postprocessing",
                "rendered_value": entry["rendered_value"],
                "digits": entry["digits"],
                "privacy_input_text": entry["privacy_input_text"],
                "final_masked_output": entry["final_masked_output"],
            }
            for entry in (self.privacy_metrics.get("postprocessing_amplification", {})
                          .get("cases", []))
        ]
        self.run.jsonl("audio_failure_analysis.jsonl",
                       failure_rows + amplification_rows + survivors)

        self._write_summary(config_payload)
        self.run.report()

    def _write_summary(self, config_payload: Mapping[str, Any]) -> None:
        info = self.system_info
        totals = self.manifest["totals"]
        selected_seconds = sum(clip["duration_seconds"] for clip in self.clips)
        lines = [
            f"# Audio pipeline benchmark - {self.run_id}",
            "",
            "A **controlled end-to-end deployment benchmark**: audio in, sanitized text",
            "out, on the real production pipeline. It measures whether the deployment is",
            "practical on this device and whether private information survives imperfect",
            "speech recognition. It is not a speech-recognition benchmark and it does not",
            "replace the held-out text evaluation, the combined validation, `ood_dev` or",
            "the frozen `ood_final`.",
            "",
            f"- **Clips scored:** {len(self.clip_results)} of {totals['clips']} available "
            f"({selected_seconds:.1f} s of audio, {self.options.mode} mode)",
            f"- **Speech recognizer:** "
            + (f"NVIDIA {self.config.parakeet['model']} "
               f"({self.config.parakeet['runtime']}, "
               f"{self.config.parakeet['device']}, "
               f"{self.config.parakeet['precision']}, "
               f"{self.config.parakeet['decoder_family']} "
               f"{self.config.parakeet['decoder_strategy']})" if self.stt_available
               else f"UNAVAILABLE - {self.stt_reason}"),
            f"- **Started / finished (UTC):** {config_payload['started_utc'][:19]} -> "
            f"{config_payload['finished_utc'][:19]} "
            f"({config_payload['duration_seconds'] / 60:.1f} min)",
            "",
            "## Device",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| OS | {info['platform']} |",
            f"| Architecture | {info['architecture']} ({info['process_architecture']}) |",
            f"| CPU | {info['cpu_model']} |",
            f"| Logical / physical cores | {info['logical_cpus']} / {info['physical_cpus']} |",
            f"| Board | {info['board_model'] or 'not reported'} |",
            f"| RAM | {common.gib(info['total_ram_bytes'])} |",
            f"| Privacy model SHA-256 | `{self.artifact['measured_sha256']}` |",
            "",
            "## Benchmark audio",
            "",
        ]
        for name, entry in self.manifest["subsets"].items():
            summary = entry["summary"]
            lines += [
                f"**{entry['name']}** - {summary['clips']} clips, "
                f"{summary['total_duration_seconds']:.1f} s, "
                f"{summary['expected_spans']} expected entities. "
                f"License: {entry['license'] if name == 'public' else 'see manifest'}.",
                "",
            ]
        lines += [
            "All private information in the synthetic subset is fictional: reserved "
            "`example` domains, the 555-01xx fictional telephone range, and invented "
            "names, addresses and document numbers.",
            "",
        ]

        if self.stt_available and self.stt_metrics.get("available"):
            overall = self.stt_metrics["overall"]
            lines += [
                "## Speech recognition",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Aggregate WER | {overall['aggregate_wer']:.4f} |",
                f"| Mean WER | {overall['mean_wer']:.4f} |",
                f"| Median WER | {overall['median_wer']:.4f} |",
                f"| Reference words | {overall['reference_words']:,} |",
                "",
                "The three rows above use the benchmark's standing definition:",
                "Levenshtein over lowercased, punctuation-stripped words. It is",
                "comparable across devices for one recognizer and **not** across",
                "recognizers, because Canary wrote spoken numbers as words while",
                "Whisper and Parakeet write numerals. The engine-comparable figure",
                "folds numbers to numerals on both sides:",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Aggregate WER, number-normalised | "
                f"{self.stt_metrics['number_normalised']['aggregate_wer']:.4f} |",
                f"| Mean WER, number-normalised | "
                f"{self.stt_metrics['number_normalised']['mean_wer']:.4f} |",
                f"| Long numbers transcribed exactly | "
                + (f"{self.stt_metrics['digit_preservation']['exact']}/"
                   f"{self.stt_metrics['digit_preservation']['identifiers']}"
                   if self.stt_metrics["digit_preservation"]["identifiers"]
                   else "no identifier-scale numbers in this selection")
                + " |",
                "| Wake-word recognition | not measurable: no clip in the corpus "
                "addresses the device by name |",
                "",
            ]
            for name, entry in self.stt_metrics["by_subset"].items():
                if entry["clips"]:
                    lines.append(f"- `{name}`: aggregate WER {entry['aggregate_wer']:.4f} "
                                 f"over {entry['clips']} clips")
            lines.append("")

            lines += [
                "### Spoken address forms rewritten before detection",
                "",
                "The recognizer writes an address the way it was spoken - "
                "`billing at example.org`. Neither the privacy model nor the "
                "structured EMAIL recognizer has seen one written that way, and "
                "the result was the domain masked and the local part released. "
                "`integrated_pipeline/spoken_form_normalizer.py` rewrites two "
                "such shapes into their written form for the detectors only; "
                "every span is projected back before anything is masked, so the "
                "released transcript is still the transcript.",
                "",
            ]
            if self.spoken_form_edits:
                lines += ["| Clip | Rule | Recognized | Shown to the detectors |",
                          "|---|---|---|---|"]
                for edit in self.spoken_form_edits:
                    lines.append(
                        f"| {edit['clip_id']} | `{edit['rule']}` | "
                        f"`{edit['original']}` | `{edit['normalised']}` |"
                    )
                lines += ["", f"{len(self.spoken_form_edits)} rewrite(s) across "
                              f"{len(set(self.spoken_form_clips))} clip(s).", ""]
            else:
                lines += ["No clip in this selection contained a spoken address "
                          "form, so the stage was a no-op.", ""]
        else:
            lines += [
                "## Speech recognition",
                "",
                f"**Unavailable on this machine.** `{self.stt_reason}`",
                "",
                "```",
                STT_UNAVAILABLE_HELP,
                "```",
                "",
            ]

        if self.latency_summary.get("stages"):
            lines += ["## Latency", "", "| Stage | Statistics |", "|---|---|"]
            for name, entry in self.latency_summary["stages"].items():
                lines.append(f"| {name} | {common.latency_line(entry)} |")
            shares = self.latency_summary.get("stage_share_of_end_to_end") or {}
            if shares:
                dominant = max(shares, key=lambda name: shares[name])
                lines += [
                    "",
                    f"Dominant stage: **{dominant}** at {shares[dominant]:.1%} of measured "
                    "processing time.",
                    "",
                ]
            rtf = self.latency_summary.get("real_time_factor", {})
            if rtf.get("stt", {}).get("count"):
                lines += [
                    "### Real-time factor",
                    "",
                    "`RTF < 1` means the device processes audio faster than the audio "
                    "lasts, which is the condition for live use.",
                    "",
                    "| Measure | Mean | P50 | P90 | P95 |",
                    "|---|---:|---:|---:|---:|",
                    f"| STT | {rtf['stt']['mean_ms']:.3f} | {rtf['stt']['p50_ms']:.3f} | "
                    f"{rtf['stt']['p90_ms']:.3f} | {rtf['stt']['p95_ms']:.3f} |",
                    f"| Full pipeline | {rtf['pipeline']['mean_ms']:.3f} | "
                    f"{rtf['pipeline']['p50_ms']:.3f} | {rtf['pipeline']['p90_ms']:.3f} | "
                    f"{rtf['pipeline']['p95_ms']:.3f} |",
                    "",
                ]

        if self.control_metrics:
            lines += [
                "## Privacy: reference text against the audio pipeline",
                "",
                "| Arm | Typed F1 | Entity recall | Character recall | Leakage |",
                "|---|---:|---:|---:|---:|",
            ]
            for arm, metrics in self.control_metrics.items():
                comparable = metrics["comparable"]
                lines.append(
                    f"| {arm} (control) | {metrics['typed_entity']['f1']:.4f} | "
                    f"{comparable['entity_recall']:.4f} | "
                    f"{comparable['character_recall']:.4f} | "
                    f"{comparable['character_leakage']:.4f} |"
                )
            end = (self.privacy_metrics or {}).get("end_to_end", {})
            if self.stt_available and end.get("expected_entities"):
                lines.append(
                    f"| audio -> STT -> privacy | not applicable | "
                    f"{end['entity_recall_after_stt']:.4f} | "
                    f"{end['character_recall_after_stt']:.4f} | "
                    f"{end['character_leakage_after_stt']:.4f} |"
                )
            lines += [
                "",
                "Entity recall and character recall use one definition across all three "
                "rows: each gold entity measured over its own characters, whitespace "
                "excluded. Typed F1 needs exact gold offsets, which recognized text does "
                "not have, so it is reported for the control arms only.",
                "",
                f"Scored on the {len(self.privacy_scored_clips)} privacy-annotated clips. "
                "The LibriSpeech subset carries no gold annotation - read prose is full "
                "of person names - so it contributes to recognition metrics only.",
                "",
            ]

        end = (self.privacy_metrics or {}).get("end_to_end", {})
        if self.stt_available and end.get("expected_entities"):
            attribution = self.privacy_metrics["failure_attribution"]
            lines += [
                "## End-to-end privacy outcome",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Expected entities | {end['expected_entities']} |",
                f"| Survived speech recognition | {end['entities_preserved_by_stt']} "
                f"({end['stt_preservation_rate']:.1%}) |",
                f"| Fully masked end to end | {end['entities_fully_masked']} "
                f"({end['entity_recall_after_stt']:.1%}) |",
                f"| Character recall after STT | {end['character_recall_after_stt']:.4f} |",
                f"| Character leakage after STT | {end['character_leakage_after_stt']:.4f} |",
                f"| Fully sanitized clips | {end['fully_sanitized_clips']} "
                f"({end['fully_sanitized_clip_rate']:.1%}) |",
                f"| Clips with any leakage | {end['clips_with_any_leakage']} "
                f"({end['clips_with_any_leakage_rate']:.1%}) |",
                f"| Hard-negative false-positive clips | "
                f"{end['hard_negative_false_positive_clips']} of {end['hard_negative_clips']} |",
                "",
                "### Failure attribution",
                "",
                "| Cause | Entities |",
                "|---|---:|",
                f"| Protected | {attribution['protected']} |",
                f"| Privacy-model failure | {attribution['privacy_model_caused']} |",
                f"| STT-caused, nothing masked | {attribution['stt_caused']} |",
                f"| STT-altered but masked anyway | "
                f"{attribution['stt_caused_but_still_masked']} |",
                "",
                "Examples of both kinds are in `audio_failure_analysis.jsonl`, which "
                "also keeps successfully masked entities whose surface the recognizer "
                "changed.",
                "",
            ]
            amplification = self.privacy_metrics.get("postprocessing_amplification", {})
            if amplification.get("cases"):
                lines += [
                    "### Spoken numbers rebuilt after masking",
                    "",
                    f"Parakeet punctuates, but it writes a number read aloud digit "
                    f"by digit as digit *words* rather than as a numeral, so the "
                    f"V3-era shape is back under V5 after being absent under Whisper. "
                    f"Where the privacy model missed one - already counted as a "
                    f"privacy-model failure above - the display renderer "
                    f"(`app/digit_normalizer.py`, which runs after masking by design) "
                    f"rewrote the surviving words into a numeral of "
                    f"{amplification['identifier_digits']}+ digits:",
                    "",
                    "| Clip | Entity | Released as | Digits |",
                    "|---|---|---|---:|",
                ]
                for case in amplification["cases"]:
                    labels = ", ".join(case["unmasked_entity_labels"]) or "none annotated"
                    lines.append(
                        f"| {case['clip_id']} | {labels} | `{case['rendered_value']}` | "
                        f"{case['digits']} |"
                    )
                lines += [
                    "",
                    f"{amplification['identifiers_reconstructed']} identifier(s) across "
                    f"{amplification['clips_affected']} clip(s). This is not additional "
                    "leaked information - the digit words were already in the released "
                    "text - but it is the compact, copyable form of it. "
                    + amplification["scope"],
                    "",
                ]

        if self.memory:
            lines += [
                "## Memory",
                "",
                "| Stage | RSS |",
                "|---|---|",
            ]
            for name in ("baseline", "vad_loaded", "stt_loaded", "privacy_loaded",
                         "all_models_loaded"):
                value = self.stage_rss.get(name)
                if value:
                    lines.append(f"| {name} | {common.mib(value)} |")
            lines += [
                f"| peak during run | {common.mib(self.memory.get('process_peak_rss_bytes'))} |",
                f"| system RAM | {common.gib(self.memory.get('system_total_bytes'))} |",
                "",
            ]

        lines += ["## Warnings", ""]
        lines += [f"- {message}" for message in self.warnings] or ["- none"]
        lines += ["", "## Sections completed", ""]
        lines += [
            f"- {name}: {'yes' if done else 'no'}"
            for name, done in sorted(self.completed.items())
        ] or ["- none"]

        path = self.run.markdown("audio_benchmark_summary.md", lines)
        print(f"\nwrote {common.relative(path)}")

    # -- orchestration ---------------------------------------------------------

    def execute(self) -> int:
        self.announce()
        self.verify_fixtures()
        self.load_models()
        self.run_clips()
        self.summarise_latency()
        self.summarise_stt()
        self.summarise_privacy()
        self.collect_memory()
        self.save()
        self.close()

        if self.failures or not self.artifact["ok"]:
            return 1
        if not self.stt_available:
            print()
            print("=" * 72)
            print("SPEECH RECOGNIZER UNAVAILABLE - the benchmark is incomplete")
            print("=" * 72)
            print(STT_UNAVAILABLE_HELP)
            print()
            print("Everything that does not need it was measured and written: fixture")
            print("verification, VAD, both reference-text control arms, privacy-model")
            print("latency and the memory profile. Exit code 2 means exactly this.")
            return 2
        return 0

    def close(self) -> None:
        for component in (self.vad, self.parakeet):
            try:
                if component is not None:
                    component.close()
            except Exception:  # closing must never lose a completed benchmark
                pass


CLIP_CSV_FIELDS = [
    "clip_id", "subset", "source", "audio_duration_s", "duration_bucket",
    "expected_entities", "predicted_entities", "wer",
    "audio_load_latency_ms", "vad_latency_ms", "stt_latency_ms",
    "normalization_latency_ms", "privacy_latency_ms", "postprocessing_latency_ms",
    "end_to_end_latency_ms", "stt_real_time_factor", "pipeline_real_time_factor",
    "vad_speech_frame_ratio", "peak_rss_mb",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Full_Pipeline_Audio_Benchmark.py",
        description=(
            "OnKith end-to-end audio pipeline benchmark. Runs committed WAV clips "
            "through the real deployment - Silero VAD, Parakeet STT, transcript "
            "normalisation, DeBERTa-v3-xsmall INT8 masking, spoken-digit rendering - "
            "and records latency, real-time factor, memory, word error rate and how "
            "much private information survives imperfect speech recognition."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python Full_Pipeline_Audio_Benchmark.py                  every clip\n"
            "  python Full_Pipeline_Audio_Benchmark.py --smoke-test     four clips\n"
            "  python Full_Pipeline_Audio_Benchmark.py --clip privacy_001\n"
            "  python Full_Pipeline_Audio_Benchmark.py --list-clips\n"
            "\n"
            "exit codes:\n"
            "  0  complete\n"
            "  1  a clip failed, or the privacy artifact did not verify\n"
            "  2  the speech recognizer is unavailable; control arms still ran\n"
        ),
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="score four representative clips instead of the whole set")
    parser.add_argument("--clip", action="append", default=[], metavar="CLIP_ID",
                        help="score only this clip; repeatable")
    parser.add_argument("--subset", choices=("all", "synthetic_privacy", "public"),
                        default="all", help="restrict to one fixture subset")
    parser.add_argument("--list-clips", action="store_true",
                        help="print the fixture inventory and exit")
    parser.add_argument("--verify-only", action="store_true",
                        help="check every clip against its manifest hash and exit")
    parser.add_argument("--skip-stt", action="store_true",
                        help="do not load the recognizer; run the control arms only")
    parser.add_argument("--device-label", default="",
                        help="optional tag in the run id, e.g. pi5 or laptop")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="write the run folder somewhere other than "
                             "research_results/cpu_benchmarks")
    return parser


def list_clips(manifest: Mapping[str, Any]) -> int:
    totals = manifest["totals"]
    print(f"{'clip_id':16} {'subset':18} {'sec':>6}  {'bucket':8} {'entities':>8}  categories")
    print("-" * 96)
    for clip in manifest["clips"]:
        print(f"{clip['clip_id']:16} {clip['subset']:18} "
              f"{clip['duration_seconds']:6.2f}  {clip['duration_bucket']:8} "
              f"{len(clip['expected_private_spans']):>8}  {','.join(clip['category']) or '-'}")
    print("-" * 96)
    print(f"{totals['clips']} clips, {totals['total_duration_seconds']:.1f} s, "
          f"{totals['expected_spans']} expected entities across "
          f"{totals['entity_types']} entity types, "
          f"{totals['hard_negative_clips']} hard negatives")
    print(f"duration buckets: {totals['duration_buckets']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if args.list_clips:
        return list_clips(load_manifest())

    options = AudioBenchmarkOptions(
        smoke_test=args.smoke_test,
        clips=tuple(args.clip),
        subset=args.subset,
        device_label=args.device_label,
        run_stt=not args.skip_stt,
        results_root=args.results_root,
    )

    if args.verify_only:
        manifest = load_manifest()
        clips = select_clips(manifest, options)
        for clip in clips:
            path = verify_clip(clip)
            print(f"OK  {clip['clip_id']:16} {path.stat().st_size:>9,} bytes  "
                  f"{clip['sha256'][:16]}...")
        print(f"\n{len(clips)} clip(s) verified against "
              f"{common.relative(MANIFEST_PATH)}")
        return 0

    return AudioBenchmark(options).execute()


if __name__ == "__main__":
    raise SystemExit(main())
