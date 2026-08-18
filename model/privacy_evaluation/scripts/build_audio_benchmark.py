#!/usr/bin/env python3
"""Build the committed audio benchmark: WAV fixtures plus a hashed manifest.

This runs **once, on a workstation**. The Raspberry Pi never runs it: it gets
ready-made 16 kHz WAV files and reads their SHA-256 out of the manifest.

Two subsets, because no single corpus answers both questions honestly.

``public`` - LibriSpeech dev-clean
    Real human read speech with official transcripts, so the word error rate and
    the real-time factor are measured on natural audio rather than on a
    vocoder's output.

    It is **not annotated for privacy** and never contributes to a privacy
    number. Read prose is full of person names - "Linnell", "Mason", "Polly" -
    so the privacy model detects entities there that are genuinely present but
    have no gold annotation. Counting those as false positives would be wrong,
    and annotating a literary corpus by hand would be worse. The subset earns
    its place by measuring recognition on real human speech; that is all it is
    asked to do.

``synthetic_privacy`` - Piper TTS
    Utterances written for this project, each containing entirely fictional
    private information, so the privacy model has something to detect. Every
    name, number, address and address-like value is invented; the reserved
    ``example.com`` domain (RFC 2606) and the ``555-01xx`` fictional telephone
    range are used throughout. This is a **synthetic privacy speech benchmark**
    and must be labelled as such wherever it is reported.

Licensing
---------

* LibriSpeech is CC BY 4.0 (Panayotov et al., ICASSP 2015). The clips here are
  copied verbatim from the dev-clean validation split.
* Piper is MIT. The three voices used are trained on LJ Speech (public domain),
  LibriTTS-R (CC BY 4.0) and VCTK (CC BY 4.0), so the generated waveforms are
  safe to redistribute with attribution.

Reproducibility
---------------

Piper's vocoder samples noise, so re-running this script produces *different
bytes* for the same sentence. The committed WAV files are therefore the artifact,
and ``audio_benchmark.manifest.json`` pins their SHA-256; the benchmark verifies
every clip against it before measuring anything. Use ``--manifest-only`` to
correct metadata - a mistyped gold span, a new category - against the audio
already on disk, rather than resynthesising and invalidating 8 MB of fixtures.

Usage
-----

    .venv-tts/Scripts/python.exe model/privacy_evaluation/scripts/build_audio_benchmark.py
    .venv-tts/Scripts/python.exe model/privacy_evaluation/scripts/build_audio_benchmark.py --manifest-only

Needs ``piper-tts``, ``scipy``, ``soundfile``, ``pyarrow`` and
``huggingface_hub``. Those belong in a throwaway environment - the benchmark
itself needs none of them, and the deployment venv must not drift.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "model" / "privacy_evaluation" / "fixtures" / "audio_benchmark"
CLIP_DIR = FIXTURE_DIR / "clips"
MANIFEST_PATH = FIXTURE_DIR / "audio_benchmark.manifest.json"

SAMPLE_RATE = 16_000
PAD_SECONDS = 0.25
PEAK_DBFS = -3.0
SCHEMA_VERSION = 1

VOICE_DIR = REPO_ROOT / ".venv-tts" / "piper_voices"
VOICES = {
    "ljspeech": {"model": "en_US-ljspeech-high.onnx", "speaker": None,
                 "dataset": "LJ Speech", "dataset_license": "public domain",
                 "accent": "en_US"},
    "libritts_f": {"model": "en_US-libritts_r-medium.onnx", "speaker": 92,
                   "dataset": "LibriTTS-R", "dataset_license": "CC BY 4.0",
                   "accent": "en_US"},
    "libritts_m": {"model": "en_US-libritts_r-medium.onnx", "speaker": 511,
                   "dataset": "LibriTTS-R", "dataset_license": "CC BY 4.0",
                   "accent": "en_US"},
    "vctk_a": {"model": "en_GB-vctk-medium.onnx", "speaker": 8,
               "dataset": "VCTK", "dataset_license": "CC BY 4.0", "accent": "en_GB"},
    "vctk_b": {"model": "en_GB-vctk-medium.onnx", "speaker": 47,
               "dataset": "VCTK", "dataset_license": "CC BY 4.0", "accent": "en_GB"},
}

LIBRISPEECH_REPO = "hf-internal-testing/librispeech_asr_dummy"
LIBRISPEECH_FILE = "clean/validation-00000-of-00001.parquet"
LIBRISPEECH_CLIPS = 12

# A segment is either plain narration or a ``(label, spoken_text)`` pair. Building
# the transcript and the gold spans from one structure is the only way to be sure
# the offsets are right; hand-written character offsets would rot on the first
# wording change.
Segment = str | tuple[str, str]


def utterance(clip_id: str, voice: str, segments: Sequence[Segment],
              categories: Sequence[str], note: str = "") -> dict[str, Any]:
    return {"clip_id": clip_id, "voice": voice, "segments": list(segments),
            "categories": list(categories), "note": note}


#: Every value below is invented. Telephone numbers use the NANP fictional
#: 555-01xx range, e-mail and web addresses use the RFC 2606 reserved
#: ``example.com`` domain, and no name, address, account or document number
#: refers to a real person or record.
SYNTHETIC: list[dict[str, Any]] = [
    # ---- very short: one clause, entity position varied ---------------------
    utterance("privacy_001", "ljspeech",
              ["My name is ", ("GIVENNAME", "Alex"), " ", ("SURNAME", "Carter"), "."],
              ["GIVENNAME", "SURNAME"], "entity at the end of a short sentence"),
    utterance("privacy_002", "libritts_f",
              [("GIVENNAME", "Priya"), " ", ("SURNAME", "Nandakumar"),
               " is joining the call."],
              ["GIVENNAME", "SURNAME"], "entity at the very start of the utterance"),
    utterance("privacy_003", "vctk_a",
              ["Call me back on ", ("TELEPHONENUM", "five five five, zero one four seven"), "."],
              ["TELEPHONENUM"], "spoken digits, telephone"),
    utterance("benign_001", "ljspeech",
              ["Turn the kitchen lights off, please."],
              [], "no private information at all"),
    utterance("benign_002", "libritts_m",
              ["Set a timer for twenty minutes."],
              [], "hard negative: a benign number that is not private"),

    # ---- short: 4-6 seconds -------------------------------------------------
    utterance("privacy_004", "ljspeech",
              ["My email is ", ("EMAIL", "alex dot carter at example dot com"), "."],
              ["EMAIL"], "spoken e-mail address, the classic STT surface change"),
    utterance("privacy_005", "vctk_b",
              ["Send the package to ", ("BUILDINGNUM", "one forty two"), " ",
               ("STREET", "Cedar Street"), " tomorrow."],
              ["BUILDINGNUM", "STREET"], "address-like phrase in a natural request"),
    utterance("privacy_006", "libritts_f",
              ["My reference number is ",
               ("ACCOUNTNUM", "seven four nine two one eight"), "."],
              ["ACCOUNTNUM"], "spoken digits, reference number"),
    utterance("privacy_007", "ljspeech",
              ["The appointment is on the ", ("DATE", "twelfth of March"),
               " if that still works."],
              ["DATE"], "spoken date"),
    utterance("privacy_008", "libritts_m",
              ["I will be in ", ("CITY", "Manchester"), " at ", ("TIME", "half past four"), "."],
              ["CITY", "TIME"], "city and spoken time in one clause"),
    utterance("benign_003", "vctk_a",
              ["The recipe needs three hundred grams of flour and two eggs."],
              [], "hard negative: quantities that must not be masked"),
    utterance("privacy_009", "ljspeech",
              [("GIVENNAME", "Tomas"), " just turned ", ("AGE", "forty one"),
               " so we should send something."],
              ["GIVENNAME", "AGE"], "name plus age"),
    utterance("privacy_010", "libritts_f",
              ["Doctor ", ("SURNAME", "Okonjo"), " works at ",
               ("ORGANISATION", "Northgate Clinic"), "."],
              ["SURNAME", "ORGANISATION"], "person and organisation together"),
    utterance("benign_004", "libritts_m",
              ["Remind me to water the plants when I get home this evening."],
              [], "ordinary assistant request, no private information"),
    utterance("privacy_011", "vctk_b",
              ["The booking page is ", ("URL", "example dot com slash booking"), "."],
              ["URL"], "spoken URL"),

    # ---- medium: 7-12 seconds ----------------------------------------------
    utterance("privacy_012", "ljspeech",
              ["If you need me urgently, ask for ", ("GIVENNAME", "Alex"), " ",
               ("SURNAME", "Carter"), " and use the mobile number ",
               ("TELEPHONENUM", "five five five, zero one six two"), "."],
              ["GIVENNAME", "SURNAME", "TELEPHONENUM"],
              "several entities, one of them spoken digits"),
    utterance("privacy_013", "libritts_f",
              ["Please forward the invoice to ",
               ("EMAIL", "billing at example dot org"),
               " before the ", ("DATE", "thirtieth of September"), "."],
              ["EMAIL", "DATE"], "e-mail and date in a natural instruction"),
    utterance("privacy_014", "vctk_a",
              ["The delivery address is ", ("BUILDINGNUM", "twenty seven"), " ",
               ("STREET", "Alder Lane"), ", ", ("CITY", "Leeds"), ", postcode ",
               ("ZIPCODE", "L S one, four B N"), "."],
              ["BUILDINGNUM", "STREET", "CITY", "ZIPCODE"],
              "full spoken address, four entity types"),
    utterance("privacy_015", "libritts_m",
              ["The card on the account ends with ",
               ("CREDITCARDNUMBER", "four one one two, six six eight three"),
               " and it expires next year."],
              ["CREDITCARDNUMBER"], "spoken card digits"),
    utterance("privacy_016", "ljspeech",
              ["They asked for my identity card number, which is ",
               ("IDCARDNUM", "X B nine four two seven one"),
               ", before they would release the parcel."],
              ["IDCARDNUM"], "alphanumeric document number"),
    utterance("privacy_017", "vctk_b",
              ["The temporary password for the guest network is ",
               ("PASSWORD", "maple river four four"), ", change it afterwards."],
              ["PASSWORD"], "an actually disclosed credential"),
    utterance("benign_005", "libritts_f",
              ["I keep forgetting the password for that site, so I should probably "
               "use the password manager instead of writing it down."],
              [], "hard negative: talks about passwords without disclosing one"),
    utterance("privacy_018", "libritts_m",
              ["The transfer went out from ", ("BANKNAME", "Harbour Mutual"),
               ", account ", ("ACCOUNTNUM", "three three one nine zero five"),
               ", yesterday afternoon."],
              ["BANKNAME", "ACCOUNTNUM"], "bank name plus spoken account digits"),
    utterance("privacy_019", "vctk_a",
              [("TITLE", "Professor"), " ", ("SURNAME", "Aldridge"),
               " is expecting the draft by ", ("TIME", "nine in the morning"),
               " on ", ("DATE", "Friday"), "."],
              ["TITLE", "SURNAME", "TIME", "DATE"],
              "title, surname and two temporal entities"),
    utterance("benign_006", "ljspeech",
              ["The number seventeen bus takes about forty minutes at this hour, "
               "so it is quicker to walk."],
              [], "hard negative: route and duration numbers, nothing private"),

    # ---- longer: 15-20 seconds ---------------------------------------------
    utterance("privacy_020", "libritts_f",
              ["Just to confirm the details for the visit. The tenant is ",
               ("GIVENNAME", "Rachel"), " ", ("SURNAME", "Whitfield"),
               ", the property is at ", ("BUILDINGNUM", "eighty four"), " ",
               ("STREET", "Beacon Road"),
               ", and the best number to reach her on is ",
               ("TELEPHONENUM", "five five five, zero one three nine"),
               ". She is usually free after ", ("TIME", "six in the evening"), "."],
              ["GIVENNAME", "SURNAME", "BUILDINGNUM", "STREET", "TELEPHONENUM", "TIME"],
              "multi-sentence, six entities across three sentences"),
    utterance("privacy_021", "vctk_b",
              ["I have sent the signed contract over already. It went to ",
               ("EMAIL", "contracts at example dot net"), " from my work account on the ",
               ("DATE", "fourth of June"),
               ", and the reference they gave me was ",
               ("ACCOUNTNUM", "eight zero two five one seven"),
               ", so it should be easy enough to find."],
              ["EMAIL", "DATE", "ACCOUNTNUM"],
              "multi-sentence with an entity in the final clause"),
    utterance("privacy_022", "libritts_m",
              ["The weather looks fine for the weekend, so we will stick with the plan. "
               "Meet ", ("GIVENNAME", "Daniel"),
               " outside the station at about ", ("TIME", "quarter past two"),
               ", and if anything changes he will text you. "
               "Bring the twenty pound note you owe me."],
              ["GIVENNAME", "TIME"],
              "private and non-private information mixed in one clip"),
    utterance("benign_007", "ljspeech",
              ["I have been meaning to reorganise the garage for months now. "
               "There are four boxes of books that should go to the charity shop, "
               "a bicycle that needs a new chain, and far too many empty jars. "
               "Perhaps this weekend."],
              [], "long benign clip, no private information"),
    utterance("privacy_023", "vctk_a",
              [("SURNAME", "Ferreira"),
               " called about the insurance claim this morning. She said the policy "
               "was opened in ", ("CITY", "Bristol"),
               ", the excess is ", ("AMOUNT", "two hundred and fifty pounds"),
               ", and she will write to us again on ",
               "the ", ("DATE", "second of November"), "."],
              ["SURNAME", "CITY", "AMOUNT", "DATE"],
              "entity at the very start and the very end of a long clip"),

    # ---- long: 20-30 seconds ------------------------------------------------
    utterance("privacy_024", "libritts_f",
              ["Here is everything the clinic asked for, so you do not have to call back. "
               "The patient is ", ("GIVENNAME", "Miriam"), " ", ("SURNAME", "Halstead"),
               ", she is ", ("AGE", "sixty three"),
               ", and she is registered with ",
               ("ORGANISATION", "Riverbank Health Centre"), ". "
               "Her contact number is ",
               ("TELEPHONENUM", "five five five, zero one eight eight"),
               ", and the follow up appointment is on ",
               "the ", ("DATE", "nineteenth of January"), " at ",
               ("TIME", "ten past eleven"),
               ". Everything else on the form can stay as it is."],
              ["GIVENNAME", "SURNAME", "AGE", "ORGANISATION", "TELEPHONENUM",
               "DATE", "TIME"],
              "seven entities in one long utterance, worst case for the decoder"),
    utterance("privacy_025", "libritts_m",
              ["We went through the whole checklist twice this afternoon and most of it "
               "was fine. Two of the smoke alarms needed new batteries and the back gate "
               "still sticks. The only thing outstanding is the paperwork for ",
               ("GIVENNAME", "Owen"), " ", ("SURNAME", "Bradshaw"),
               ", whose driving licence number is ",
               ("DRIVERLICENSENUM", "B R A D S nine one one zero four"),
               ". Everything else can wait until the next visit."],
              ["GIVENNAME", "SURNAME", "DRIVERLICENSENUM"],
              "one entity buried late in a long, mostly benign clip"),
]


def build_transcript(segments: Sequence[Segment]) -> tuple[str, list[dict[str, Any]]]:
    """Assemble the reference transcript and its source-aligned gold spans."""

    text_parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in segments:
        if isinstance(segment, str):
            text_parts.append(segment)
            cursor += len(segment)
            continue
        label, value = segment
        spans.append({"label": label, "text": value,
                      "start": cursor, "end": cursor + len(value)})
        text_parts.append(value)
        cursor += len(value)
    text = "".join(text_parts)
    for span in spans:
        assert text[span["start"]:span["end"]] == span["text"], span
    return text, spans


def duration_bucket(seconds: float) -> str:
    if seconds < 5.0:
        return "0-5s"
    if seconds < 10.0:
        return "5-10s"
    if seconds < 20.0:
        return "10-20s"
    return "20-30s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_wav(path: Path, samples, sample_rate: int = SAMPLE_RATE) -> None:
    """16-bit PCM mono. The stdlib ``wave`` module reads it, so the Pi needs nothing."""

    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def synthesise(text: str, voice_key: str, cache: dict[str, Any]):
    """One Piper waveform, resampled to the pipeline's 16 kHz and level-matched."""

    import numpy as np
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    spec = VOICES[voice_key]
    voice = cache.get(spec["model"])
    if voice is None:
        voice = PiperVoice.load(VOICE_DIR / spec["model"])
        cache[spec["model"]] = voice

    config = SynthesisConfig(speaker_id=spec["speaker"], normalize_audio=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        voice.synthesize_wav(text, stream, syn_config=config)
    buffer.seek(0)
    with wave.open(buffer, "rb") as stream:
        rate = stream.getframerate()
        raw = stream.readframes(stream.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0

    if rate != SAMPLE_RATE:
        from math import gcd

        divisor = gcd(SAMPLE_RATE, rate)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio * (10.0 ** (PEAK_DBFS / 20.0) / peak)

    # Digital silence at both ends: the VAD needs a lead-in, and zeros keep the
    # committed bytes reproducible.
    pad = np.zeros(int(PAD_SECONDS * SAMPLE_RATE), dtype=np.float64)
    return np.concatenate((pad, audio, pad))


def build_synthetic(clip_dir: Path) -> list[dict[str, Any]]:
    cache: dict[str, Any] = {}
    records = []
    for index, item in enumerate(SYNTHETIC, start=1):
        text, spans = build_transcript(item["segments"])
        audio = synthesise(text, item["voice"], cache)
        path = clip_dir / f"{item['clip_id']}.wav"
        write_wav(path, audio)
        seconds = len(audio) / SAMPLE_RATE
        spec = VOICES[item["voice"]]
        records.append({
            "clip_id": item["clip_id"],
            "audio_path": path.relative_to(FIXTURE_DIR).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "duration_seconds": round(seconds, 3),
            "duration_bucket": duration_bucket(seconds),
            "sample_rate": SAMPLE_RATE,
            "reference_transcript": text,
            "expected_private_spans": spans,
            "category": item["categories"],
            "source": "synthetic",
            "subset": "synthetic_privacy",
            "privacy_annotated": True,
            "is_hard_negative": not spans,
            "speaker": {"voice": item["voice"], "model": spec["model"],
                        "speaker_id": spec["speaker"], "accent": spec["accent"],
                        "tts_dataset": spec["dataset"],
                        "tts_dataset_license": spec["dataset_license"]},
            "note": item["note"],
        })
        print(f"  [{index:02d}/{len(SYNTHETIC)}] {item['clip_id']:14} "
              f"{seconds:5.2f}s  {len(spans)} span(s)  {item['voice']}")
    return records


def build_public(clip_dir: Path, count: int = LIBRISPEECH_CLIPS) -> list[dict[str, Any]]:
    """A deterministic spread of LibriSpeech dev-clean across duration buckets."""

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LIBRISPEECH_REPO, LIBRISPEECH_FILE, repo_type="dataset")
    rows = pq.read_table(path).to_pylist()

    decoded = []
    for row in rows:
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float64")
        if rate != SAMPLE_RATE:  # dev-clean is already 16 kHz; guard anyway
            continue
        decoded.append((row["id"], row["text"], audio, len(audio) / rate))
    decoded.sort(key=lambda item: item[0])

    # Round-robin over the buckets, taking the shortest unused clip in each, so
    # the selection is a pure function of the corpus rather than of a sample.
    buckets: dict[str, list[tuple]] = {}
    for item in decoded:
        buckets.setdefault(duration_bucket(item[3]), []).append(item)
    for entries in buckets.values():
        entries.sort(key=lambda item: (item[3], item[0]))

    chosen: list[tuple] = []
    order = sorted(buckets)
    while len(chosen) < count and any(buckets[name] for name in order):
        for name in order:
            if len(chosen) >= count:
                break
            if buckets[name]:
                chosen.append(buckets[name].pop(0))
    chosen.sort(key=lambda item: item[0])

    records = []
    for index, (clip_id, text, audio, seconds) in enumerate(chosen, start=1):
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio * (10.0 ** (PEAK_DBFS / 20.0) / peak)
        target = clip_dir / f"public_{index:03d}.wav"
        write_wav(target, audio)
        records.append({
            "clip_id": f"public_{index:03d}",
            "audio_path": target.relative_to(FIXTURE_DIR).as_posix(),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
            "duration_bucket": duration_bucket(len(audio) / SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "reference_transcript": text,
            "expected_private_spans": [],
            "category": [],
            "source": "librispeech_dev_clean",
            "subset": "public",
            "privacy_annotated": False,
            "is_hard_negative": False,
            "speaker": {"librispeech_utterance_id": clip_id},
            "note": ("LibriSpeech dev-clean, verbatim. A real-speech word-error-rate "
                     "and real-time-factor reference. Not annotated for privacy: read "
                     "prose contains person names, so detections here are unscored "
                     "rather than counted as false positives."),
        })
        print(f"  [{index:02d}/{len(chosen)}] public_{index:03d}    "
              f"{len(audio) / SAMPLE_RATE:5.2f}s  <- {clip_id}")
    return records


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    durations = sorted(record["duration_seconds"] for record in records)
    labels: Counter[str] = Counter()
    for record in records:
        for span in record["expected_private_spans"]:
            labels[span["label"]] += 1
    return {
        "clips": len(records),
        "total_duration_seconds": round(sum(durations), 3),
        "total_bytes": sum(record["bytes"] for record in records),
        "duration_seconds": {
            "min": durations[0] if durations else 0.0,
            "median": durations[len(durations) // 2] if durations else 0.0,
            "mean": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "max": durations[-1] if durations else 0.0,
        },
        "duration_buckets": dict(sorted(Counter(r["duration_bucket"] for r in records).items())),
        "clips_with_private_information": sum(
            1 for r in records if r["expected_private_spans"]
        ),
        "privacy_annotated_clips": sum(
            1 for r in records if r.get("privacy_annotated", True)
        ),
        "hard_negative_clips": sum(1 for r in records if r["is_hard_negative"]),
        "expected_spans": int(sum(labels.values())),
        "entity_types": len(labels),
        "spans_by_entity": dict(sorted(labels.items())),
    }


def describe_existing(clip_dir: Path) -> list[dict[str, Any]]:
    """Rebuild the records from the audio already on disk, synthesising nothing.

    Synthetic clips get their transcript and gold spans re-derived from the
    definitions above, so a corrected span reaches the manifest. Public clips
    carry their existing entry forward: their transcript comes from LibriSpeech,
    not from anything in this file.
    """

    previous = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["clips"]
        if MANIFEST_PATH.is_file() else []
    )
    records = []
    for item in SYNTHETIC:
        path = clip_dir / f"{item['clip_id']}.wav"
        if not path.is_file():
            raise SystemExit(f"{path} is missing; run without --manifest-only first")
        text, spans = build_transcript(item["segments"])
        with wave.open(str(path), "rb") as stream:
            seconds = stream.getnframes() / stream.getframerate()
        spec = VOICES[item["voice"]]
        records.append({
            "clip_id": item["clip_id"],
            "audio_path": path.relative_to(FIXTURE_DIR).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "duration_seconds": round(seconds, 3),
            "duration_bucket": duration_bucket(seconds),
            "sample_rate": SAMPLE_RATE,
            "reference_transcript": text,
            "expected_private_spans": spans,
            "category": item["categories"],
            "source": "synthetic",
            "subset": "synthetic_privacy",
            "privacy_annotated": True,
            "is_hard_negative": not spans,
            "speaker": {"voice": item["voice"], "model": spec["model"],
                        "speaker_id": spec["speaker"], "accent": spec["accent"],
                        "tts_dataset": spec["dataset"],
                        "tts_dataset_license": spec["dataset_license"]},
            "note": item["note"],
        })
        print(f"  {item['clip_id']:14} {seconds:5.2f}s  {len(spans)} span(s)")

    carried = [record for record in previous if record["subset"] == "public"]
    for record in carried:
        record["sha256"] = sha256_file(FIXTURE_DIR / record["audio_path"])
        record["privacy_annotated"] = False
        record["is_hard_negative"] = False
        record["note"] = (
            "LibriSpeech dev-clean, verbatim. A real-speech word-error-rate and "
            "real-time-factor reference. Not annotated for privacy: read prose "
            "contains person names, so detections here are unscored rather than "
            "counted as false positives."
        )
    print(f"  carried {len(carried)} public clip(s) forward unchanged")
    return records + carried


def build(*, skip_public: bool = False, skip_synthetic: bool = False,
          manifest_only: bool = False) -> dict[str, Any]:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if manifest_only:
        print("rebuilding the manifest from the committed audio")
        records = describe_existing(CLIP_DIR)
    else:
        if not skip_synthetic:
            print("synthetic privacy speech")
            records += build_synthetic(CLIP_DIR)
        if not skip_public:
            print("\npublic speech (LibriSpeech dev-clean)")
            records += build_public(CLIP_DIR)
    records.sort(key=lambda record: record["clip_id"])

    by_subset = {
        name: [record for record in records if record["subset"] == name]
        for name in sorted({record["subset"] for record in records})
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "audio_benchmark",
        "purpose": (
            "A small, fixed set of 16 kHz WAV clips scored end to end by "
            "Full_Pipeline_Audio_Benchmark.py on every device. It is a controlled "
            "deployment benchmark, not a comprehensive speech-recognition benchmark "
            "and not a replacement for the large held-out text evaluation."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "model/privacy_evaluation/scripts/build_audio_benchmark.py",
        "ontology_version": "model_v2.ontology.1",
        "sample_rate": SAMPLE_RATE,
        "audio_format": "16-bit PCM mono WAV",
        "silence_padding_seconds": PAD_SECONDS,
        "peak_dbfs": PEAK_DBFS,
        "subsets": {
            "synthetic_privacy": {
                "name": "synthetic privacy speech benchmark",
                "source": "Piper TTS (MIT) reading sentences written for this project",
                "voices": VOICES,
                "license": (
                    "Piper is MIT. Voice models are trained on LJ Speech (public "
                    "domain), LibriTTS-R (CC BY 4.0) and VCTK (CC BY 4.0)."
                ),
                "privacy_notice": (
                    "Every name, number, address, credential and document identifier "
                    "is fictional. Telephone numbers use the NANP 555-01xx fictional "
                    "range and web/e-mail addresses use the RFC 2606 reserved "
                    "example domains. No real personal information is present."
                ),
                "privacy_annotated": True,
                "reporting_label": "synthetic privacy speech benchmark",
                "summary": summarise(by_subset.get("synthetic_privacy", [])),
            },
            "public": {
                "name": "LibriSpeech dev-clean subset",
                "source": f"{LIBRISPEECH_REPO} :: {LIBRISPEECH_FILE}",
                "upstream": "LibriSpeech ASR corpus, dev-clean validation split",
                "license": "CC BY 4.0",
                "citation": (
                    "V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "
                    "\"Librispeech: An ASR corpus based on public domain audio books\", "
                    "ICASSP 2015, pp. 5206-5210."
                ),
                "split": "validation (dev-clean)",
                "selection": (
                    "Round-robin over duration buckets, shortest first within each "
                    "bucket, corpus sorted by utterance id. No randomness."
                ),
                "privacy_annotated": False,
                "privacy_notice": (
                    "Not annotated for private information and excluded from every "
                    "privacy metric. Read prose contains person names, so the model "
                    "detects real entities here that carry no gold annotation."
                ),
                "reporting_label": "LibriSpeech dev-clean subset",
                "summary": summarise(by_subset.get("public", [])),
            },
        },
        "totals": summarise(records),
        "clips": records,
    }

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-public", action="store_true",
                        help="build only the synthetic privacy clips")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="build only the LibriSpeech clips")
    parser.add_argument("--manifest-only", action="store_true",
                        help="rewrite the manifest from the audio already on disk, "
                             "synthesising nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = build(skip_public=args.skip_public, skip_synthetic=args.skip_synthetic,
                     manifest_only=args.manifest_only)
    totals = manifest["totals"]
    print(f"\nwrote {MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"  clips              : {totals['clips']}")
    print(f"  total duration     : {totals['total_duration_seconds']:.1f} s")
    print(f"  total bytes        : {totals['total_bytes']:,} "
          f"({totals['total_bytes'] / 1024 ** 2:.1f} MiB)")
    print(f"  duration buckets   : {totals['duration_buckets']}")
    print(f"  clips with PII     : {totals['clips_with_private_information']}")
    print(f"  privacy annotated  : {totals['privacy_annotated_clips']}")
    print(f"  hard negatives     : {totals['hard_negative_clips']}")
    print(f"  expected spans     : {totals['expected_spans']} "
          f"across {totals['entity_types']} entity types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
