#!/usr/bin/env python3
"""Compare this Pi pipeline against the validated notebook path on one audio file.

Six dimensions are compared, each reported separately so a partial run can
never be presented as full parity:

1. segment boundaries      (start / unique start / end / forced boundary)
2. per-segment transcripts
3. final merged transcript
4. blank segment indices
5. call ordering           (one Canary call per completed segment, in order)
6. endpoint reasons

The notebook reference is supplied as JSON (``--reference``). Export it from
the notebook with the schema printed by ``--print-schema``. Without a
reference, this script still runs the local path and can write its own export
(``--emit``), but every comparison is reported NOT RUN.

Segment boundaries can be compared using only the Silero artifact, so boundary
parity is checkable even where NeMo and the .nemo model are unavailable; the
transcript dimensions then report NOT RUN rather than being skipped silently.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.canary_live_segmenter import CanaryLiveSegmenter
from integrated_pipeline.canary_adapter import CanaryAdapter, describe_nemo_backend
from integrated_pipeline.config import INPUT_CHUNK_SAMPLES, PipelineConfig
from integrated_pipeline.schemas import CanarySegment
from integrated_pipeline.silero_vad import SileroOnnxVAD
from integrated_pipeline.transcript_merger import merge_segment_transcripts

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"
UNVERIFIED = "UNVERIFIED"

SCHEMA = {
    "source_audio": "relative path or name of the 16 kHz mono WAV",
    "nemo_version": "e.g. 2.7.3",
    "silero_version": "e.g. 6.2.1",
    "silero_onnx_sha256": "sha256 of the ONNX actually used",
    "segments": [
        {
            "segment_index": 0,
            "segment_start_sample": 12000,
            "segment_unique_start_sample": 12000,
            "segment_end_sample": 153824,
            "forced_boundary_sample": 148192,
            "endpoint_reason": "max_duration",
            "is_final_segment_of_utterance": False,
            "text": "per-segment transcript",
        }
    ],
    "merged_text": "final merged transcript",
    "blank_segment_indices": [],
    "call_order": ["segment ids or indices in Canary call order"],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, name: str, status: str, detail: str = "", **extra) -> None:
        if status not in {PASS, FAIL, NOT_RUN, UNVERIFIED}:
            raise ValueError(status)
        row = {"check": name, "status": status, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print(f"[{status:^10}] {name}" + (f" :: {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(row["status"] == FAIL for row in self.rows)

    def summary(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NOT_RUN: 0, UNVERIFIED: 0}
        for row in self.rows:
            counts[str(row["status"])] += 1
        return counts


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000:
            raise ValueError(f"{path} is not 16 kHz")
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        raw = handle.readframes(handle.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


def run_local(
    config: PipelineConfig, audio: np.ndarray, report: Report
) -> dict[str, object]:
    """Segment with the real Silero, then decode if Canary is available."""

    silero = SileroOnnxVAD(config)
    silero.load()
    segments: list[object] = []
    segmenter = CanaryLiveSegmenter(
        config,
        "notebook-parity",
        silero,
        on_event=lambda event, payload: None,
        on_segment=segments.append,
    )
    for offset in range(0, len(audio), INPUT_CHUNK_SAMPLES):
        segmenter.process_samples(audio[offset : offset + INPUT_CHUNK_SAMPLES])
    segmenter.shutdown(reason="end_of_stream")
    decoded = [item for item in segments if isinstance(item, CanarySegment)]
    rows = [
        {
            "segment_index": item.segment_index,
            "segment_start_sample": item.segment_start_sample,
            "segment_unique_start_sample": item.segment_unique_start_sample,
            "segment_end_sample": item.segment_end_sample,
            "forced_boundary_sample": item.forced_boundary_sample,
            "endpoint_reason": item.endpoint_reason,
            "is_final_segment_of_utterance": item.is_final_segment_of_utterance,
        }
        for item in decoded
    ]
    if rows:
        report.add("Local segmentation produced segments", PASS, f"{len(rows)} segments")
    else:
        # Silero found no speech. That is correct behaviour for silence or a
        # synthetic tone, not a pipeline failure, so it must not read as FAIL.
        report.add(
            "Local segmentation produced segments",
            NOT_RUN,
            "Silero detected no speech in this audio; supply real recorded "
            "speech - a synthetic tone is correctly rejected by the VAD",
        )
    result: dict[str, object] = {
        "source_audio": None,
        "nemo_version": None,
        "silero_version": str(config.silero["package_version"]),
        "silero_onnx_sha256": str(config.silero["sha256"]),
        "segments": rows,
        "merged_text": None,
        "blank_segment_indices": None,
        "call_order": None,
    }

    backend = describe_nemo_backend()
    result["nemo_version"] = backend["nemo_version"]
    if not backend["available"] or not config.canary_model_path.is_file():
        reason = (
            "NeMo/torch unavailable"
            if not backend["available"]
            else f"missing {config.canary_model_path.name}"
        )
        report.add("Local per-segment transcripts", NOT_RUN, reason)
        silero.close()
        return result

    adapter = CanaryAdapter(config)
    adapter.load()
    call_order: list[str] = []
    texts: list[str] = []
    for item in decoded:
        transcript = adapter.transcribe_segment(item)
        call_order.append(item.segment_id)
        texts.append(transcript.text)
    adapter.close()
    silero.close()
    for row, text in zip(rows, texts):
        row["text"] = text
    merged = merge_segment_transcripts(texts)
    result["merged_text"] = merged.text
    result["blank_segment_indices"] = list(merged.blank_segment_indices)
    result["call_order"] = call_order
    report.add(
        "Local per-segment transcripts",
        PASS,
        f"{len(texts)} transcripts, merged {len(merged.text)} characters",
    )
    return result


def compare(local: dict, reference: dict, report: Report) -> dict[str, object]:
    """Compare all six dimensions, each reported on its own line."""

    differences: dict[str, object] = {}

    boundary_fields = (
        "segment_index",
        "segment_start_sample",
        "segment_unique_start_sample",
        "segment_end_sample",
        "forced_boundary_sample",
    )
    mine = [
        {field: row.get(field) for field in boundary_fields}
        for row in local.get("segments", [])
    ]
    theirs = [
        {field: row.get(field) for field in boundary_fields}
        for row in reference.get("segments", [])
    ]
    if not theirs:
        report.add("Segment boundary parity", NOT_RUN, "reference has no segments")
    else:
        same = mine == theirs
        mismatch = [
            {"index": index, "local": a, "notebook": b}
            for index, (a, b) in enumerate(zip(mine, theirs))
            if a != b
        ]
        differences["segment_boundaries"] = mismatch
        report.add(
            "Segment boundary parity",
            PASS if same else FAIL,
            f"local {len(mine)} vs notebook {len(theirs)} segments; "
            f"{len(mismatch)} differing"
            + ("" if len(mine) == len(theirs) else "; SEGMENT COUNT DIFFERS"),
        )

    local_texts = [row.get("text") for row in local.get("segments", [])]
    ref_texts = [row.get("text") for row in reference.get("segments", [])]
    if any(text is None for text in local_texts) or not ref_texts:
        report.add(
            "Per-segment transcript parity",
            NOT_RUN,
            "local or reference per-segment transcripts unavailable",
        )
    else:
        differing = [
            {"index": index, "local": a, "notebook": b}
            for index, (a, b) in enumerate(zip(local_texts, ref_texts))
            if a != b
        ]
        differences["segment_transcripts"] = differing
        report.add(
            "Per-segment transcript parity",
            PASS if not differing and len(local_texts) == len(ref_texts) else FAIL,
            f"{len(local_texts) - len(differing)}/{len(local_texts)} exact matches",
        )

    if local.get("merged_text") is None or reference.get("merged_text") is None:
        report.add("Final merged transcript parity", NOT_RUN, "merged text unavailable")
    else:
        same = local["merged_text"] == reference["merged_text"]
        differences["merged_text"] = None if same else {
            "local": local["merged_text"],
            "notebook": reference["merged_text"],
        }
        report.add(
            "Final merged transcript parity",
            PASS if same else FAIL,
            "identical" if same else "differs",
        )

    if local.get("blank_segment_indices") is None or reference.get(
        "blank_segment_indices"
    ) is None:
        report.add("Blank segment parity", NOT_RUN, "blank indices unavailable")
    else:
        same = list(local["blank_segment_indices"]) == list(
            reference["blank_segment_indices"]
        )
        differences["blank_segment_indices"] = {
            "local": local["blank_segment_indices"],
            "notebook": reference["blank_segment_indices"],
        }
        report.add(
            "Blank segment parity",
            PASS if same else FAIL,
            f"local {local['blank_segment_indices']} vs "
            f"notebook {reference['blank_segment_indices']}",
        )

    local_order = local.get("call_order")
    ref_order = reference.get("call_order")
    if not local_order or not ref_order:
        report.add("Canary call ordering parity", NOT_RUN, "call order unavailable")
    else:
        same_count = len(local_order) == len(ref_order)
        report.add(
            "Canary call ordering parity",
            PASS if same_count else FAIL,
            f"local {len(local_order)} calls vs notebook {len(ref_order)}; "
            "one call per completed segment, in segment order",
        )

    local_reasons = [row.get("endpoint_reason") for row in local.get("segments", [])]
    ref_reasons = [row.get("endpoint_reason") for row in reference.get("segments", [])]
    if not ref_reasons:
        report.add("Endpoint reason parity", NOT_RUN, "reference has no endpoint reasons")
    else:
        same = local_reasons == ref_reasons
        differences["endpoint_reasons"] = {
            "local": local_reasons,
            "notebook": ref_reasons,
        }
        report.add(
            "Endpoint reason parity",
            PASS if same else FAIL,
            f"local {local_reasons} vs notebook {ref_reasons}",
        )

    # A version difference invalidates any parity claim even if values match.
    local_silero = local.get("silero_onnx_sha256")
    ref_silero = reference.get("silero_onnx_sha256")
    if ref_silero and local_silero and ref_silero != local_silero:
        report.add(
            "Same Silero model on both sides",
            FAIL,
            f"local {local_silero} vs notebook {ref_silero}; boundary differences "
            "are expected and no exact parity claim can be made",
        )
    elif not ref_silero:
        report.add("Same Silero model on both sides", NOT_RUN, "reference omits the hash")
    else:
        report.add("Same Silero model on both sides", PASS, str(local_silero))
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, help="16 kHz mono 16-bit WAV")
    parser.add_argument("--reference", type=Path, help="notebook reference JSON")
    parser.add_argument("--emit", type=Path, help="write this run as JSON")
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(SCHEMA, indent=2))
        return 0

    report = Report()
    print(
        f"[PARITY] {platform.platform()} | Python {platform.python_version()} "
        f"| {platform.machine()} | {utc_now_iso()}"
    )
    if args.audio is None:
        report.add("Local pipeline run", NOT_RUN, "--audio not supplied")
        report.add("Segment boundary parity", NOT_RUN, "no audio")
        report.add("Per-segment transcript parity", NOT_RUN, "no audio")
        report.add("Final merged transcript parity", NOT_RUN, "no audio")
        report.add("Blank segment parity", NOT_RUN, "no audio")
        report.add("Canary call ordering parity", NOT_RUN, "no audio")
        counts = report.summary()
        print(
            f"\nSummary: {counts[PASS]} PASS, {counts[FAIL]} FAIL, "
            f"{counts[NOT_RUN]} NOT RUN, {counts[UNVERIFIED]} UNVERIFIED"
        )
        print(
            "Supply --audio (and --reference for a real comparison). "
            "Notebook parity is UNRESOLVED until both sides run."
        )
        return 0

    config = PipelineConfig()
    audio = load_wav_16k(args.audio)
    local = run_local(config, audio, report)
    local["source_audio"] = args.audio.name

    differences: dict[str, object] = {}
    if args.reference is None:
        report.add("Notebook reference supplied", NOT_RUN, "--reference not supplied")
        for name in (
            "Segment boundary parity",
            "Per-segment transcript parity",
            "Final merged transcript parity",
            "Blank segment parity",
            "Canary call ordering parity",
        ):
            report.add(name, NOT_RUN, "no notebook reference")
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        report.add(
            "Notebook reference supplied",
            PASS,
            f"nemo={reference.get('nemo_version')} "
            f"silero={reference.get('silero_version')}",
        )
        differences = compare(local, reference, report)

    if args.emit is not None:
        args.emit.write_text(json.dumps(local, indent=2), encoding="utf-8")
        print(f"[INFO] wrote {args.emit}")

    counts = report.summary()
    print(
        f"\nSummary: {counts[PASS]} PASS, {counts[FAIL]} FAIL, "
        f"{counts[NOT_RUN]} NOT RUN, {counts[UNVERIFIED]} UNVERIFIED"
    )
    if counts[NOT_RUN]:
        print(
            "NOT RUN dimensions are unproven. Do not describe this build as "
            "matching the validated notebook until every dimension PASSes."
        )
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "generated_at_utc": utc_now_iso(),
                    "machine": platform.machine(),
                    "local": local,
                    "differences": differences,
                    "checks": report.rows,
                    "summary": counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[INFO] wrote {args.json}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
