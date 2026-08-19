"""The CPU/text deployment benchmark, as a plain Python runner.

This is the code that used to live in ``Full_Pipeline_CPU_Benchmark.ipynb``. The
Raspberry Pi has no Jupyter, so the notebook became a script; the measurements,
the row selection, the metric definitions, the output file names and the summary
layout are unchanged, which is what makes an already-recorded workstation run
comparable with a Pi run of this module.

What it measures
----------------

============================  ==========================================
Part A - full pipeline        How long one utterance takes to travel
                              through VAD -> STT -> transcript merge ->
                              buffer -> privacy masking on this device.
Part B - privacy model        Whether the accepted DeBERTa-v3-xsmall INT8
                              deployment path produces the expected
                              quality here, and how fast and how large it
                              is.
============================  ==========================================

Part B is the project's research contribution, so it is measured in detail.
Part A shows the deployment is feasible end to end; the STT model is a pinned
third-party component and only its timing matters here. The clip-level audio
evaluation - real WAV files, real transcripts, end-to-end privacy leakage after
speech recognition - is a separate benchmark in :mod:`privacy_eval.audio_benchmark`.

No source text is written to any output file. Predictions are recorded as
offsets, labels and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import benchmark_common as common
from .benchmark_common import (
    RunDirectory,
    WarningLog,
    latency_line,
    print_summary,
    summarise,
)

common.ensure_import_paths()

DEFAULT_BENCHMARK_ROWS = 2000
DEFAULT_SMOKE_ROWS = 50
DEFAULT_WARMUP_RUNS = 3
DEFAULT_LATENCY_PROBE_ROWS = 60
DEFAULT_PIPELINE_WARMUP_RUNS = 2
DEFAULT_PIPELINE_LATENCY_RUNS = 20
DEFAULT_RANDOM_SEED = 20260817

WARMUP_TEXT = (
    "Hi, my name is Sarah Walusiak, my number is 555 0147 and my email is "
    "sarah.walusiak@example.org. Remind me about the appointment on 12 March."
)


@dataclass
class BenchmarkOptions:
    """Everything the run depends on, so ``benchmark_config.json`` is complete."""

    smoke_test: bool = False
    rows: int | None = None
    smoke_rows: int = DEFAULT_SMOKE_ROWS
    benchmark_rows: int = DEFAULT_BENCHMARK_ROWS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    latency_probe_rows: int = DEFAULT_LATENCY_PROBE_ROWS
    run_full_pipeline: bool = True
    pipeline_warmup_runs: int = DEFAULT_PIPELINE_WARMUP_RUNS
    pipeline_latency_runs: int = DEFAULT_PIPELINE_LATENCY_RUNS
    run_ood_dev: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED
    device_label: str = ""
    onnx_intra_op_threads: int | None = None
    results_root: Path | None = None

    #: Resolved in __post_init__ so callers never have to derive them.
    mode: str = field(init=False, default="full")
    target_rows: int = field(init=False, default=DEFAULT_BENCHMARK_ROWS)

    def __post_init__(self) -> None:
        self.mode = "smoke" if self.smoke_test else "full"
        if self.smoke_test:
            # A smoke run checks every code path quickly; the fine-grained probe
            # and the pipeline loop shrink with it.
            self.pipeline_latency_runs = min(self.pipeline_latency_runs, 5)
            self.latency_probe_rows = min(self.latency_probe_rows, 10)
        default_rows = self.smoke_rows if self.smoke_test else self.benchmark_rows
        self.target_rows = int(self.rows) if self.rows is not None else default_rows
        if self.target_rows < 1:
            raise ValueError("--rows must be at least 1")


def spans_of(entities: Iterable[Mapping[str, Any]], model_only: bool = False) -> list[dict]:
    return [
        {"start": int(entity["start"]), "end": int(entity["end"]),
         "label": str(entity["entity_type"])}
        for entity in entities
        if not model_only or not str(entity.get("source", "")).startswith("structured")
    ]


def span_digest(spans: Sequence[Mapping[str, Any]]) -> str:
    payload = ";".join(f"{s['start']}-{s['end']}:{s['label']}" for s in spans)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def deterministic_speech(seconds: float = 3.0, sample_rate: int = 16_000,
                         seed: int = DEFAULT_RANDOM_SEED):
    """Speech-shaped audio from a fixed seed: identical bytes on every device.

    Part A is a *timing* comparison between devices, and VAD/STT cost depends on
    length and energy structure, both of which are pinned here. Real recordings
    are evaluated separately, from committed WAV fixtures, by
    :mod:`privacy_eval.audio_benchmark`.
    """

    import numpy as np

    generator = np.random.default_rng(seed)
    samples = int(seconds * sample_rate)
    time_axis = np.arange(samples, dtype=np.float64) / sample_rate
    # Voiced-ish carrier with formant-like harmonics, amplitude modulated at a
    # syllable rate, plus a little breath noise.
    carrier = sum(
        np.sin(2 * np.pi * frequency * time_axis) / weight
        for frequency, weight in ((120.0, 1.0), (360.0, 2.0), (720.0, 3.0), (1_450.0, 5.0))
    )
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.1 * time_axis))
    noise = generator.normal(0.0, 0.02, samples)
    waveform = 0.25 * carrier * envelope + noise
    silence = int(0.35 * sample_rate)
    waveform[:silence] = noise[:silence] * 0.1
    waveform[-silence:] = noise[-silence:] * 0.1
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


class CpuBenchmark:
    """One run, section by section, in the order the notebook ran them."""

    def __init__(self, options: BenchmarkOptions) -> None:
        self.options = options
        self.warnings = WarningLog()
        self.completed: dict[str, bool] = {}
        self.started_utc = common.utc_now()
        # The engine is part of the run id, so a V4 run can never land in or be
        # mistaken for one of the frozen Canary folders. The historical Canary
        # runs carry no engine token and are identified by its absence.
        self.run_id = common.make_run_id(
            self.started_utc, options.mode, options.device_label, prefix="whisper"
        )
        self.run = RunDirectory(self.run_id, options.results_root)

        self.system_info: dict[str, Any] = {}
        self.baseline_rss: int | None = None
        self.artifact: dict[str, Any] = {}
        self.adapter = None
        self.load_seconds: float | None = None
        self.rss_after_load: int | None = None
        self.warmup_latencies: list[float] = []
        self.peak_rss = 0

        self.benchmark_rows_all: list[dict[str, Any]] = []
        self.benchmark_set: list[dict[str, Any]] = []
        self.subset_manifest: dict[str, Any] = {}
        self.ood_rows: list[dict[str, Any]] = []
        self.pipeline_utterances: list[str] = []

        self.tokenize_probe: dict[str, Any] = {}
        self.eval_rows: list[dict[str, Any]] = []
        self.parity_records: list[dict[str, Any]] = []
        self.row_latency: list[float] = []
        self.stage_latency: dict[str, list[float]] = {
            "neural_decode": [], "structured": [], "span_merge": []
        }
        self.failures: list[dict[str, str]] = []
        self.eval_seconds = 0.0
        self.deberta_latency: dict[str, Any] = {}
        self.evaluation: dict[str, Any] = {}
        self.per_entity: list[dict[str, Any]] = []
        self.ood_result: dict[str, Any] = {}
        self.pipeline_stages: dict[str, Any] = {}
        self.pipeline_rows: list[dict[str, Any]] = []
        self.pipeline_end_to_end: list[float] = []
        self.pipeline_summary: dict[str, Any] = {}
        self.memory: dict[str, Any] = {}
        self.predictions_digest = ""
        self.audio_seconds = 0.0

    # -- 1. hardware ------------------------------------------------------

    def collect_system(self) -> None:
        print("\n" + "=" * 72)
        print(f"mode        : {self.options.mode.upper()}  ({self.options.target_rows} rows)")
        print(f"run id      : {self.run_id}")
        print(f"output      : {self.run.relative_path}")
        print(f"started     : {self.started_utc.isoformat()}")
        print("=" * 72 + "\n")
        self.system_info = common.collect_system_info(self.run_id, self.started_utc)
        self.baseline_rss = self.system_info["baseline_process_rss_bytes"]
        self.peak_rss = self.baseline_rss or 0
        common.print_system_info(self.system_info)

    # -- 2. artifact identity ---------------------------------------------

    def verify_artifacts(self) -> None:
        from integrated_pipeline.config import PipelineConfig

        self.config = PipelineConfig(root=common.PRODUCTION_ROOT)
        print("\n--- Artifact identity ---")
        self.artifact = common.verify_privacy_artifacts(
            self.config, self.run_id, self.warnings
        )
        common.print_artifact_identity(self.config, self.artifact)
        if not self.artifact["ok"]:
            self.warnings.note("ARTIFACT IDENTITY FAILED - the privacy benchmark will be skipped")
        else:
            print("\nThis is the accepted production artifact. "
                  "Results are comparable across devices.")

    # -- 3. rows ----------------------------------------------------------

    def load_rows(self) -> None:
        from privacy_eval import deployment_benchmark as bench
        from privacy_eval.schema import load_cases

        self.bench = bench
        self.benchmark_rows_all, self.subset_manifest = bench.load_subset()
        self.benchmark_set = bench.take(self.benchmark_rows_all, self.options.target_rows)

        # A prefix of 50 rows can easily contain none of the four zero-PII rows,
        # which would leave the hard-negative false-positive rate with no
        # denominator. Add them back; they are a fixed, named set, so the smoke
        # selection stays exact.
        if len(self.benchmark_set) < len(self.benchmark_rows_all):
            present = {row["row_id"] for row in self.benchmark_set}
            self.benchmark_set += [
                row for row in self.benchmark_rows_all
                if not row["privacy_mask"] and row["row_id"] not in present
            ]

        print("\n--- Benchmark rows ---")
        print(f"benchmark subset : {common.relative(bench.SUBSET_PATH)}")
        print(f"  sha256         : {self.subset_manifest['fixture_sha256']}")
        print(f"  available rows : {len(self.benchmark_rows_all):,}")
        print(f"  scoring        : {len(self.benchmark_set):,} rows ({self.options.mode} mode)")
        coverage = self.subset_manifest["coverage"]
        print(f"  entity types   : {coverage['entity_types_present']}")
        print(f"  zero-PII rows  : {coverage['rows_without_pii']}")
        print(f"  source         : {self.subset_manifest['source']['path']} "
              f"({self.subset_manifest['source']['view']} split)")

        ood_path = common.EVALUATION_ROOT / "fixtures" / "ood_dev.jsonl"
        all_dev = sorted(load_cases(ood_path), key=lambda case: case.case_id)

        if self.options.run_ood_dev:
            self.ood_rows = [
                {
                    "row_id": case.case_id,
                    "source_text": case.text,
                    "privacy_mask": [
                        {"start": entity.start, "end": entity.end,
                         "label": entity.entity_type, "value": entity.text}
                        for entity in case.entities
                    ],
                    "category": case.category,
                    "tags": list(case.tags),
                }
                for case in all_dev
            ]
            if self.options.smoke_test:
                # A prefix would be one category; a stride keeps every category
                # and the hard negatives, so the smoke number is at least the
                # right shape. It is still a subset, so it is not expected to
                # equal the release figure.
                stride = max(1, len(self.ood_rows) // 40)
                self.ood_rows = self.ood_rows[::stride][:40]
            print(f"\nOOD dev cases    : {len(self.ood_rows)}")

        # Deterministic spoken-style utterances for the assembled pipeline,
        # taken from the committed OOD dev fixture so every device pushes the
        # same words through.
        self.pipeline_utterances = [
            case.text
            for case in all_dev
            if "speech_like" in case.tags and 40 <= len(case.text) <= 220
        ][: self.options.pipeline_latency_runs]
        print(f"pipeline inputs  : {len(self.pipeline_utterances)} spoken-style utterances")

    # -- 4. load the production privacy path ------------------------------

    def load_privacy_model(self) -> None:
        from integrated_pipeline.pii_adapter import PIIMaskerAdapter

        print("\n--- Production privacy path ---")
        if not self.artifact["ok"]:
            self.warnings.note("privacy model not loaded because artifact identity failed")
            return

        self.adapter = PIIMaskerAdapter(self.config)
        started = time.perf_counter()
        self.adapter.load()
        self.load_seconds = time.perf_counter() - started
        self.rss_after_load = common.rss_bytes()
        print(f"model load          : {self.load_seconds:.2f} s")
        if self.rss_after_load and self.baseline_rss:
            print(f"RSS after load      : {self.rss_after_load / 1024 ** 2:.1f} MiB "
                  f"(+{(self.rss_after_load - self.baseline_rss) / 1024 ** 2:.1f} MiB "
                  "over baseline)")

        for _ in range(self.options.warmup_runs):
            started = time.perf_counter()
            self.adapter.mask_text(WARMUP_TEXT)
            self.warmup_latencies.append(time.perf_counter() - started)
        print("warm-up (excluded)  : "
              + ", ".join(f"{value * 1000:.1f} ms" for value in self.warmup_latencies))

        demo = self.adapter.mask_text(WARMUP_TEXT)
        print("\nproduction masking, one example:")
        print(" ", demo.masked_text)
        print(f"  {len(demo.entities)} entities, "
              f"{demo.raw_model_output['window_count']} window(s), "
              f"{demo.raw_model_output['label_count']} labels")

    # -- 5. fine-grained stage probe --------------------------------------

    def probe_stages(self) -> None:
        if self.adapter is None:
            return
        print("\n--- Privacy-model stage latency ---")
        masker = self.adapter._masker  # the object the adapter already loaded
        probe_rows = self.benchmark_set[: self.options.latency_probe_rows]
        tokenize_seconds, decode_seconds, structured_seconds, token_counts = [], [], [], []

        for row in probe_rows:
            text = row["source_text"]
            started = time.perf_counter()
            masker._windows(text)
            tokenize_seconds.append(time.perf_counter() - started)

            started = time.perf_counter()
            decoding = masker.decode(text)
            decode_seconds.append(time.perf_counter() - started)
            token_counts.append(decoding.token_count)

            started = time.perf_counter()
            self.adapter._structured.find_spans(text)
            structured_seconds.append(time.perf_counter() - started)

        inference_seconds = [
            max(0.0, decode - tokenize)
            for decode, tokenize in zip(decode_seconds, tokenize_seconds)
        ]
        self.tokenize_probe = {
            "rows": len(probe_rows),
            "note": (
                "Tokenization is measured directly; 'inference_and_consolidation' is "
                "decode() minus tokenization, because the ONNX call and the word "
                "consolidation share one pass in the production decoder."
            ),
            "tokenization": summarise(tokenize_seconds),
            "decode_total": summarise(decode_seconds),
            "inference_and_consolidation": summarise(inference_seconds),
            "structured_recognizer": summarise(structured_seconds),
            "source_tokens": {
                "mean": statistics.fmean(token_counts) if token_counts else 0,
                "min": min(token_counts) if token_counts else 0,
                "max": max(token_counts) if token_counts else 0,
            },
        }
        print(f"fine-grained probe over {len(probe_rows)} rows "
              f"(mean {self.tokenize_probe['source_tokens']['mean']:.0f} source tokens)\n")
        print_summary("tokenize + window", self.tokenize_probe["tokenization"])
        print_summary("ONNX + consolidate", self.tokenize_probe["inference_and_consolidation"])
        print_summary("decode() total", self.tokenize_probe["decode_total"])
        print_summary("structured regex", self.tokenize_probe["structured_recognizer"])

    # -- 6. the evaluation pass -------------------------------------------

    def evaluate(self) -> None:
        from privacy_eval.dataset_metrics import evaluate_spans

        print("\n--- Deployment evaluation ---")
        if self.adapter is None:
            self.completed["deberta_benchmark"] = False
            return

        production_predictions: list[list[dict]] = []
        model_predictions: list[list[dict]] = []

        print(f"scoring {len(self.benchmark_set):,} rows ...")
        loop_started = time.perf_counter()
        for index, row in enumerate(self.benchmark_set, start=1):
            text = row["source_text"]
            started = time.perf_counter()
            try:
                result = self.adapter.mask_text(text)
            except Exception as exc:  # a failed row must not lose the whole benchmark
                self.failures.append(
                    {"row_id": row["row_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            elapsed = time.perf_counter() - started

            production = spans_of(result.entities)
            model_only = spans_of(result.entities, model_only=True)
            raw = result.raw_model_output

            self.eval_rows.append(row)
            production_predictions.append(production)
            model_predictions.append(model_only)
            self.row_latency.append(elapsed)
            self.stage_latency["neural_decode"].append(float(raw["pii_model_latency_seconds"]))
            self.stage_latency["structured"].append(float(raw["structured_latency_seconds"]))
            self.stage_latency["span_merge"].append(float(raw["span_merger_latency_seconds"]))

            # Offsets, labels and hashes only. No source text leaves this loop.
            self.parity_records.append({
                "row_id": row["row_id"],
                "characters": len(text),
                "windows": int(raw["window_count"]),
                "bio_repairs": int(raw["bio_repairs"]),
                "production_spans": [[s["start"], s["end"], s["label"]] for s in production],
                "production_digest": span_digest(production),
                "model_digest": span_digest(model_only),
                "masked_text_sha256": hashlib.sha256(
                    result.masked_text.encode("utf-8")
                ).hexdigest()[:16],
                "latency_ms": elapsed * 1000.0,
            })

            if index % 25 == 0 or index == len(self.benchmark_set):
                current = common.rss_bytes()
                if current:
                    self.peak_rss = max(self.peak_rss, current)
            if index % 250 == 0:
                rate = index / (time.perf_counter() - loop_started)
                print(f"  {index:>6,} / {len(self.benchmark_set):,}  ({rate:.1f} rows/s)")

        self.eval_seconds = time.perf_counter() - loop_started
        print(f"\nscored {len(self.eval_rows):,} rows in {self.eval_seconds:.1f} s "
              f"({len(self.eval_rows) / self.eval_seconds:.2f} rows/s)")
        if self.failures:
            self.warnings.note(
                f"{len(self.failures)} row(s) raised during masking; see evaluation_metrics.json"
            )
        self.completed["deberta_benchmark"] = True

        if not self.eval_rows:
            return

        self.deberta_latency = {
            "rows": len(self.row_latency),
            "mode": self.options.mode,
            "load_seconds": self.load_seconds,
            "warmup_runs": self.options.warmup_runs,
            "warmup_ms": [value * 1000.0 for value in self.warmup_latencies],
            "wall_clock_seconds": self.eval_seconds,
            "throughput_rows_per_second": (
                len(self.eval_rows) / self.eval_seconds if self.eval_seconds else None
            ),
            "end_to_end_masking_call": summarise(self.row_latency),
            "stages": {name: summarise(values) for name, values in self.stage_latency.items()},
            "fine_grained_probe": self.tokenize_probe,
        }

        production_metrics = evaluate_spans(self.eval_rows, production_predictions)
        model_metrics = evaluate_spans(self.eval_rows, model_predictions)
        self.evaluation = {
            "run_id": self.run_id,
            "mode": self.options.mode,
            "view": "deployment_benchmark_subset",
            "rows": len(self.eval_rows),
            "subset_sha256": self.subset_manifest["fixture_sha256"],
            "int8_sha256": self.artifact["measured_sha256"],
            "failures": self.failures,
            "production_path": production_metrics,
            "model_attributed": model_metrics,
        }

        for label, entry in sorted(production_metrics["per_label"].items()):
            self.per_entity.append({
                "view": "deployment_benchmark_subset",
                "entity": label,
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1": entry["f1"],
                "support": entry["support"],
                "predicted": entry["predicted"],
                "true_positives": entry["true_positives"],
            })

        print("\nDeployment benchmark - production path")
        print(f"  rows                              {production_metrics['rows']:,}")
        print(f"  typed entity F1                   {production_metrics['typed_entity']['f1']:.4f}")
        print(f"  typed entity precision            {production_metrics['typed_entity']['precision']:.4f}")
        print(f"  typed entity recall               {production_metrics['typed_entity']['recall']:.4f}")
        print(f"  binary entity F1                  {production_metrics['binary_entity']['f1']:.4f}")
        print(f"  private character recall          {production_metrics['private_character_recall']:.4f}")
        print(f"  private character leakage         {production_metrics['private_character_leakage']:.4f}")
        print(f"  non-private overmasking rate      {production_metrics['non_private_overmasking_rate']:.4f}")
        print(f"  hard-negative false positive rate {production_metrics['hard_negative_false_positive_rate']:.4f}"
              f"  ({production_metrics['hard_negative_rows']} zero-PII rows)")
        print(f"\n  model-attributed typed F1         {model_metrics['typed_entity']['f1']:.4f}")

        print("\nlatency")
        print_summary("full masking call", self.deberta_latency["end_to_end_masking_call"])
        for name in ("neural_decode", "structured", "span_merge"):
            print_summary(name, self.deberta_latency["stages"][name])

        weakest = sorted(
            (entry for entry in self.per_entity if entry["support"] >= 5),
            key=lambda entry: entry["f1"],
        )[:5]
        print("\nweakest entity types with at least 5 gold spans:")
        for entry in weakest:
            print(f"  {entry['entity']:20} F1 {entry['f1']:.4f}  (support {entry['support']})")

    # -- 7. out-of-distribution cross-check --------------------------------

    def cross_check_ood(self) -> None:
        from integrated_pipeline.model_v2_masker import ModelV2OnnxMasker
        from privacy_eval.dataset_metrics import evaluate_spans

        if not (self.ood_rows and self.adapter is not None):
            self.completed["ood_dev_crosscheck"] = False
            if self.options.run_ood_dev and self.adapter is None:
                self.warnings.note("ood_dev cross-check skipped: privacy model unavailable")
            return

        print("\n--- Out-of-distribution cross-check ---")
        # Model-only, exactly as the release measured it: no structured recognizer.
        model_only_masker = ModelV2OnnxMasker(
            self.config.pii_model_path,
            self.config.pii_tokenizer_path,
            self.artifact["label_map"]["id2label"],
            context_limit=int(self.config.pii["context_limit"]),
            overflow_stride=int(self.config.pii["overflow_stride"]),
            intra_op_threads=self.options.onnx_intra_op_threads,
        )
        model_only_masker.load()
        model_only_masker.mask_text(WARMUP_TEXT)  # warm-up, excluded

        started = time.perf_counter()
        ood_predictions, ood_latency = [], []
        for row in self.ood_rows:
            row_started = time.perf_counter()
            result = model_only_masker.mask_text(row["source_text"])
            ood_latency.append(time.perf_counter() - row_started)
            ood_predictions.append([
                {"start": int(entity["start"]), "end": int(entity["end"]),
                 "label": str(entity["entity_type"])}
                for entity in result.entities
            ])
        ood_seconds = time.perf_counter() - started
        ood_metrics = evaluate_spans(self.ood_rows, ood_predictions)

        released = self.artifact["release_metrics"].get("ood_dev", {})
        delta = (
            ood_metrics["typed_entity"]["f1"] - released["typed_entity_f1"]
            if released else None
        )
        self.ood_result = {
            "rows": len(self.ood_rows),
            "complete_fixture": len(self.ood_rows) == 360,
            "seconds": ood_seconds,
            "metrics": ood_metrics,
            "latency": summarise(ood_latency),
            "release_reference": released or None,
            "typed_f1_delta_vs_release": delta,
            "parity": [
                {"case_id": row["row_id"],
                 "digest": span_digest(prediction),
                 "category": row.get("category")}
                for row, prediction in zip(self.ood_rows, ood_predictions)
            ],
        }

        print(f"ood_dev, model-only path, {len(self.ood_rows)} cases in {ood_seconds:.1f} s")
        print(f"  typed entity F1                   {ood_metrics['typed_entity']['f1']:.4f}")
        print(f"  private character recall          {ood_metrics['private_character_recall']:.4f}")
        print(f"  hard-negative false positive rate {ood_metrics['hard_negative_false_positive_rate']:.4f}")
        if released:
            print(f"\n  release recorded typed F1         {released['typed_entity_f1']:.4f}")
            print(f"  delta                             {delta:+.4f}")
            if len(self.ood_rows) != 360:
                print("  (smoke mode scores a subset, so a delta here is expected)")
            elif abs(delta) > 1e-4:
                self.warnings.note(
                    f"ood_dev typed F1 differs from the release number by {delta:+.4f}"
                )
            else:
                print("  the deployment path reproduces the release number exactly")
        self.completed["ood_dev_crosscheck"] = True

    # -- 8. part A: the assembled pipeline ---------------------------------

    def _record_stage(self, name: str, samples: Sequence[float], *, available: bool = True,
                      reason: str = "", extra: Mapping[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"available": available, "reason": reason}
        if available:
            entry.update(summarise(samples))
        if extra:
            entry.update(extra)
        self.pipeline_stages[name] = entry

    def run_pipeline(self) -> None:
        print("\n--- Full pipeline (part A) ---")
        audio = deterministic_speech(seed=self.options.random_seed)
        self.audio_seconds = len(audio) / common.SAMPLE_RATE

        self._stage_vad(audio)
        self._stage_stt(audio)
        self._stage_text_path()
        self._assemble_pipeline_summary()

    def _stage_vad(self, audio) -> None:
        if not self.options.run_full_pipeline:
            self._record_stage("vad_silero", [], available=False,
                               reason="full pipeline disabled")
            return
        try:
            from integrated_pipeline.silero_vad import SileroOnnxVAD

            vad = SileroOnnxVAD(self.config)
            vad.load()
            frame_samples = int(self.config.silero["frame_samples"])
            frames = [
                audio[start:start + frame_samples]
                for start in range(0, len(audio) - frame_samples + 1, frame_samples)
            ]
            for _ in range(self.options.pipeline_warmup_runs):
                vad.reset()
                for frame in frames:
                    vad.probability(frame)

            vad_seconds, speech_frames = [], 0
            threshold = float(self.config.silero["threshold"])
            for _ in range(max(3, self.options.pipeline_latency_runs // 4)):
                vad.reset()
                started = time.perf_counter()
                probabilities = [vad.probability(frame) for frame in frames]
                vad_seconds.append(time.perf_counter() - started)
                speech_frames = sum(1 for value in probabilities if value >= threshold)
            vad.close()

            per_frame = [value / len(frames) for value in vad_seconds]
            self._record_stage(
                "vad_silero", vad_seconds,
                extra={
                    "audio_seconds": self.audio_seconds,
                    "frames_per_pass": len(frames),
                    "speech_frames_last_pass": speech_frames,
                    "per_frame_ms": summarise(per_frame),
                    "real_time_factor": statistics.fmean(vad_seconds) / self.audio_seconds,
                },
            )
            print(f"VAD  : {len(frames)} frames over {self.audio_seconds:.1f} s audio, "
                  f"mean {statistics.fmean(vad_seconds) * 1000:.1f} ms per pass "
                  f"(RTF {self.pipeline_stages['vad_silero']['real_time_factor']:.4f})")
        except Exception as exc:
            self._record_stage("vad_silero", [], available=False,
                               reason=f"{type(exc).__name__}: {exc}")
            self.warnings.note(f"VAD stage unavailable: {type(exc).__name__}: {exc}")

    def _stage_stt(self, audio) -> None:
        # The stage key names the engine. The frozen Canary runs wrote
        # "stt_canary" and are never rewritten, so a reader - and the figures
        # notebook - can tell which recognizer a result folder measured from the
        # key alone.
        if not self.options.run_full_pipeline:
            self._record_stage("stt_whisper", [], available=False,
                               reason="full pipeline disabled")
            return
        try:
            from integrated_pipeline.whisper_adapter import (
                WhisperAdapter,
                describe_whisper_backend,
            )

            backend = describe_whisper_backend()
            if not backend.get("available"):
                raise RuntimeError(
                    "faster-whisper backend unavailable: "
                    + "; ".join(backend.get("errors", ["unknown"]))
                )

            common.quieten_stt()
            whisper = WhisperAdapter(self.config)
            whisper.load()

            def make_segment(index: int):
                return common.make_canary_segment(
                    audio,
                    session_id="benchmark",
                    utterance_id="benchmark-utterance",
                    segment_id=f"benchmark-segment-{index}",
                    sequence_number=index + 1,
                )

            for index in range(self.options.pipeline_warmup_runs):
                whisper.transcribe_segment(make_segment(index))

            stt_seconds = []
            transcript = ""
            for index in range(max(3, self.options.pipeline_latency_runs // 4)):
                started = time.perf_counter()
                transcript = whisper.transcribe_segment(
                    make_segment(100 + index)
                ).text
                stt_seconds.append(time.perf_counter() - started)
            whisper.close()

            self._record_stage("stt_whisper", stt_seconds, extra={
                "audio_seconds": self.audio_seconds,
                "real_time_factor": statistics.fmean(stt_seconds) / self.audio_seconds,
                "engine": "whisper",
                "model": self.config.whisper["model"],
                "runtime": self.config.whisper["runtime"],
                "runtime_version": backend.get("faster_whisper_version"),
                "device": self.config.whisper["device"],
                "compute_type": self.config.whisper["compute_type"],
                "beam_size": self.config.whisper["beam_size"],
                # The input is `deterministic_speech`: a speech-shaped tone, not
                # speech. It is identical bytes on every device, which is what
                # made it a fair cross-device timing probe for Canary.
                "condition": "synthetic_tone",
                "probe_transcript": transcript,
                "representative_of_deployment": False,
                "note": (
                    "NOT the deployment cost of this recognizer. The probe is a "
                    "synthetic tone; Whisper has no words to find in it, decodes "
                    "against its own prompt and runs until the 30 s window is "
                    "full, so this real-time factor measures a failure mode "
                    "rather than transcription. Canary had no such loop, which "
                    "is why the same probe was a fair timing comparison for it "
                    "and is not one between the two engines. The live pipeline "
                    "never submits unvoiced audio: Silero gates it first. For "
                    "the recognizer cost on real speech, use "
                    "audio_latency_summary.json from "
                    "Full_Pipeline_Audio_Benchmark.py, which drives 44 committed "
                    "WAV clips. This stage is kept because it is still a valid "
                    "device-to-device comparison for one engine, and because the "
                    "assembled-utterance estimate below is built from it."
                ),
            })
            print(f"STT  : mean {statistics.fmean(stt_seconds) * 1000:.0f} ms for "
                  f"{self.audio_seconds:.1f} s "
                  f"(RTF {self.pipeline_stages['stt_whisper']['real_time_factor']:.3f})")
            print("       synthetic tone, not speech - Whisper decodes against its "
                  "own prompt here.")
            print("       The recognizer's cost on real speech is in "
                  "Full_Pipeline_Audio_Benchmark.py.")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._record_stage("stt_whisper", [], available=False, reason=reason)
            self.warnings.note(f"STT stage unavailable - {reason}")
            print("STT  : unavailable, the remaining stages still run")

    def _stage_text_path(self) -> None:
        if not (self.options.run_full_pipeline and self.pipeline_utterances):
            self.completed["pipeline_text_path"] = False
            if self.options.run_full_pipeline:
                self.warnings.note(
                    "pipeline text path skipped: no deterministic utterances resolved"
                )
            return

        from app.digit_normalizer import normalize_spoken_digits
        from app.wake_word_normalizer import normalize_wake_word
        from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
        from integrated_pipeline.schemas import TranscriptFragment
        from integrated_pipeline.transcript_merger import merge_segment_transcripts

        def split_into_segments(text: str) -> list[str]:
            """Two overlapping ASR segments, the shape the merger is built for."""

            words = text.split()
            if len(words) < 8:
                return [text]
            cut = len(words) // 2
            overlap = min(4, cut)
            return [" ".join(words[: cut + overlap]), " ".join(words[cut:])]

        merge_seconds, wake_seconds, buffer_seconds = [], [], []
        mask_seconds, digit_seconds, end_to_end = [], [], []

        for _ in range(self.options.pipeline_warmup_runs):
            for text in self.pipeline_utterances[:2]:
                merged = merge_segment_transcripts(split_into_segments(text))
                normalize_wake_word(merged.text)
                if self.adapter is not None:
                    self.adapter.mask_text(merged.text)

        for index, text in enumerate(self.pipeline_utterances):
            segments = split_into_segments(text)
            utterance_started = time.perf_counter()

            started = time.perf_counter()
            merged = merge_segment_transcripts(segments)
            merge_seconds.append(time.perf_counter() - started)

            started = time.perf_counter()
            wake_text, _variant = normalize_wake_word(merged.text)
            wake_seconds.append(time.perf_counter() - started)

            started = time.perf_counter()
            buffer = InactivityTranscriptBuffer(
                inactivity_seconds=self.config.buffer_inactivity_seconds,
                maximum_tokens=self.config.maximum_buffer_tokens,
                maximum_duration_seconds=self.config.maximum_buffer_duration_seconds,
            )
            buffer.add_fragment(
                TranscriptFragment(
                    fragment_id=f"benchmark-{index}",
                    text=wake_text,
                    arrival_time_seconds=0.0,
                    source_audio_id=f"benchmark-audio-{index}",
                    session_id="benchmark",
                    batch_id=f"benchmark-{index}",
                    sequence_number=index,
                )
            )
            flushed = buffer.flush(0.0, reason="manual")
            buffer_seconds.append(time.perf_counter() - started)
            buffered_text = flushed.text if flushed else wake_text

            masked_text = buffered_text
            if self.adapter is not None:
                started = time.perf_counter()
                masked = self.adapter.mask_text(buffered_text)
                mask_seconds.append(time.perf_counter() - started)
                masked_text = masked.masked_text

            started = time.perf_counter()
            normalize_spoken_digits(masked_text)
            digit_seconds.append(time.perf_counter() - started)

            total = time.perf_counter() - utterance_started
            end_to_end.append(total)
            self.pipeline_rows.append({
                "utterance_index": index,
                "characters": len(text),
                "words": len(text.split()),
                "transcript_merge_ms": merge_seconds[-1] * 1000.0,
                "wake_word_ms": wake_seconds[-1] * 1000.0,
                "inactivity_buffer_ms": buffer_seconds[-1] * 1000.0,
                "privacy_masking_ms": (mask_seconds[-1] * 1000.0) if mask_seconds else None,
                "digit_normalisation_ms": digit_seconds[-1] * 1000.0,
                "text_path_total_ms": total * 1000.0,
            })

        self._record_stage("transcript_merge", merge_seconds)
        self._record_stage("wake_word_normalisation", wake_seconds)
        self._record_stage("inactivity_buffer", buffer_seconds)
        if mask_seconds:
            self._record_stage("privacy_masking", mask_seconds)
        else:
            self._record_stage("privacy_masking", [], available=False,
                               reason="privacy model unavailable")
        self._record_stage("digit_normalisation", digit_seconds)
        self.pipeline_end_to_end = end_to_end

        print(f"text path over {len(self.pipeline_utterances)} utterances\n")
        for name in ("transcript_merge", "wake_word_normalisation", "inactivity_buffer",
                     "privacy_masking", "digit_normalisation"):
            if self.pipeline_stages[name]["available"]:
                print_summary(name, self.pipeline_stages[name])
        print_summary("text path end to end", summarise(end_to_end))
        self.completed["pipeline_text_path"] = True

    def _assemble_pipeline_summary(self) -> None:
        available_audio = [
            name for name in ("vad_silero", "stt_whisper")
            if self.pipeline_stages.get(name, {}).get("available")
        ]
        self.pipeline_summary = {
            "run_id": self.run_id,
            "mode": self.options.mode,
            "audio_seconds": self.audio_seconds,
            "utterances": len(self.pipeline_utterances),
            "stages": self.pipeline_stages,
            "text_path_end_to_end": (
                summarise(self.pipeline_end_to_end) if self.pipeline_end_to_end
                else {"count": 0}
            ),
            "complete": bool(self.pipeline_end_to_end) and len(available_audio) == 2,
            "audio_stages_available": available_audio,
        }
        self.completed["full_pipeline"] = self.pipeline_summary["complete"]

        if not self.pipeline_end_to_end:
            return
        audio_mean_ms = sum(
            self.pipeline_stages[name]["mean_ms"] for name in available_audio
        )
        self.pipeline_summary["full_utterance_estimate_ms"] = (
            audio_mean_ms + self.pipeline_summary["text_path_end_to_end"]["mean_ms"]
        )
        unrepresentative = sorted(
            name for name in available_audio
            if self.pipeline_stages.get(name, {}).get(
                "representative_of_deployment", True
            ) is False
        )
        note = (
            "Mean of the available audio stages plus the measured text path. "
            f"Audio stages included: {available_audio or 'none'}."
        )
        if unrepresentative:
            note += (
                " This estimate is dominated by "
                f"{', '.join(unrepresentative)}, which measured a synthetic tone "
                "rather than speech and says so in its own entry; it is a "
                "device-to-device figure for one recognizer, not a deployment "
                "latency. The end-to-end cost on real speech is in "
                "audio_latency_summary.json."
            )
        self.pipeline_summary["full_utterance_estimate_note"] = note
        self.pipeline_summary["full_utterance_estimate_representative"] = (
            not unrepresentative
        )
        print(f"\nassembled estimate for one {self.audio_seconds:.1f} s utterance: "
              f"{self.pipeline_summary['full_utterance_estimate_ms']:.1f} ms")
        print(f"  audio stages included: {available_audio or 'none'}")
        if not self.pipeline_summary["complete"]:
            print("  incomplete: at least one audio stage could not run on this machine")

    # -- 9. memory ---------------------------------------------------------

    def collect_memory(self) -> None:
        print("\n--- Memory ---")
        int8_path = self.config.pii_model_path
        tokenizer_json = self.config.pii_tokenizer_path / "tokenizer.json"
        self.memory = common.collect_memory(
            self.run_id, baseline_rss=self.baseline_rss, peak_rss=self.peak_rss,
        )
        self.memory.pop("stage_rss_bytes", None)
        self.memory["model_file_bytes"] = int8_path.stat().st_size if int8_path.is_file() else None
        self.memory["tokenizer_bytes"] = (
            tokenizer_json.stat().st_size if tokenizer_json.is_file() else None
        )
        self.memory["peak_rss_source"] = "sampled during the evaluation loop"

        if not common.psutil:
            self.warnings.note("psutil is not installed; memory metrics are limited to artifact sizes")
        else:
            if self.baseline_rss and self.rss_after_load:
                self.memory["rss_after_model_load_bytes"] = self.rss_after_load
                self.memory["privacy_stack_resident_bytes"] = max(
                    0, self.rss_after_load - self.baseline_rss
                )
                self.memory["privacy_stack_resident_note"] = (
                    "RSS immediately after PIIMaskerAdapter.load() minus the RSS before it. "
                    "This is the whole privacy stack, not the graph alone: the ONNX session, "
                    "the SentencePiece tokenizer, and the transformers import chain that "
                    "AutoTokenizer pulls in. On a 4 GB device that import chain, not the "
                    f"{(self.memory['model_file_bytes'] or 0) / 1024 ** 2:.0f} MiB INT8 file, "
                    "is the larger cost."
                )
                print(f"baseline RSS      : {self.baseline_rss / 1024 ** 2:>8.1f} MiB")
                print(f"privacy stack     : "
                      f"{self.memory['privacy_stack_resident_bytes'] / 1024 ** 2:>8.1f} MiB resident")
            print(f"process RSS       : {self.memory['process_rss_bytes'] / 1024 ** 2:>8.1f} MiB")
            print(f"process peak RSS  : {self.memory['process_peak_rss_bytes'] / 1024 ** 2:>8.1f} MiB")
            print(f"system RAM        : {self.memory['system_total_bytes'] / 1024 ** 3:>8.2f} GiB "
                  f"({self.memory['system_available_bytes'] / 1024 ** 3:.2f} GiB available)")
        if self.memory["model_file_bytes"]:
            print(f"INT8 model file   : {self.memory['model_file_bytes'] / 1024 ** 2:>8.2f} MiB")

    # -- 10. save -----------------------------------------------------------

    def save(self) -> dict[str, Any]:
        print("\n--- Save results ---")
        finished_utc = common.utc_now()
        config_payload = {
            "run_id": self.run_id,
            "benchmark": "cpu_text",
            # Recorded so the figures notebook can separate this run from the
            # frozen Canary runs, which predate these keys and default to canary.
            "pipeline_version": "V4",
            "pipeline_root": common.relative(common.PRODUCTION_ROOT),
            "stt_engine": "whisper",
            "stt_model": self.config.whisper["model"],
            "stt_runtime": self.config.whisper["runtime"],
            "stt_device": self.config.whisper["device"],
            "stt_compute_type": self.config.whisper["compute_type"],
            "stt_beam_size": self.config.whisper["beam_size"],
            "mode": self.options.mode,
            "smoke_test": self.options.smoke_test,
            "device_label": self.options.device_label or None,
            "target_rows": self.options.target_rows,
            "rows_scored": len(self.eval_rows),
            "benchmark_rows_full": self.options.benchmark_rows,
            "smoke_rows": self.options.smoke_rows,
            "warmup_runs": self.options.warmup_runs,
            "latency_probe_rows": self.options.latency_probe_rows,
            "pipeline_warmup_runs": self.options.pipeline_warmup_runs,
            "pipeline_latency_runs": self.options.pipeline_latency_runs,
            "run_full_pipeline": self.options.run_full_pipeline,
            "run_ood_dev": self.options.run_ood_dev,
            "random_seed": self.options.random_seed,
            "onnx_intra_op_threads": self.options.onnx_intra_op_threads,
            "entry_point": "Full_Pipeline_CPU_Benchmark.py",
            "subset": {
                "path": common.relative(self.bench.SUBSET_PATH),
                "sha256": self.subset_manifest["fixture_sha256"],
                "selection": self.subset_manifest["selection"],
                "source": self.subset_manifest["source"],
                "coverage": self.subset_manifest["coverage"],
            },
            "started_utc": self.started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "duration_seconds": (finished_utc - self.started_utc).total_seconds(),
        }

        self.run.json("system_info.json", self.system_info)
        self.run.json("benchmark_config.json", config_payload)
        self.run.json("artifact_manifest.json", common.artifact_manifest_for_disk(self.artifact))
        self.run.json("memory_metrics.json", self.memory)
        self.run.json("pipeline_latency_summary.json", self.pipeline_summary)
        if self.deberta_latency:
            self.run.json("deberta_latency_summary.json", self.deberta_latency)
        if self.evaluation:
            payload = dict(self.evaluation)
            if self.ood_result:
                payload["ood_dev"] = {
                    key: value for key, value in self.ood_result.items() if key != "parity"
                }
            self.run.json("evaluation_metrics.json", payload)

        self.run.csv(
            "pipeline_latency.csv", self.pipeline_rows,
            ["utterance_index", "characters", "words", "transcript_merge_ms", "wake_word_ms",
             "inactivity_buffer_ms", "privacy_masking_ms", "digit_normalisation_ms",
             "text_path_total_ms"],
        )
        self.run.csv(
            "deberta_latency.csv",
            [{"row_id": record["row_id"], "characters": record["characters"],
              "windows": record["windows"], "latency_ms": record["latency_ms"]}
             for record in self.parity_records],
            ["row_id", "characters", "windows", "latency_ms"],
        )
        self.run.csv(
            "per_entity_metrics.csv", self.per_entity,
            ["view", "entity", "precision", "recall", "f1", "support", "predicted",
             "true_positives"],
        )
        self.run.jsonl("prediction_parity_data.jsonl", self.parity_records)
        if self.ood_result:
            self.run.jsonl("ood_dev_parity_data.jsonl", self.ood_result["parity"])

        # One digest over every prediction, in fixture order: two devices that
        # agree have the same string here, and no further comparison is needed.
        self.predictions_digest = common.sha256_text(
            "\n".join(
                f"{record['row_id']}:{record['production_digest']}"
                for record in self.parity_records
            )
        )
        self.run.report()
        print(f"\npredictions digest: {self.predictions_digest}")
        self._write_summary(config_payload)
        return config_payload

    def _write_summary(self, config_payload: Mapping[str, Any]) -> None:
        info = self.system_info
        memory = self.memory
        benchmark_ok = bool(self.evaluation) and self.artifact["ok"] and not self.failures
        model_file = self.artifact["files"].get("model_int8.onnx", {})

        lines = [
            f"# CPU benchmark - {self.run_id}",
            "",
            f"- **Completed successfully:** {'yes' if benchmark_ok else 'no - see warnings'}",
            f"- **Mode:** {self.options.mode} ({len(self.eval_rows):,} of "
            f"{len(self.benchmark_rows_all):,} benchmark rows)",
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
            f"| Python | {info['python']} |",
            f"| ONNX Runtime | {info['onnxruntime']} |",
            "",
            "## Model artifact",
            "",
            f"- Path: `{model_file.get('path', 'missing')}`",
            f"- SHA-256: `{self.artifact['measured_sha256']}`",
            f"- Size: {common.mib(memory.get('model_file_bytes'))}",
            "- Identity verified against the release records: "
            f"{'yes' if self.artifact['identity_verified'] else 'NO'}",
            f"- Benchmark subset SHA-256: `{self.subset_manifest['fixture_sha256']}`",
            f"- Predictions digest: `{self.predictions_digest}`",
            "",
        ]

        if self.evaluation:
            production = self.evaluation["production_path"]
            lines += [
                "## Privacy model quality (production path)",
                "",
                "| Metric | Value |",
                "|---|---|",
                f"| Typed entity F1 | {production['typed_entity']['f1']:.4f} |",
                f"| Typed entity precision | {production['typed_entity']['precision']:.4f} |",
                f"| Typed entity recall | {production['typed_entity']['recall']:.4f} |",
                f"| Binary entity F1 | {production['binary_entity']['f1']:.4f} |",
                f"| Private character recall | {production['private_character_recall']:.4f} |",
                f"| Private character leakage | {production['private_character_leakage']:.4f} |",
                f"| Non-private overmasking | {production['non_private_overmasking_rate']:.4f} |",
                f"| Hard-negative FP rate | {production['hard_negative_false_positive_rate']:.4f} "
                f"({production['hard_negative_rows']} rows) |",
                "| Model-attributed typed F1 | "
                f"{self.evaluation['model_attributed']['typed_entity']['f1']:.4f} |",
                "",
            ]

        if self.ood_result:
            released = self.ood_result.get("release_reference")
            lines += [
                "## Out-of-distribution cross-check (ood_dev, model-only)",
                "",
                f"- Cases scored: {self.ood_result['rows']}"
                f"{'' if self.ood_result['complete_fixture'] else ' (subset - smoke mode)'}",
                f"- Typed entity F1: {self.ood_result['metrics']['typed_entity']['f1']:.4f}",
            ]
            if released:
                lines.append(
                    f"- Release reference: {released['typed_entity_f1']:.4f} "
                    f"(delta {self.ood_result['typed_f1_delta_vs_release']:+.4f})"
                )
            lines.append("")

        lines += ["## Latency", "", "| Stage | Statistics |", "|---|---|"]
        if self.deberta_latency:
            lines += [
                "| DeBERTa full masking call | "
                f"{latency_line(self.deberta_latency['end_to_end_masking_call'])} |",
                f"| - neural decode | {latency_line(self.deberta_latency['stages']['neural_decode'])} |",
                f"| - structured regex | {latency_line(self.deberta_latency['stages']['structured'])} |",
                f"| - span merge | {latency_line(self.deberta_latency['stages']['span_merge'])} |",
            ]
        for name, entry in self.pipeline_stages.items():
            lines.append(
                f"| pipeline: {name} | "
                + (latency_line(entry) if entry.get("available")
                   else "unavailable - " + entry.get("reason", ""))
                + " |"
            )
        if self.pipeline_summary.get("text_path_end_to_end", {}).get("count"):
            lines.append("| pipeline: text path end to end | "
                         f"{latency_line(self.pipeline_summary['text_path_end_to_end'])} |")
        if "full_utterance_estimate_ms" in self.pipeline_summary:
            lines.append(
                f"| pipeline: assembled {self.audio_seconds:.1f}s utterance | "
                f"{self.pipeline_summary['full_utterance_estimate_ms']:.1f} ms "
                f"({'complete' if self.pipeline_summary['complete'] else 'incomplete'}) |"
            )
        if self.deberta_latency.get("throughput_rows_per_second"):
            lines.append("| throughput | "
                         f"{self.deberta_latency['throughput_rows_per_second']:.2f} rows/s |")

        lines += [
            "",
            "## Memory",
            "",
            f"- Process RSS: {common.mib(memory.get('process_rss_bytes'))}",
            f"- Process peak RSS: {common.mib(memory.get('process_peak_rss_bytes'))}",
            "- Privacy stack resident (session + tokenizer + imports): "
            f"{common.mib(memory.get('privacy_stack_resident_bytes'))}",
            f"- System RAM: {common.gib(memory.get('system_total_bytes'))}",
            f"- INT8 model on disk: {common.mib(memory.get('model_file_bytes'))}",
            "",
            "## Warnings",
            "",
        ]
        lines += [f"- {message}" for message in self.warnings] or ["- none"]
        lines += ["", "## Sections completed", ""]
        lines += [
            f"- {name}: {'yes' if done else 'no'}"
            for name, done in sorted(self.completed.items())
        ] or ["- none"]

        path = self.run.markdown("benchmark_summary.md", lines)
        print(f"\nwrote {common.relative(path)}\n")
        print("\n".join(lines))

    # -- orchestration ------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        self.collect_system()
        self.verify_artifacts()
        self.load_rows()
        self.load_privacy_model()
        self.probe_stages()
        self.evaluate()
        self.cross_check_ood()
        self.run_pipeline()
        self.collect_memory()
        return self.save()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Full_Pipeline_CPU_Benchmark.py",
        description=(
            "OnKith CPU/text deployment benchmark. Scores the committed 2,000-row "
            "held-out subset through the production DeBERTa-v3-xsmall INT8 path and "
            "records latency, memory, quality and an out-of-distribution parity check. "
            "Run it on the workstation and on the Raspberry Pi; commit both result "
            "folders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python Full_Pipeline_CPU_Benchmark.py               full 2,000-row benchmark\n"
            "  python Full_Pipeline_CPU_Benchmark.py --smoke-test  ~50-row validation\n"
            "  python Full_Pipeline_CPU_Benchmark.py --rows 100    explicit row count\n"
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=f"score {DEFAULT_SMOKE_ROWS} rows and a strided ood_dev subset, for a fast check",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="explicit number of benchmark rows (debugging; overrides the mode default)",
    )
    parser.add_argument(
        "--device-label", default="",
        help="optional tag in the run id, e.g. pi5 or laptop",
    )
    parser.add_argument(
        "--no-pipeline", action="store_true",
        help="skip part A, the assembled VAD/STT/text pipeline timing",
    )
    parser.add_argument(
        "--no-ood-dev", action="store_true",
        help="skip the ood_dev cross-check against the release number",
    )
    parser.add_argument(
        "--intra-op-threads", type=int, default=None,
        help="ONNX Runtime intra-op threads for the model-only decoder (default: production)",
    )
    parser.add_argument(
        "--results-root", type=Path, default=None,
        help="write the run folder somewhere other than research_results/cpu_benchmarks",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> BenchmarkOptions:
    """Resolve CLI plus the two environment overrides the notebook honoured."""

    smoke = bool(args.smoke_test)
    override = os.environ.get("ONKITH_BENCHMARK_SMOKE")
    if override is not None and not smoke:
        smoke = override.strip() not in ("", "0", "false", "False")
    device_label = args.device_label or os.environ.get("ONKITH_BENCHMARK_DEVICE", "")
    return BenchmarkOptions(
        smoke_test=smoke,
        rows=args.rows,
        run_full_pipeline=not args.no_pipeline,
        run_ood_dev=not args.no_ood_dev,
        device_label=device_label,
        onnx_intra_op_threads=args.intra_op_threads,
        results_root=args.results_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print("OnKith Full Pipeline CPU Benchmark")
    print("==================================")
    benchmark = CpuBenchmark(options_from_args(args))
    benchmark.execute()
    ok = bool(benchmark.evaluation) and benchmark.artifact["ok"] and not benchmark.failures
    print(f"\nresult folder: {benchmark.run.relative_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
