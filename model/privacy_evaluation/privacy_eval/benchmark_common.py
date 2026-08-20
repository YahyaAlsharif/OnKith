"""Plumbing shared by the two deployment benchmark entry points.

``Full_Pipeline_CPU_Benchmark.py`` measures the privacy model on 2,000 committed
text rows. ``Full_Pipeline_Audio_Benchmark.py`` measures the assembled
audio -> STT -> privacy path on a small committed clip set. They answer different
questions, but a workstation run and a Raspberry Pi run of *either* are only
comparable if both recorded the device the same way, hashed the same artifact,
summarised latency by the same definition and wrote the same file names. That
agreement lives here rather than being restated in each script.

Nothing in this module measures anything by itself. It identifies the machine,
proves the artifact, and formats results.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOT = REPO_ROOT / "FullPipeline_V4"
#: The superseded Canary pipeline, kept at the repository root beside V4 so the
#: progression is readable. Nothing in these benchmarks imports it: the Canary
#: numbers in research_results/ are frozen and are read from disk, never
#: regenerated. It is named here so the path is discoverable rather than lore.
CANARY_V3_ROOT = REPO_ROOT / "FullPipeline_V3"
EVALUATION_ROOT = REPO_ROOT / "model" / "privacy_evaluation"
RESULTS_ROOT = REPO_ROOT / "research_results" / "cpu_benchmarks"

SAMPLE_RATE = 16_000

try:  # optional: only the memory sections need it
    import psutil
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore[assignment]


def ensure_import_paths() -> None:
    """Put the pipeline and the evaluation harness on ``sys.path``.

    Neither is a pip package; both are imported by bare name, exactly as the
    notebooks did. Importing them is the whole point - a benchmark that
    reimplemented the deployment path would be measuring something else.
    """

    for extra in (PRODUCTION_ROOT, EVALUATION_ROOT):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))


# --------------------------------------------------------------------------
# Hashing and small formatting helpers
# --------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gib(value: float | int | None) -> str:
    return f"{value / 1024 ** 3:.2f} GiB" if value else "unknown"


def mib(value: float | int | None) -> str:
    return f"{value / 1024 ** 2:.1f} MiB" if value else "unknown"


def slug(value: str) -> str:
    """Filesystem- and URL-safe token; run ids become directory names."""

    kept = [character if character.isalnum() else "-" for character in value.strip().lower()]
    return "".join(kept).strip("-")[:24] or "unknown"


def relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


# --------------------------------------------------------------------------
# Latency statistics
# --------------------------------------------------------------------------


def summarise(samples: Iterable[float], unit_scale: float = 1000.0) -> dict[str, Any]:
    """count/mean/std/min/max/median/P90/P95/P99, in milliseconds by default.

    Linear interpolation between order statistics, so a 20-sample audio run and
    a 2,000-sample text run use one definition of P95.
    """

    values = sorted(float(sample) * unit_scale for sample in samples)
    if not values:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": values[0],
        "max_ms": values[-1],
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "total_ms": sum(values),
    }


def print_summary(title: str, summary: Mapping[str, Any]) -> None:
    if not summary.get("count"):
        print(f"{title}: no samples")
        return
    print(
        f"{title:<28} n={summary['count']:<6} mean {summary['mean_ms']:>9.2f}  "
        f"p50 {summary['p50_ms']:>9.2f}  p90 {summary['p90_ms']:>9.2f}  "
        f"p95 {summary['p95_ms']:>9.2f}  p99 {summary['p99_ms']:>9.2f}  "
        f"max {summary['max_ms']:>9.2f}  (ms)"
    )


def latency_line(summary: Mapping[str, Any]) -> str:
    """One markdown table cell. No pipe characters - they would split the row."""

    if not summary.get("count"):
        return "not measured"
    return (
        f"mean {summary['mean_ms']:.1f} ms, P50 {summary['p50_ms']:.1f}, "
        f"P95 {summary['p95_ms']:.1f}, P99 {summary['p99_ms']:.1f} (n={summary['count']})"
    )


# --------------------------------------------------------------------------
# Device identification
# --------------------------------------------------------------------------


def _read_first(*paths: str) -> str:
    for candidate in paths:
        try:
            text = Path(candidate).read_text(errors="ignore").strip("\x00").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def cpu_model_name() -> str:
    """Best available CPU name. platform.processor() is empty on most Linux ARM."""

    if platform.system() == "Linux":
        cpuinfo = _read_first("/proc/cpuinfo")
        fields: dict[str, str] = {}
        for line in cpuinfo.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields.setdefault(key.strip().lower(), value.strip())
        for key in ("model name", "hardware", "cpu implementer", "processor"):
            if fields.get(key):
                return fields[key]
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def board_model() -> str:
    """Raspberry Pi and other device-tree boards announce themselves here."""

    return _read_first(
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
        "/sys/devices/virtual/dmi/id/product_name",
    )


def installed(distribution: str) -> str | None:
    """Version without importing the package.

    ``import torch`` costs several hundred MiB of RSS, which would land in the
    benchmark's own memory numbers and make them useless on a 4 GB Pi.
    """

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def rss_bytes() -> int | None:
    return psutil.Process().memory_info().rss if psutil else None


def collect_system_info(run_id: str, collected_utc: datetime) -> dict[str, Any]:
    """Everything about this machine that could explain a difference in numbers."""

    import numpy as np
    import onnxruntime as ort

    virtual_memory = psutil.virtual_memory() if psutil else None
    board = board_model()
    return {
        "run_id": run_id,
        "collected_utc": collected_utc.isoformat(),
        "os": platform.system(),
        "platform": platform.platform(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "process_architecture": platform.architecture()[0],
        "byte_order": sys.byteorder,
        "cpu_model": cpu_model_name(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False) if psutil else None,
        "board_model": board or None,
        "is_raspberry_pi": "raspberry pi" in board.lower(),
        "hostname": platform.node(),
        "total_ram_bytes": virtual_memory.total if virtual_memory else None,
        "available_ram_bytes": virtual_memory.available if virtual_memory else None,
        "baseline_process_rss_bytes": rss_bytes(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": list(ort.get_available_providers()),
        "transformers": installed("transformers"),
        "tokenizers": installed("tokenizers"),
        "torch": installed("torch"),
        "nemo_toolkit": installed("nemo-toolkit"),
        "psutil": psutil.__version__ if psutil else None,
    }


def print_system_info(info: Mapping[str, Any]) -> None:
    print("Hardware / Runtime")
    print("------------------")
    print(f"Platform     : {info['platform']}")
    print(f"Architecture : {info['architecture']} ({info['process_architecture']} process)")
    print(f"CPU          : {info['cpu_model']}")
    print(f"Logical CPUs : {info['logical_cpus']}  (physical {info['physical_cpus']})")
    print(f"Board        : {info['board_model'] or 'not reported'}")
    print(f"RAM          : {gib(info['total_ram_bytes'])} total, "
          f"{gib(info['available_ram_bytes'])} available")
    print(f"Python       : {info['python']} ({info['python_implementation']})")
    print(f"ONNX Runtime : {info['onnxruntime']}  providers={info['onnxruntime_providers']}")
    print(f"transformers : {info['transformers']}  numpy {info['numpy']}  torch {info['torch']}")
    print(f"baseline RSS : {gib(info['baseline_process_rss_bytes'])} before any model is loaded")
    if info["is_raspberry_pi"]:
        print("\nRaspberry Pi detected. Same model, same code, same inputs as the workstation run.")


# --------------------------------------------------------------------------
# Run identity and output
# --------------------------------------------------------------------------


def make_run_id(started_utc: datetime, mode: str, device_label: str = "", *,
                prefix: str = "") -> str:
    """``<UTC>_<os>-<arch>_<mode>[_<label>]``, optionally namespaced by prefix."""

    parts = (
        started_utc.strftime("%Y%m%dT%H%M%SZ"),
        f"{slug(platform.system())}-{slug(platform.machine())}",
        slug(prefix) if prefix else "",
        mode,
        slug(device_label) if device_label else "",
    )
    return "_".join(part for part in parts if part)


class RunDirectory:
    """One benchmark run's output folder, plus the writers that fill it.

    Every file is written with explicit LF endings so a Windows workstation run
    and a Pi run are byte-comparable, which is the same reason the JSONL
    fixtures are marked ``-text`` in ``.gitattributes``.
    """

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.path = (root or RESULTS_ROOT) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.written: list[Path] = []

    @property
    def relative_path(self) -> str:
        return relative(self.path)

    def json(self, name: str, payload: Any) -> Path:
        path = self.path / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, default=float) + "\n")
        self.written.append(path)
        return path

    def csv(self, name: str, rows: Sequence[Mapping[str, Any]],
            fieldnames: Sequence[str]) -> Path | None:
        if not rows:
            return None
        path = self.path / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.written.append(path)
        return path

    def jsonl(self, name: str, rows: Sequence[Any]) -> Path | None:
        if not rows:
            return None
        path = self.path / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, default=float) + "\n")
        self.written.append(path)
        return path

    def markdown(self, name: str, lines: Sequence[str]) -> Path:
        path = self.path / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        self.written.append(path)
        return path

    def report(self) -> None:
        print(f"wrote {len(self.written)} files to {self.relative_path}")
        for path in self.written:
            print(f"  {path.name:<34} {path.stat().st_size:>9,} bytes")


class WarningLog:
    """Warnings are collected, printed once, and repeated in the summary file."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def note(self, message: str) -> None:
        self.messages.append(message)
        print("WARNING:", message)

    def __bool__(self) -> bool:
        return bool(self.messages)

    def __iter__(self):
        return iter(self.messages)


# --------------------------------------------------------------------------
# Privacy artifact identity
# --------------------------------------------------------------------------

ACCEPTED_CANDIDATE_PATH = (
    REPO_ROOT / "model" / "DeBERTa-v3-xsmall" / "artifacts" / "model_v2"
    / "metrics" / "accepted_int8_candidate.json"
)


def verify_privacy_artifacts(config: Any, run_id: str, warnings: WarningLog) -> dict[str, Any]:
    """Re-hash the packaged INT8 graph and check it against every written record.

    The point of running a benchmark twice is to compare two devices on the
    *same* model, so the hash is checked against ``live_config.json`` (which the
    live pipeline refuses to start without), the deployment package manifest,
    and - when the model workspace is present - the release gate. A mismatch is
    reported rather than silently producing numbers for an unknown artifact.
    """

    int8_path = config.pii_model_path
    tokenizer_dir = config.pii_tokenizer_path
    label_map_path = config.pii_label_map_path
    deploy_manifest_path = int8_path.parent / "model_manifest.json"

    complete = True
    artifact_files: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("model_int8.onnx", int8_path),
        ("tokenizer/tokenizer.json", tokenizer_dir / "tokenizer.json"),
        ("tokenizer/tokenizer_config.json", tokenizer_dir / "tokenizer_config.json"),
        ("config/model_v2_label_map.json", label_map_path),
    ):
        if not path.is_file():
            warnings.note(f"missing deployment artifact: {path}")
            complete = False
            continue
        artifact_files[label] = {
            "path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    measured_sha = artifact_files.get("model_int8.onnx", {}).get("sha256")
    deploy_manifest = json.loads(deploy_manifest_path.read_text(encoding="utf-8"))

    checks = {
        "live_config.pii.sha256": str(config.pii["sha256"]) == measured_sha,
        "model_manifest.files.model_int8.onnx.sha256":
            deploy_manifest["files"]["model_int8.onnx"]["sha256"] == measured_sha,
        "label_map.ontology_version":
            label_map["ontology_version"] == config.pii["ontology_version"],
        "label_map.num_labels": int(label_map["num_labels"]) == int(config.pii["label_count"]),
    }

    accepted_candidate = None
    if ACCEPTED_CANDIDATE_PATH.is_file():
        accepted_candidate = json.loads(ACCEPTED_CANDIDATE_PATH.read_text(encoding="utf-8"))
        checks["accepted_int8_candidate.model.int8_sha256"] = (
            accepted_candidate["model"]["int8_sha256"] == measured_sha
        )
        checks["accepted_int8_candidate.accepted"] = accepted_candidate.get("accepted") is True
    else:
        warnings.note(
            "accepted_int8_candidate.json not found; hash checked against the "
            "deployment records only"
        )

    frozen_ood_path = ACCEPTED_CANDIDATE_PATH.parent / "evaluation_ood_final_int8.json"

    manifest = {
        "run_id": run_id,
        "model": {
            "name": deploy_manifest.get("model"),
            "model_id": deploy_manifest.get("model_id"),
            "ontology_version": label_map["ontology_version"],
            "entity_types": int(label_map["num_entities"]),
            "label_count": int(label_map["num_labels"]),
            "precision": "int8",
            "provider": str(config.pii["provider"]),
            "context_limit": int(config.pii["context_limit"]),
            "overflow_stride": int(config.pii["overflow_stride"]),
            "decoder": "integrated_pipeline.model_v2_masker.ModelV2OnnxMasker",
        },
        "files": artifact_files,
        "identity_checks": checks,
        "identity_verified": all(checks.values()),
        "accepted_candidate_manifest": (
            relative(ACCEPTED_CANDIDATE_PATH) if accepted_candidate else None
        ),
        "accepted_candidate_manifest_sha256": (
            sha256_file(ACCEPTED_CANDIDATE_PATH) if accepted_candidate else None
        ),
        "frozen_ood_final_spent": frozen_ood_path.is_file(),
    }
    manifest["artifacts_complete"] = complete
    manifest["ok"] = complete and manifest["identity_verified"]
    manifest["measured_sha256"] = measured_sha
    manifest["label_map"] = {
        "id2label": {int(key): str(value) for key, value in label_map["id2label"].items()},
        "num_entities": int(label_map["num_entities"]),
        "num_labels": int(label_map["num_labels"]),
        "ontology_version": label_map["ontology_version"],
    }
    manifest["release_metrics"] = (
        (accepted_candidate or {}).get("evaluation", {}).get("int8_onnx", {})
    )
    return manifest


def print_artifact_identity(config: Any, manifest: Mapping[str, Any]) -> None:
    int8_path = config.pii_model_path
    label_map = manifest["label_map"]
    print(f"INT8 model   : {relative(int8_path)}")
    if int8_path.is_file():
        size = int8_path.stat().st_size
        print(f"  size       : {size:,} bytes ({size / 1024 ** 2:.2f} MiB)")
    print(f"  sha256     : {manifest['measured_sha256']}")
    print(f"tokenizer    : {relative(config.pii_tokenizer_path)}")
    print(f"label map    : {label_map['num_entities']} entities / "
          f"{label_map['num_labels']} BIO labels  ({label_map['ontology_version']})")
    print("\nidentity checks")
    for name, passed in manifest["identity_checks"].items():
        print(f"  {'OK  ' if passed else 'FAIL'}  {name}")


#: Keys stripped before ``artifact_manifest.json`` is written. They are runtime
#: conveniences for the caller, not part of the recorded artifact identity.
_MANIFEST_RUNTIME_KEYS = ("label_map", "release_metrics", "ok", "artifacts_complete",
                          "measured_sha256")


def artifact_manifest_for_disk(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in _MANIFEST_RUNTIME_KEYS}


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def collect_memory(run_id: str, *, baseline_rss: int | None, peak_rss: int,
                   stages: Mapping[str, int | None] | None = None) -> dict[str, Any]:
    """Process memory, including the platform-native peak where the OS keeps one."""

    memory: dict[str, Any] = {
        "run_id": run_id,
        "psutil_available": bool(psutil),
        "baseline_rss_bytes": baseline_rss,
        "stage_rss_bytes": {name: value for name, value in (stages or {}).items()},
    }
    if not psutil:
        return memory

    process = psutil.Process()
    memory_info = process.memory_info()
    virtual = psutil.virtual_memory()
    memory.update({
        "process_rss_bytes": memory_info.rss,
        "process_vms_bytes": memory_info.vms,
        "process_peak_rss_bytes": max(peak_rss, memory_info.rss),
        "peak_rss_source": "sampled during the benchmark loop",
        "system_total_bytes": virtual.total,
        "system_available_bytes": virtual.available,
        "system_used_percent": virtual.percent,
    })

    native_peak = getattr(memory_info, "peak_wset", None)
    if native_peak is None and platform.system() != "Windows":
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS reports bytes.
            native_peak = usage * 1024 if platform.system() == "Linux" else usage
        except Exception:  # pragma: no cover - platform dependent
            native_peak = None
    if native_peak:
        memory["process_peak_rss_bytes_os"] = int(native_peak)
        memory["process_peak_rss_bytes"] = max(
            memory["process_peak_rss_bytes"], int(native_peak)
        )
    return memory


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

# --------------------------------------------------------------------------
# Speech recognizer plumbing shared by both benchmarks
# --------------------------------------------------------------------------


def quieten_stt() -> None:
    """Keep the recognizer's own logging off the benchmark's console.

    Both benchmarks are meant to be watched over SSH, and the decoding settings
    that matter are verified by ``WhisperAdapter`` and recorded in the result
    files either way.
    """

    import logging

    for name in ("faster_whisper", "ctranslate2"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:  # pragma: no cover - only affects console noise
            pass


def make_canary_segment(samples, *, session_id: str, utterance_id: str,
                        segment_id: str, sequence_number: int = 1):
    """One whole clip as a single final ``CanarySegment``.

    Live capture splits long speech into overlapping segments; a benchmark
    reading a fixed file has no reason to, so each clip is submitted as one
    complete, reliable, final segment. The schema validates every field, so the
    values here are the real ones rather than placeholders:
    ``utterance_sequence_number`` starts at 1, the sample interval matches the
    audio length, and ``endpoint_reason`` is one of the accepted reasons.
    """

    import time
    from integrated_pipeline.schemas import CanarySegment

    now = time.monotonic()
    stamp = utc_now().isoformat().replace("+00:00", "Z")
    return CanarySegment(
        session_id=session_id,
        utterance_id=utterance_id,
        segment_id=segment_id,
        utterance_sequence_number=max(1, int(sequence_number)),
        segment_index=0,
        samples=samples,
        sample_rate=SAMPLE_RATE,
        segment_start_sample=0,
        segment_end_sample=len(samples),
        decision_sample=len(samples),
        last_speech_end_sample=len(samples),
        forced_boundary_sample=None,
        endpoint_reason="end_of_stream",
        is_final_segment_of_utterance=True,
        reliable=True,
        dropped_audio_chunks=0,
        recording_closed_monotonic=now,
        audio_started_monotonic=now,
        audio_ended_monotonic=now,
        audio_started_at_utc=stamp,
        audio_ended_at_utc=stamp,
        utterance_start_sample=0,
        utterance_captured_end_sample=len(samples),
    )
