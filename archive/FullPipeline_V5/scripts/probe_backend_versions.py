#!/usr/bin/env python3
"""Decide, with evidence, whether a candidate backend version works on this host.

The validated Kaggle notebook used NeMo 2.7.3 and Silero 6.2.1. This project
pins NeMo 2.5.3 and the Silero 6.0.0 ONNX artifact. **Do not change either pin
to match the notebook until this script reports PASS on a real Raspberry Pi.**
Matching a notebook is not a reason to change a deployed backend; proving the
candidate installs, loads and decodes on the target is.

Three levels of evidence, never conflated:

``metadata``   PyPI wheel/dependency facts. Shows whether a version *can*
               install on this platform. Reported as UNVERIFIED for runtime
               behaviour - it proves nothing about decoding.
``install``    Actually resolves and installs into a throwaway virtualenv
               (``--install``). Proves installability on this exact machine.
``runtime``    Imports the installed package and runs a real operation
               (``--install --run``). Proves it runs here.

Usage:
    python scripts/probe_backend_versions.py                  # metadata only
    python scripts/probe_backend_versions.py --install --run  # on the Pi
    python scripts/probe_backend_versions.py --json report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"
UNVERIFIED = "UNVERIFIED"

# Currently pinned by this project.
PINNED_NEMO = "2.5.3"
PINNED_SILERO_SHA256 = (
    "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004"
)
PINNED_SILERO_PACKAGE = "6.0.0"
# Used by the validated notebook.
NOTEBOOK_NEMO = "2.7.3"
NOTEBOOK_SILERO = "6.2.1"


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


def pypi(pkg: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def is_target_platform() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def wheel_kinds(data: dict) -> list[str]:
    return [item["filename"] for item in data["urls"] if item["filename"].endswith(".whl")]


def check_metadata(report: Report) -> dict[str, object]:
    """PyPI facts only. Installability signal, never a runtime claim."""

    facts: dict[str, object] = {}
    try:
        for version in (PINNED_NEMO, NOTEBOOK_NEMO):
            data = pypi("nemo_toolkit", version)
            wheels = wheel_kinds(data)
            requires = data["info"].get("requires_dist") or []
            torch_pin = [
                item for item in requires if item.split(";")[0].strip().startswith("torch")
                and "torchmetrics" not in item and "torchdiffeq" not in item
                and "torchsde" not in item
            ]
            asr = [item for item in requires if item.strip().endswith('extra == "asr"')]
            source_only = [
                item.split(";")[0].strip()
                for item in asr
                if "ctc_segmentation" in item
            ]
            facts[f"nemo_{version}"] = {
                "wheels": wheels,
                "pure_python": all("py3-none-any" in name for name in wheels),
                "torch_requirement": torch_pin,
                "asr_extra_entries": len(asr),
                "ctc_segmentation_required": source_only,
            }
            report.add(
                f"nemo_toolkit {version} ships a platform-independent wheel",
                PASS if all("py3-none-any" in name for name in wheels) else FAIL,
                ", ".join(wheels),
            )
            if source_only:
                report.add(
                    f"nemo_toolkit[asr] {version} pulls a source-only C extension",
                    UNVERIFIED,
                    f"{source_only} has no prebuilt wheels; it must compile from "
                    "source on ARM64, which needs a toolchain and build time",
                )
            else:
                report.add(
                    f"nemo_toolkit[asr] {version} pulls a source-only C extension",
                    PASS,
                    "no ctc_segmentation dependency",
                )
    except Exception as exc:
        report.add("PyPI metadata for nemo_toolkit", NOT_RUN, f"{type(exc).__name__}: {exc}")

    try:
        data = pypi("torch", "2.7.1")
        arm = sorted(
            name for name in wheel_kinds(data) if "aarch64" in name
        )
        facts["torch_2.7.1_aarch64_wheels"] = arm
        report.add(
            "torch 2.7.1 publishes Linux aarch64 wheels",
            PASS if arm else FAIL,
            ", ".join(arm) or "none",
        )
    except Exception as exc:
        report.add("PyPI metadata for torch", NOT_RUN, f"{type(exc).__name__}: {exc}")

    # The Silero ONNX artifact is pinned by SHA-256, so a version bump is a
    # model change unless the bytes are identical.
    try:
        digests = {}
        for version in (PINNED_SILERO_PACKAGE, NOTEBOOK_SILERO):
            data = pypi("silero-vad", version)
            url = next(
                item["url"] for item in data["urls"] if item["filename"].endswith(".whl")
            )
            with urllib.request.urlopen(url, timeout=180) as response:
                blob = response.read()
            with zipfile.ZipFile(__import__("io").BytesIO(blob)) as archive:
                payload = archive.read("silero_vad/data/silero_vad.onnx")
            digests[version] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        facts["silero_onnx"] = digests
        same = (
            digests[PINNED_SILERO_PACKAGE]["sha256"]
            == digests[NOTEBOOK_SILERO]["sha256"]
        )
        report.add(
            f"silero-vad {NOTEBOOK_SILERO} ships the same ONNX as {PINNED_SILERO_PACKAGE}",
            PASS if same else FAIL,
            f"{PINNED_SILERO_PACKAGE}={digests[PINNED_SILERO_PACKAGE]['sha256']} "
            f"({digests[PINNED_SILERO_PACKAGE]['size_bytes']} bytes) vs "
            f"{NOTEBOOK_SILERO}={digests[NOTEBOOK_SILERO]['sha256']} "
            f"({digests[NOTEBOOK_SILERO]['size_bytes']} bytes)"
            + (
                ""
                if same
                else " -- identical size, different content: a silent VAD model "
                "change that the size check alone would not catch"
            ),
        )
        report.add(
            "Pinned artifact still matches silero-vad " + PINNED_SILERO_PACKAGE,
            PASS
            if digests[PINNED_SILERO_PACKAGE]["sha256"] == PINNED_SILERO_SHA256
            else FAIL,
            digests[PINNED_SILERO_PACKAGE]["sha256"],
        )
    except Exception as exc:
        report.add("Silero ONNX artifact comparison", NOT_RUN, f"{type(exc).__name__}: {exc}")
    return facts


def check_install(report: Report, run_runtime: bool) -> None:
    """Actually install the candidate versions into a throwaway virtualenv."""

    if not is_target_platform():
        report.add(
            f"nemo_toolkit[asr]=={NOTEBOOK_NEMO} installs on target",
            NOT_RUN,
            f"host is {platform.machine()}, not aarch64; an install here would "
            "prove nothing about the Raspberry Pi",
        )
        report.add(
            f"silero-vad=={NOTEBOOK_SILERO} installs on target", NOT_RUN, "not aarch64"
        )
        report.add("Candidate backend runs on target", NOT_RUN, "not aarch64")
        return

    with tempfile.TemporaryDirectory() as directory:
        venv = Path(directory) / "probe-venv"
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv)],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except Exception as exc:
            report.add("Create probe virtualenv", FAIL, f"{type(exc).__name__}: {exc}")
            return
        python = venv / "bin" / "python"
        for spec, label in (
            (f"nemo_toolkit[asr]=={NOTEBOOK_NEMO}", "nemo_toolkit[asr]"),
            (f"silero-vad=={NOTEBOOK_SILERO}", "silero-vad"),
        ):
            try:
                completed = subprocess.run(
                    [str(python), "-m", "pip", "install", spec],
                    capture_output=True,
                    text=True,
                    timeout=5400,
                )
            except Exception as exc:
                report.add(f"{label} installs on target", FAIL, f"{type(exc).__name__}: {exc}")
                continue
            ok = completed.returncode == 0
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            report.add(
                f"{label} {spec.split('==')[-1]} installs on target",
                PASS if ok else FAIL,
                " | ".join(tail[-3:]) if tail else f"exit {completed.returncode}",
            )

        if not run_runtime:
            report.add("Candidate backend runs on target", NOT_RUN, "--run not supplied")
            return
        probe = (
            "import torch, numpy;"
            "from nemo.collections.asr.models import EncDecMultiTaskModel;"
            "from nemo.collections.asr.models.aed_multitask_models import "
            "MultiTaskTranscriptionConfig;"
            "print('nemo import OK', torch.__version__)"
        )
        try:
            completed = subprocess.run(
                [str(python), "-c", probe],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            report.add(
                "Candidate backend runs on target",
                PASS if completed.returncode == 0 else FAIL,
                (completed.stdout or completed.stderr or "").strip()[-300:],
            )
        except Exception as exc:
            report.add("Candidate backend runs on target", FAIL, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--skip-metadata", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    print(
        f"[PROBE] {platform.platform()} | Python {platform.python_version()} "
        f"| {platform.machine()} | {sysconfig.get_platform()} | {utc_now_iso()}"
    )
    print(
        f"[PROBE] pinned: nemo {PINNED_NEMO}, silero artifact {PINNED_SILERO_PACKAGE} "
        f"| notebook: nemo {NOTEBOOK_NEMO}, silero {NOTEBOOK_SILERO}"
    )
    facts = {} if args.skip_metadata else check_metadata(report)
    if args.install:
        check_install(report, args.run)
    else:
        report.add(
            f"nemo_toolkit[asr]=={NOTEBOOK_NEMO} installs on target",
            NOT_RUN,
            "--install not supplied",
        )
        report.add("Candidate backend runs on target", NOT_RUN, "--install not supplied")

    counts = report.summary()
    print(
        f"\nSummary: {counts[PASS]} PASS, {counts[FAIL]} FAIL, "
        f"{counts[NOT_RUN]} NOT RUN, {counts[UNVERIFIED]} UNVERIFIED"
    )
    print(
        "\nDecision rule: keep the current pins unless every 'installs on "
        "target' and 'runs on target' check PASSes on a physical Raspberry Pi "
        "AND a real-model transcript parity run shows an acceptable difference. "
        "PyPI metadata alone is never sufficient."
    )
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "generated_at_utc": utc_now_iso(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "is_target_platform": is_target_platform(),
                    "pinned": {"nemo": PINNED_NEMO, "silero": PINNED_SILERO_PACKAGE},
                    "notebook": {"nemo": NOTEBOOK_NEMO, "silero": NOTEBOOK_SILERO},
                    "metadata_facts": facts,
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
