#!/usr/bin/env python3
"""OnKith end-to-end audio pipeline benchmark - the canonical entry point.

Where ``Full_Pipeline_CPU_Benchmark.py`` measures the privacy model on 2,000
committed text rows, this one starts from sound:

    audio -> Silero VAD -> Whisper STT -> transcript merge / wake-word
          -> inactivity buffer -> DeBERTa-v3-xsmall INT8 -> masking
          -> spoken-digit rendering -> final sanitized text

It answers the deployment questions the text benchmark cannot: what does the
recognizer actually produce, which stage dominates the runtime, how does cost
scale with clip length, how much memory does the whole stack need, and - the
research question - does private information stay protected after imperfect
speech recognition?

    git pull
    python Full_Pipeline_Audio_Benchmark.py

The clips are committed, so nothing is downloaded and no TTS runs on the device.

    python Full_Pipeline_Audio_Benchmark.py                  every clip
    python Full_Pipeline_Audio_Benchmark.py --smoke-test     four clips
    python Full_Pipeline_Audio_Benchmark.py --clip privacy_001
    python Full_Pipeline_Audio_Benchmark.py --list-clips
    python Full_Pipeline_Audio_Benchmark.py --help           every option

Results land in ``research_results/cpu_benchmarks/<run_id>/`` beside the text
benchmark's, tagged ``"benchmark": "audio_pipeline"`` and
``"stt_engine": "whisper"``.

V6 results are written to ``research_results/v6_benchmarks/``, not to
``cpu_benchmarks/`` where the frozen V3, V4 and V5 folders live. V6 runs the
same recognizer V4 did, so a folder is not identifiable by its engine token
alone; keeping the roots apart means a V6 run can never be written into, or read
as, a frozen historical one. The figures notebook reads both roots and keys on
``pipeline_version``.

If the speech recognizer is not installed the script says exactly what is
missing and how to get it, runs everything that does not depend on it, and exits
with code 2. It never fabricates a transcript.

The measured pipeline is ``FullPipeline_V6``, selected by
``privacy_eval.benchmark_common.PRODUCTION_ROOT``. Install its runtime first::

    python -m pip install --requirement FullPipeline_V6/requirements-rpi.txt

which brings ``faster-whisper`` on CTranslate2 - no torch, and wheels for both
Windows and Raspberry Pi OS (aarch64). The ``base.en`` weights are fetched once
on first use and cached; after that the device runs offline.

This file is deliberately thin. The measurement lives in
``model/privacy_evaluation/privacy_eval/audio_benchmark.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = REPO_ROOT / "model" / "privacy_evaluation"

if not (REPO_ROOT / "FullPipeline_V6").is_dir():
    raise SystemExit("Run this script from inside the OnKith repository.")
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval.audio_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
