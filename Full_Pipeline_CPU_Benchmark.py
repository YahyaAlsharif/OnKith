#!/usr/bin/env python3
"""OnKith CPU/text deployment benchmark - the canonical entry point.

Run it on the development workstation and run the identical file on the
Raspberry Pi. Both runs score the *same committed rows* through the *same
hash-verified INT8 model*, so the two result folders are comparable
measurements of one artifact on two devices.

    git pull
    python Full_Pipeline_CPU_Benchmark.py

No Jupyter, no notebook server, no editor. The Pi needs Python and the
repository, nothing else: the 2,000-row benchmark subset is committed, so no
corpus has to be rebuilt there.

    python Full_Pipeline_CPU_Benchmark.py               full benchmark (default)
    python Full_Pipeline_CPU_Benchmark.py --smoke-test  ~50 rows, a couple of minutes
    python Full_Pipeline_CPU_Benchmark.py --rows 100    explicit row count
    python Full_Pipeline_CPU_Benchmark.py --help        every option

Everything lands in ``research_results/cpu_benchmarks/<run_id>/``,
machine-readable, plus a ``benchmark_summary.md`` you can read directly.
``Research_Paper_Figures.ipynb`` discovers those folders automatically.

The measured pipeline is ``FullPipeline_V6``, selected by
``privacy_eval.benchmark_common.PRODUCTION_ROOT``. The run id carries
``v6_whisper`` and results land in ``research_results/v6_benchmarks/``, away
from the frozen V3, V4 and V5 folders in ``cpu_benchmarks/``. V6 runs the same
recognizer V4 did, so the engine token alone would not tell the two apart.

The ``stt_whisper`` stage probes the recognizer with a synthetic tone -
identical bytes on every device, which is what makes it a cross-device timing
comparison. It is **not** the recognizer's deployment cost: Whisper has no words
to find in a tone, decodes against its own prompt and runs until its 30 second
window is full, so the figure measures a failure mode rather than
transcription. Whisper's cost on real speech is in
``Full_Pipeline_Audio_Benchmark.py``, which drives 44 committed WAV clips.

This file is deliberately thin. The measurement lives in
``model/privacy_evaluation/privacy_eval/cpu_benchmark.py``, next to the row
selection and the metric definitions it reuses, so there is one implementation
rather than two that can drift.
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

from privacy_eval.cpu_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
