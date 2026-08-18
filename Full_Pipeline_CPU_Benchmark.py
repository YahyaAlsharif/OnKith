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

if not (REPO_ROOT / "FullPipeline_V3").is_dir():
    raise SystemExit("Run this script from inside the OnKith repository.")
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval.cpu_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
