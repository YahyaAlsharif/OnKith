#!/usr/bin/env python3
"""OnKith V6 integrated round-trip benchmark - the canonical entry point.

The other two benchmarks measure the half of V6 that never leaves the device:

    Full_Pipeline_CPU_Benchmark.py    privacy model on 2,000 committed rows
    Full_Pipeline_Audio_Benchmark.py  audio -> Whisper -> masking, 44 clips

This one measures the half that is the point of the project:

    audio -> Whisper -> DeBERTa masking -> placeholder map (LOCAL)
          -> masked request -> Groq -> masked reply
          -> local rehydration -> restored answer -> local TTS

It is an **integration and latency** benchmark on a small, deliberately chosen
slice of the same committed corpus, not a second accuracy benchmark. The
44-clip audio benchmark is where recognition and privacy accuracy are actually
measured. This one answers what that benchmark structurally cannot:

  * does a raw private value ever reach the outbound request - checked against
    the bytes handed to ``urllib`` inside ``app/llm.py``, and every payload is
    written to the run folder verbatim so the claim can be verified rather than
    believed;
  * do the placeholders survive the model, or come back corrupted into
    something no local map can fill;
  * does local rehydration restore the right value, including for repeated,
    unknown and malformed placeholders, and across turns that reuse a
    placeholder name for a different person;
  * what each leg costs, with the network round trip reported **separately**
    from the local pipeline.

Live calls are kept small on purpose. Six representative cases cover the
privacy families that matter and are enough to establish that the integration
works; the free tier is token-limited as well as request-limited, so the
default does not spend it proving something six cases already show.

    python Full_Pipeline_V6_EndToEnd_Benchmark.py               6 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --extended    10 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --smoke-test  3 live cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --llm-stub    no key needed
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --offline     payload audit only
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --no-tts
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --list-cases
    python Full_Pipeline_V6_EndToEnd_Benchmark.py --help

Configure the key in the repository-root ``.env``, which is git-ignored::

    cp FullPipeline_V6/.env.example .env      # then edit
    # or, for this shell only:
    export GROQ_API_KEY=...

Without a key the run still does everything local, still builds and audits the
outbound payload, still exercises rehydration, and reports the live sections as
``not run`` with exit code 2. It never invents an API response, and a stubbed
run is labelled as stubbed rather than reported as Groq latency.

Results land in ``research_results/v6_benchmarks/<run_id>/``, tagged
``"benchmark": "v6_end_to_end"``. That is a different root from the frozen V3,
V4 and V5 folders in ``cpu_benchmarks/``, which the report's figures notebook
discovers by globbing.

This file is deliberately thin. The measurement lives in
``model/privacy_evaluation/privacy_eval/end_to_end_benchmark.py``.
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

from privacy_eval.end_to_end_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
