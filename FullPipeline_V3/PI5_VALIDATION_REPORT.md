# Raspberry Pi 5 Validation Report

## Status

**PENDING — NOT RUN ON PHYSICAL RASPBERRY PI 5 HARDWARE.**
**Real-model (NeMo + `.nemo`) validation also NOT RUN.**

No Pi RSS, model RTF, microphone latency, long-run stability, or Onkith
command-latency result is claimed. This package is **not** production-ready:
production readiness requires physical Pi evidence that does not yet exist.

## Development host used for the checks below

| Property | Value |
|---|---|
| OS | Windows 11 (10.0.26200), x86-64 |
| Python | 3.14.0 |
| Architecture | AMD64 — **not** the `aarch64` target |
| NeMo / torch | not installed (see blockers) |
| Canary `.nemo` | not present (intentionally not bundled) |

Nothing measured on this host may be reported as a Raspberry Pi result.

---

## Executed here — mock and logic tests only

| Check | Status | Evidence |
|---|---|---|
| `python -m compileall -q .` | PASS | exit 0 |
| `python -m unittest discover -s tests` | PASS | 130 tests, 1 skipped (archive test before the ZIP exists) |
| Safety-split rows sort chronologically in `query_session` | PASS | real SQLite; reserved-range regression fails 4 of 6 |
| Legacy database gains the split-metadata columns | PASS | `ALTER TABLE` migration test |
| Notebook parity comparison logic | PASS | detects all six divergence kinds |
| Forced-split online↔reference parity | PASS | 4 chunk plans × 9 scenarios, exact field equality |
| Parity gate detects the original defect | PASS | reintroducing it flipped `end_of_stream`@210,656 → `endpoint_silence`@189,152 |
| Calibration never enters the first pre-roll | PASS | all first-segment starts ≥ 12,000 across 5 speech-onset cases |
| Segment audio released after inference | PASS | `weakref` proves collection while the accumulator is open |
| Accumulator retention bounded | PASS | < 8 KiB retained vs > 8 MB of audio for the same span |
| 120 s / 16-segment safety split | PASS | both bounds, no text dropped, metadata preserved |
| Exact 14-word merge incl. one-word overlap | PASS | existing merger suite unchanged |
| Blank and all-blank utterances | PASS | existing runtime suite unchanged |
| No forced segment enters downstream alone | PASS | runtime and Pi-check assertions |
| Onkith once after normal final merge | PASS | runtime suite |
| Stage 5 immediate, 3 s buffer unused | PASS | `test_release_gates` |
| PII fail-closed | PASS | regression suite + Pi check |
| SQLite schema/WAL/writer ownership | PASS | storage suite + Pi check |
| Drain-first shutdown across all lanes | PASS | shutdown suite |
| No concurrent Canary inference | PASS | blocking-model thread test |
| Model loads once, warms once, close idempotent | PASS | adapter suite (mocked model) |
| `torch.inference_mode()` wraps warmup + inference | PASS | recording fake proves one enter/exit per call |
| Exact prompt contract verified | PASS | `pnc=yes` rejected |
| Archive member paths are POSIX | PASS | validator unit tests + built-archive check |

`scripts/check_pi.py` on this host: **10 PASS, 0 FAIL, 14 NOT RUN, 0 UNVERIFIED**.

## NOT RUN / UNVERIFIED — the pending list

| Check | Status | Reason |
|---|---|---|
| Target platform is 64-bit ARM | NOT RUN | host is AMD64 |
| Target runtime Python 3.11 | NOT RUN | host is 3.14.0 |
| Pinned package versions | NOT RUN | torch/NeMo/transformers not installed |
| Deployment artifacts present | NOT RUN | `canary-180m-flash.nemo` not bundled |
| NeMo/torch backend importable | NOT RUN | see blockers |
| Canary model load and warmup | NOT RUN | backend unavailable |
| Model load / warmup seconds | NOT RUN | backend unavailable |
| Real speech transcript | NOT RUN | backend unavailable; no speech WAV supplied |
| Expected keyword check | NOT RUN | requires a real transcript |
| Second independent speech call | NOT RUN | backend unavailable |
| Per-segment processing time, RTF | NOT RUN | backend unavailable |
| Peak RSS before/after load, during inference, after close | NOT RUN | backend unavailable; `resource` absent on Windows |
| TinyBERT PII masking removes PII | NOT RUN | `transformers` not installed |
| Input device enumeration | NOT RUN | `sounddevice` not installed |
| Live microphone open at 16 kHz | NOT RUN | no audio backend |
| **Native hardware 16 kHz capture** | **UNVERIFIED** | accepting a 16 kHz request through sounddevice/ALSA never proves native capture; an ALSA plug device resamples silently |
| NeMo 2.5.3 vs 2.7.3 transcript parity | NOT RUN | only one environment available, and neither version installed |
| NeMo 2.7.3 installs on Pi ARM64 | NOT RUN | no ARM64 host; run `probe_backend_versions.py --install --run` |
| NeMo 2.7.3 runs on Pi ARM64 | NOT RUN | same |
| NeMo 2.5.3 installs on Pi ARM64 | UNVERIFIED | its `[asr]` extra must compile `ctc_segmentation` from source (no wheels exist) |
| Notebook segment-boundary parity | NOT RUN | no notebook reference export |
| Notebook per-segment transcript parity | NOT RUN | no reference and no backend |
| Notebook merged transcript / blank / call-order parity | NOT RUN | same |
| Exact notebook transcript parity | **NOT CLAIMED** | different NeMo version *and* different Silero VAD model (6.2.1 ONNX has a different SHA-256) |
| Long-run soak, thermal, drop counts | NOT RUN | requires the target device |

**NOT RUN and UNVERIFIED are not passes.**

---

## Blockers

1. **Pinned backend not installable on the development host.**
   `torch==2.7.1` publishes no wheel for Python 3.14 (earliest offered: 2.9.0),
   and `nemo_toolkit[asr]==2.5.3` fails metadata generation there. The pins
   target Python 3.11 on ARM64, which is the correct place to validate them.
2. **The 700 MB `.nemo` artifact is intentionally not bundled**; it is fetched
   by `scripts/acquire_canary.py` during installation on the device.
3. **NeMo version parity needs two environments.** Run
   `scripts/validate_canary_backend.py --emit-parity parity_253.json` under
   2.5.3 and `--emit-parity parity_273.json` under 2.7.3, then compare with
   `--parity-report`. Until then, no claim is made that 2.5.3 and the 2.7.3
   benchmark reference produce identical transcripts.
4. **The Silero model differs from the notebook's.** silero-vad 6.2.1 ships a
   `silero_vad.onnx` with the same byte size but a different SHA-256 than the
   pinned 6.0.0 artifact. Segment boundaries are therefore expected to differ
   from the notebook, and no exact parity claim is possible until the same VAD
   model runs on both sides. Do not bump Silero to match the notebook without
   re-validating segmentation end to end — see `BACKEND_VERSION_DECISION.md`.
5. **No notebook reference export exists here**, so the six-dimension
   comparison in `scripts/compare_notebook_parity.py` has never been run
   against real notebook output.

---

## Required physical Pi 5 procedure

Target: Raspberry Pi 5, 4 GB, 64-bit Raspberry Pi OS (`aarch64`), Python 3.11,
documented swap/thermal policy, 16-kHz-capable microphone.

1. Record OS/kernel/firmware, Pi revision, memory, storage, microphone and
   driver, power supply, cooling, ambient temperature and free disk.
2. `./install_pi.sh` — preserve output showing ARM64 wheels and the exact model
   and Silero SHA-256 values.
3. `.venv/bin/python -m compileall -q .`
4. `.venv/bin/python -m unittest discover -s tests -v`
5. `./run_live.sh --self-test` — preserve decoder read-back and warmup evidence.
6. `./run_live.sh --print-config`
7. `python scripts/validate_canary_backend.py --json --emit-parity parity_253.json`
   with a real speech WAV via `--parity-audio`.
8. `python scripts/check_pi.py --speech-wav your_speech.wav --expect-keyword <word> --json pi_report.json`
   — every previously NOT RUN row must become PASS or an explained FAIL.
9. `./run_live.sh --list-devices` — record `capture_16000_hz_status` for the
   chosen device; confirm native 16 kHz against the hardware datasheet or
   `arecord --dump-hw-params`, since software cannot establish it.
10. `python scripts/check_pi.py --with-mic --device ID` — remain silent during
    the exact 0.75 s calibration.
11. Exercise: short speech; high-energy non-speech; short pause; 2.496 s
    endpoint; continuous speech beyond 8.512 s (forced split); continuous
    speech beyond 120 s (safety split); Ctrl+C during speech; Onkith play and
    close.
12. Verify one database row per logical utterance, none per forced segment,
    masked-only terminal output, fail-closed behaviour, and clean WAL shutdown.
13. Run a ≥ 30-minute soak with multiple forced splits and commands; record
    drop counts and every queue high-water mark.

## Measurements to fill on target

| Measurement | Result |
|---|---|
| Model artifact SHA verified | Pending |
| Active beam engine and values | Pending |
| Active prompt read-back | Pending |
| `inference_mode_active` | Pending |
| Cold load seconds | Pending |
| Warmup seconds | Pending |
| Peak RSS before load / after load / during inference / after close | Pending |
| Median and P95 model RTF (unique and submitted audio) | Pending |
| Median and P95 endpoint-to-masked-output latency | Pending |
| Onkith command latency | Pending |
| Audio callback maximum seconds | Pending |
| Audio and completed-segment queue high-water marks | Pending |
| Dropped chunks per 30 minutes | Pending |
| Peak CPU temperature / throttling | Pending |
| Native 16 kHz capture confirmed against hardware | Pending |
| NeMo 2.5.3 vs 2.7.3 exact transcript equality count | Pending |
| Clean shutdown and database integrity | Pending |

## Acceptance guidance

Do not describe this deployment as validated or production-ready until every
row above is filled on the target device, the model stays within the 4 GB
envelope without unsafe swapping or termination, RTF and end-to-end latency are
acceptable for the intended use, and the soak shows no queue growth, data loss,
duplicate command execution or database corruption.
