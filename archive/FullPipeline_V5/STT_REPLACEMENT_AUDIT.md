# Whisper to Parakeet 110M TDT replacement audit

## Result

The active V4 speech recognizer is NVIDIA `nvidia/parakeet-tdt_ctc-110m` through
the checkpoint's native NeMo TDT/RNNT branch. The old Whisper adapter and active
Whisper imports were removed. The adapter remains a drop-in
`CanarySegment -> SegmentTranscript` component and the existing STT worker,
queues, transcript merger, privacy path, database writer and downstream
consumers retain their contracts.

The exact checkpoint is pinned to revision
`431a349f3051ab85c22b9b7a2741b5fe77065665`, filename
`parakeet-tdt_ctc-110m.nemo`, size `459243520` bytes and SHA-256
`1e6b5c3bcaa390b479bd8112512ede515d2ef5b61495568f2c266d6d5e680c3a`.
Runtime startup rejects a different checkpoint, a CTC branch, a non-TDT greedy
implementation, or decoder settings that differ from the required values.

## Required configuration

| Setting | Applied value |
|---|---:|
| Decoder | TDT/RNNT `greedy_batch` |
| Greedy max symbols | 10 |
| Temperature | 1.0 |
| VAD onset / negative threshold | 0.50 / 0.35 |
| VAD window | 512 samples |
| Minimum speech / silence | 0.25 s / 0.10 s |
| Speech pad | 0.03 s |
| Pre-roll | 0.40 s requested; 6,880 submitted samples with pad |
| Start confirmation | 0.20 s |
| Endpoint silence | 2.50 s / 40,000 samples |
| Maximum unique segment | 8.50 s / 136,000 samples |
| Forced overlap | 0.39 s / 6,240 samples |
| Trailing context | 0.35 s requested; 6,080 submitted samples with pad |
| Input chunk | 0.10 s / 1,600 samples |
| Timestamps / alignments | Off / off |

## Changed files

| File | STT-replacement reason |
|---|---|
| `integrated_pipeline/parakeet_adapter.py` | New once-loaded, serialized CPU FP32 NeMo adapter; pins and verifies the checkpoint, forces and proves TDT decoding, accepts 16 kHz mono segment arrays, and returns the existing transcript schema. |
| `integrated_pipeline/whisper_adapter.py` | Removed because Whisper is no longer an active backend. |
| `integrated_pipeline/config.py` | Replaced Whisper constants with exact Parakeet identity/decoder pins and applied the required VAD/segmentation sample counts. Privacy, Stage 5 and storage constants were retained. |
| `config/live_config.json` | Replaced the Whisper block and recorded every required Parakeet/VAD/segmentation value. Non-STT sections were preserved. |
| `integrated_pipeline/__init__.py` | Exports the Parakeet adapter instead of the removed Whisper adapter. |
| `integrated_pipeline/schemas.py` | Documentation now identifies Parakeet; data structures and fields are unchanged. |
| `app/omit_ai_live.py` | Instantiates Parakeet at the same recognizer seam, updates STT self-tests and model labels, and retains the existing lifecycle and downstream wiring. |
| `app/canary_live_segmenter.py` | Applies the required 0.50 onset and 0.35 active-speech thresholds without changing the state-machine architecture. |
| `app/stt_runtime.py` | Replaced two Whisper-specific comments; runtime logic is byte-for-byte otherwise unchanged. |
| `onkith_ui.py` | Suppresses NeMo/Lightning startup noise instead of faster-whisper logging and updates the STT example label. |
| `requirements-rpi.txt` | Replaces faster-whisper/CTranslate2 with pinned torch, NeMo ASR, Transformers and Hugging Face Hub dependencies. |
| `requirements-windows.txt` | Mirrors the pinned Parakeet test runtime for Windows. |
| `install_pi.sh` | Installs the native NeMo/Torch stack, downloads and byte-verifies the exact checkpoint, then loads it and proves TDT `greedy_batch` before accepting installation. |
| `setup_windows.ps1` | Performs the equivalent Windows dependency, checkpoint and model self-test setup. |
| `scripts/check_pi.py` | Replaces Whisper checks with Parakeet package, load-once, decoder-readback, speech and CPU/RTF checks. |
| `scripts/compare_notebook_parity.py` | Uses the new adapter at the existing parity seam. |
| `tests/test_parakeet_adapter.py` | Replaces Whisper adapter tests with drop-in contract, TDT proof, load-once, callback, input and cleanup tests. |
| `tests/test_whisper_adapter.py` | Removed with the obsolete backend. |
| `tests/test_config.py` | Pins the exact model, decoder and requested/effective timing values while retaining privacy/storage assertions. |
| `tests/reference_segmenter.py` | Mirrors the required threshold and timing values for parity checks. |
| `tests/test_segmentation.py` | Verifies threshold hysteresis, 2.50-second endpoints and exact non-frame-aligned 8.50-second forced boundaries. |
| `tests/test_forced_split_parity.py` | Updates parity expectations for the requested pre-roll, overlap and maximum segment. |
| `tests/test_memory_and_safety_split.py` | Updates real-shaped forced-segment fixtures to 8.50 seconds and the new adjacent context. |
| `tests/helpers.py` | Updates synthetic STT segment timing metadata used by downstream regression tests. |
| `tests/test_release_gates.py` | Uses the Parakeet mock and verifies serialized inference; unrelated release gates remain unchanged. |
| `tests/test_notebook_parity_harness.py` | Updates the STT backend description only. |
| `README.md` | Documents the active Parakeet build, exact settings and honest validation boundary. |
| `START_HERE.md` | Updates Windows/Pi install and run instructions for NeMo/Parakeet. |
| `KNOWN_ISSUES.md` | Marks Whisper-specific findings as historical so they are not misreported as Parakeet results. |
| `STT_REPLACEMENT_AUDIT.md` | Records the requested file-by-file scope and final verification audit. |

## Scope verification

- PII masking implementation and packaged DeBERTa artifacts: unchanged.
- SQLite schema, migrations, writer and storage behavior: unchanged.
- Transcript merger, wake-word repair, Onkith logic and downstream release
  path: unchanged.
- Audio callback, capture queue, completed-segment queue and worker threading
  architecture: unchanged.
- External STT contract: unchanged (`CanarySegment -> SegmentTranscript`).
- Model initialization: once per adapter/process, followed by one warm-up;
  inference calls are serialized.
- Active CTC-only deployment: explicitly rejected.
- Unrelated refactors: none intentionally made.

## Verification performed off-device

- Python compile pass for `app`, `integrated_pipeline`, `scripts` and `tests`.
- Static release self-test passed.
- Shell scripts passed `bash -n`; Windows setup passed PowerShell parser checks.
- Full regression suite: 372 tests completed successfully, with two existing
  environment-dependent skips.
- Release archive CRC and POSIX member-path validation are run by the builder.

The exact Parakeet checkpoint was not present in the local Windows cache, and
the target Raspberry Pi was not attached. Therefore a real checkpoint restore,
Pi package installation, microphone run, Pi memory/RTF/thermal profile and Pi
WER remain hardware acceptance steps. The supplied experiment's 15.12% Custom
30 WER, 4.60% LibriSpeech WER and GPU RTF values are not presented as Pi
measurements.
