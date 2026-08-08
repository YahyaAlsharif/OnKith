# Changed Files — defect-correction pass

Relative to the input archive
(SHA-256 `2e41f9860a865e4ea31ddf08e3c707c6e63b0915605db32397d5c5ff9bae2574`).

## Modified

| File | Change |
|---|---|
| `app/canary_live_segmenter.py` | **Defect 1:** forced split now resets `total_voiced_samples` to 0 and the dead `_active_contributions` / `_candidate_contributions` bookkeeping is removed. **Defect 2:** new `_operational_start()` clamps `_activate()` pre-roll, forced-split look-back and both `_trim_store()` branches to sample 12,000. Emits `segment_unique_start_sample`. |
| `app/canary_stt_runtime.py` | **Defect 3:** `_Accumulator` holds `SegmentSummary` scalars instead of `SegmentTranscript`/`CanarySegment`; segment audio released immediately after inference; `_worker_loop()` drops its reference before `task_done()`; failure path carries a scalar duration instead of a zero array; 120 s / 16-segment logical-utterance safety split with `utterance_safety_split`, `_continuation_identity()`, `safety_split_count`, and empty-continuation handling at shutdown. |
| `integrated_pipeline/canary_adapter.py` | **Defect 4:** `torch.inference_mode()` wraps warmup and every inference via `_inference_context()`; exact prompt verification (`_verify_prompt`, `REQUIRED_PROMPT_SLOTS`); `describe_nemo_backend()` API introspection; `audit()` exposes `active_prompt` and `inference_mode_active`. |
| `integrated_pipeline/config.py` | Added `OPERATIONAL_AUDIO_START_SAMPLE`, `MAXIMUM_LOGICAL_UTTERANCE_SECONDS`, `MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS`, `SAFETY_SPLIT_SEQUENCE_BASE`. Existing validated contract untouched. |
| `integrated_pipeline/schemas.py` | Added `SegmentSummary`; `CanarySegment.segment_unique_start_sample` (optional, validated); `SegmentTranscript.summarize()`; `RecordedUtterance.submitted_audio_seconds_override`. |
| `integrated_pipeline/__init__.py` | Export `SegmentSummary` and the new config constants. |
| `app/live_audio.py` | Honest 16 kHz reporting: `classify_16k_capture()`, `capture_16000_hz_status`, `device_default_sample_rate`; `microphone_open` event and `audit()` now report `requested_16000_hz` and `native_hardware_16000_hz_verified: false` instead of `direct_16000_hz: true`; `list_devices()` reports `accepts_16000_hz_request` + status. |
| `app/omit_ai_live.py` | `static_self_test(config, *, require_artifacts=True)`; device listing and banner/config lines reworded to stop claiming verified direct 16 kHz capture. |
| `scripts/check_pi.py` | Rewritten: PASS/FAIL/NOT RUN/UNVERIFIED reporting, `is_absence()` cause-chain classification, no Linux-only `import resource` at module scope, real-speech validation via `--speech-wav`/`--expect-keyword`, second independent call, queue-bound and safety-split checks, RTF marked UNVERIFIED off-ARM, optional `--json` report. |
| `tests/test_segmentation.py` | `test_shutdown_after_forced_emit_uses_control_not_overlap_only_audio` signal corrected `265 × 512` → `277 × 512` for the clamped 12,000 start; stale comment describing the calibration leak replaced. |
| `tests/test_canary_adapter.py` | Added prompt-contract, active-prompt audit, inference-mode wrapping, torch-required, and backend-introspection tests. |
| `tests/test_live_audio.py` | Added `CaptureHonestyTests` for capture classification and audit honesty. |
| `README.md` | Calibration boundary, forced-split transition table, bounded memory and safety split, 16 kHz verification table, NeMo version note, new verification and packaging commands, expanded limitations. |
| `IMPLEMENTATION_REPORT.md` | Rewritten around the baseline, the four defects, what was and was not executed. |
| `PI5_VALIDATION_REPORT.md` | Rewritten with explicit PASS / NOT RUN / UNVERIFIED status and the pending physical procedure. |
| `CHANGED_FILES.md` | This file. |

### Second pass — safety-split ordering and version investigation

| File | Change |
|---|---|
| `app/canary_stt_runtime.py` | Chronological `_downstream_sequence` counter replaces the reserved high range; `_continuation_identity()` replaced by `continuation_batch_id()`; `_record_failure()` takes the accumulator so failure rows share the same identity; `logical_utterance_id` / `continuation_part_index` propagated. |
| `integrated_pipeline/config.py` | Removed `SAFETY_SPLIT_SEQUENCE_BASE`. |
| `integrated_pipeline/schemas.py` | `STTResult` and `RecordedUtterance` gained `logical_utterance_id` and `continuation_part_index`. |
| `integrated_pipeline/sqlite_storage.py` | Two nullable columns plus an idempotent `ALTER TABLE` migration; both insert paths and `StoredTranscriptBatch` carry them. All existing columns, constraints and settings unchanged. |
| `tests/test_memory_and_safety_split.py` | Updated for chronological numbering; added chronological-ordering and unchanged-normal-session tests. |
| `README.md`, `IMPLEMENTATION_REPORT.md`, `PI5_VALIDATION_REPORT.md` | Ordering scheme, version findings, and the unresolved notebook-parity status. |

### Third pass — live masked preview

| File | Change |
|---|---|
| `app/live_preview.py` | **New.** Display-only preview: rolling 30 s raw buffer, whole-buffer re-mask each cycle, blurred unstable tail, selective risk tests, stable-region eviction, salted label registry, blur cap. Owns a worker thread so no other lane waits. |
| `app/canary_stt_runtime.py` | Optional `preview=None` parameter plus `_notify_preview` / `_finish_preview`, both fully guarded. With `None` the authoritative path is unchanged. |
| `app/terminal_ui.py` | New `preview()` render method. Plain, JSON and live-UI modes; never raises. |
| `app/omit_ai_live.py` | Construct, start and close the preview; wire it into `LiveSTTRuntime`. |
| `KNOWN_ISSUES.md` | **New.** Records the `merge_pii_spans` placeholder-numbering issue found during this work and deliberately not fixed. |
| `README.md` | Live masked preview section. |
| `tests/test_live_preview.py`, `tests/test_preview_integration.py` | **New.** 53 tests covering split entities, selective blurring, eviction, registry stability, the blur cap, failure isolation, and that authoritative output is identical with and without the preview. |

Protected files verified byte-identical to the previous release: `structured_pii.py`,
`pii_adapter.py`, `span_merger.py`, `sqlite_storage.py`, `transcript_merger.py`,
`schemas.py`, `canary_adapter.py`, `config.py`, `onkith_music.py`,
`canary_live_segmenter.py`, `live_audio.py`, `live_runtime.py`,
`config/live_config.json`.

### Fourth pass — spoken digits to numerals at render time

| File | Change |
|---|---|
| `app/digit_normalizer.py` | **New.** `normalize_spoken_digits(text)`: display-only conversion of runs of spelled-out digit words into numerals. Four passes (tokenize, tag, resolve `oh`/`double`/`triple` against digit neighbours, group runs and decide which convert). Never raises; falls back to the original text and logs once per session. |
| `app/terminal_ui.py` | `output()` and `session_transcripts()` each normalise their local `masked_text` copy immediately before display. `pii.masked_text` and the stored rows are untouched. |
| `app/live_preview.py` | The rendered `line` is normalised after `_render_line()` decides blurring/redaction and after eviction fingerprints `raw`, so normalisation can never shift a blur boundary or desync the stability check. |
| `tests/test_digit_normalizer.py` | **New.** 30 tests: every rule and worked example from the task spec, the opaque-span (entity placeholder / blur token / redaction marker) guard, idempotency, malformed-input fallback and its once-per-session logging, storage byte-identity with and without the renderer, and preview integration (masker still sees spelled-out digits, blur/redaction decisions unaffected by normalisation). |
| `scripts/scan_digit_normalizer_dryrun.py` | **New.** Read-only dry run against a real `session_transcript.db`: reports how many stored fields the renderer would change, with before/after samples. Not part of the release archive's runtime path; a diagnostic only. |

Threshold decision for the "N words but one value" cases (`twenty five`, `double four`):
counted as their own units, not by raw word count. A tens+unit pair or a resolved
double/triple converts on its own regardless of how many bare digit words surround
it (`twenty five` alone becomes `25`; `double four` alone becomes `44`), matching the
task spec's own worked examples in rules 2 and 3 literally. The three-token threshold
in rule 5 applies only to runs of **bare** single digit words (`one`..`nine`/`oh`) —
`room seven four` (two bare digits) stays as words, `room seven four two` (three)
becomes `room 742`. Bare digits and unit-pairs combine freely in a shared run:
`room double four seven` -> `room 447` (one pair + one bare digit, pair forces
conversion regardless of the bare count).

Protected files verified byte-identical to the previous release: everything listed
above plus `onkith_ui.py`, `requirements-rpi.txt`, `install_pi.sh`, `run_live.sh`,
`check_pi.sh`.

### Fifth pass — V2 fix pass (preview, grammar, PII leak)

Baseline for this pass: `Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL_FIXED.zip`
(SHA-256 `442e6c52194de7bd8eb06bd5e671baade57f791f47e026c57e96c02906c44afc`).

| File | Change |
|---|---|
| `integrated_pipeline/supplemental_pii.py` | **New.** Closes a *confirmed* plaintext leak: extra cue spellings ("bank id", "account id", "customer id", "contact number"), a bounded ASR-filler allowlist inside cued numeric sequences, and a cue-less long-run floor. `SupplementalPIIMasker` re-merges through the authoritative `merge_pii_spans`, so the correction reaches canonical masked text, SQLite, Onkith and display together. |
| `app/omit_ai_live.py` | Wire `SupplementalPIIMasker` between `PIIMaskerAdapter` and `FailClosedMasker`, so supplemental failures still fail closed. |
| `app/live_preview.py` | Blank segments no longer repaint or re-run TinyBERT; render-signature dedup recorded only after a successful draw; generation IDs invalidate queued work the moment an utterance closes; finalize now clears the label registry and regenerates the salt on the worker lane. |
| `app/canary_stt_runtime.py` | Seal the preview *before* the authoritative release rather than after, closing the window in which a queued render could print after `[MASKED]`. |
| `app/digit_normalizer.py` | Rewritten as a context-aware grammar: standalone tens/teens, tens+unit, digit runs, cue-led joining, clock times, and no partly-spelled sequences. Runs touching the blur token or `[[REDACTED]]` are left as words to stop flicker. Numeric literals are inert, so idempotency holds by construction. |
| `app/terminal_ui.py` | Canonical/display split: JSON keeps `masked_text` canonical and adds `display_text`; terminal and live UI render `display_text`. `pii.masked_text` is never overwritten. |
| `scripts/__init__.py` | **New.** Makes `scripts` a real package. `nemo_toolkit` installs its own `site-packages/scripts/`, which shadowed the repo's namespace package and broke two test modules on any host with NeMo — including the Pi. |
| `tests/test_v2_fixes.py` | **New.** 41 tests across repetition, endpoint, worker ordering, grammar, flicker, supplemental PII, canonical/display separation and registry reset. |
| `tests/test_digit_normalizer.py` | Four expectations **inverted**, not relaxed, because the specification reversed them: standalone tens/teens now convert and clock times are in scope. Each still pins an exact output. |

Protected files verified byte-identical to this pass's baseline: all thirteen
(`structured_pii.py`, `pii_adapter.py`, `span_merger.py`, `sqlite_storage.py`,
`transcript_merger.py`, `schemas.py`, `canary_adapter.py`, `config.py`,
`onkith_music.py`, `canary_live_segmenter.py`, `live_audio.py`,
`live_runtime.py`, `config/live_config.json`).

### Sixth pass — wake-word repair

| File | Change |
|---|---|
| `app/wake_word_normalizer.py` | **New.** Rewrites known Canary mis-transcriptions of the wake word ("on kids", "on kiss", "on care", "uncle", "aunt Keith", "Ankit", ...) to the canonical `Onkith`, but only at the *wake position*: utterance start, optionally after up to two greeting words. Case-insensitive, punctuation-preserving, never raises. |
| `app/canary_stt_runtime.py` | `_release_final` repairs the interceptor's copy only. `result` itself is untouched, so when no command is found the speaker's original words continue to masking and storage unchanged. Emits `onkith_wake_word_normalized` carrying the variant and canonical form, never the surrounding transcript. |
| `tests/test_wake_word_normalizer.py` | **New.** 22 tests: every listed variant and every worked example, case/punctuation handling, idempotency, robustness, false-positive guards, and integration tests driving the real `intercept_onkith_command` to prove repaired text is actually accepted. |

The hook sits before PII masking because it has to: once the masker has run,
"aunt Keith" reads "aunt [GIVENNAME_1]" and the wake word is unrecoverable.
`onkith_music.py` is unchanged - its deliberate exclusion of "Ankit" as a
free-floating wake alias still holds, because the repair is position-anchored
and "Someone named Ankit called" never reaches the wake position.

### Added in the second pass

| File | Purpose |
|---|---|
| `BACKEND_VERSION_DECISION.md` | Evidence and decision record for NeMo 2.7.3 / Silero 6.2.1; resolution procedure for the Pi. |
| `scripts/probe_backend_versions.py` | Metadata / install / runtime evidence levels for candidate backend versions; `--install --run` settles it on the Pi. |
| `scripts/compare_notebook_parity.py` | Six-dimension comparison against a notebook reference export. |
| `tests/test_safety_split_ordering.py` | Real-SQLite ordering regression: continuations precede later utterances; three-way split; legacy-schema migration. |
| `tests/test_notebook_parity_harness.py` | Proves the comparison detects every divergence kind and never upgrades NOT RUN to PASS. |

## Added

| File | Purpose |
|---|---|
| `tests/reference_segmenter.py` | Literal, independent offline implementation of the locked sample-count rules; the parity gate's reference. |
| `tests/test_forced_split_parity.py` | Online-vs-reference parity across four chunk plans, all forced-split scenarios, and the calibration-boundary cases. |
| `tests/test_memory_and_safety_split.py` | Audio release after inference, scalar-only accumulator, bounded retention, and the full safety-split policy. |
| `tests/test_release_gates.py` | Silero reset lifecycle, no concurrent Canary inference, Stage 5 immediate, overflow reliability, and archive path rules. |
| `scripts/validate_canary_backend.py` | Real-model validation and NeMo version parity reporting with non-upgradable statuses. |
| `scripts/build_release_zip.py` | POSIX-only release archive builder and member-name validator. |

## Unchanged

`pii_adapter.py`, `structured_pii.py`,
`span_merger.py`, `pipeline.py`, `silero_vad.py`, `transcript_merger.py`,
`inactivity_buffer.py`, `app/onkith_music.py`, `app/terminal_ui.py`,
`app/live_runtime.py`, `onkith_ui.py`, `config/live_config.json`,
`requirements-rpi.txt`, `install_pi.sh`, `run_live.sh`, `check_pi.sh`,
`scripts/acquire_canary.py`, and all model artifacts.

`requirements-rpi.txt`, `config/live_config.json` and the model manifest are
deliberately unchanged. NeMo stays at 2.5.3 and the Silero artifact at 6.0.0
until a candidate is proven to install *and run* on a physical Raspberry Pi and
a real-model transcript comparison shows an acceptable difference. See
`BACKEND_VERSION_DECISION.md`.
