# Implementation Report — Canary 180M Flash Pi 5 Corrections

Scope: correct the four specified defects, harden packaging, and report all
validation honestly. The locked pipeline, model, decoder and segmentation
contracts were not redesigned.

Input archive verified before any edit:
`Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL.zip`,
SHA-256 `2e41f9860a865e4ea31ddf08e3c707c6e63b0915605db32397d5c5ff9bae2574`.

Backend remains the native NeMo reference implementation: `nvidia/canary-180m-flash`
at revision `c770bfa0f70f66ad89dbf9827985f999675943a3`, official
`canary-180m-flash.nemo` (SHA-256
`7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675`),
`nemo_toolkit[asr]==2.5.3`, `torch==2.7.1`, CPU/FP32, beam size 2. No model,
decoder, precision or runtime substitution was made.

---

## Baseline before edits

Development host: Windows 11, x86-64, Python 3.14.0 (not the Pi target).

| Command | Baseline result |
|---|---|
| `python -m compileall -q .` | exit 0 |
| `python -m unittest discover -s tests` | **63 tests, 3 errors** — all three in `test_silero_vad` from a missing `onnx` module (environment, not code). After `pip install onnx onnxruntime`: **63 tests OK** |
| `python -m app.omit_ai_live --self-test` | exit 1 — `FileNotFoundError`, the 700 MB `canary-180m-flash.nemo` is intentionally not bundled |
| `python -m app.omit_ai_live --print-config` | exit 0, valid JSON |
| `python scripts/check_pi.py` | exit 1 — crashed immediately on `import resource` (Linux-only) before running any check |

---

## Defect 1 — forced-split segmenter parity

**Was:** `_check_forced_split()` emitted the segment, then retained the
post-boundary entries of `_active_contributions` and set
`total_voiced_samples` to their total.

**Why that is wrong:** post-boundary samples already processed while waiting
for trailing context could immediately satisfy the next segment's 3,072-sample
minimum-speech eligibility. A later `end_of_stream`/`end_of_file` boundary
could become `endpoint_silence`, changing the emitted segment count.

**Now:** the exact notebook transition, in `app/canary_live_segmenter.py`:

| State after emitting at forced boundary `B` | Value |
|---|---|
| Emitted segment | `[segment_start, B + 5,632)`, `max_duration`, not final |
| Next `segment_start` | `max(12,000, B − 5,632)` |
| Next `segment_unique_start` | `B` |
| Adjacent submitted overlap | 11,264 samples |
| `total_voiced_samples` | **0** |
| Silero model state | continued, not reset |
| Acoustic utterance | kept alive |
| `raw_silence_start`, `last_raw_speech_end`, `last_effective_speech_end` | untouched |
| `pending_forced_boundary` | cleared |

The now-unused `_active_contributions` / `_candidate_contributions` lists were
removed; they were per-frame lists that also grew during long utterances.

`CanarySegment` gained `segment_unique_start_sample` (optional, defaulted) so
the unique boundary is assertable downstream.

**Proof.** `tests/reference_segmenter.py` is a literal, independent offline
implementation of the sample-count rules operating on the complete signal.
`tests/test_forced_split_parity.py` drives both implementations with identical
audio and an identical frozen per-frame Silero probability script across four
chunk plans (whole-signal, 1,600, 512, and 400 random 1–4,000-sample chunks)
and asserts exact equality of segment count, submitted start, unique start,
end, forced boundary, decision sample, endpoint reason, overlap size, final
marker behaviour and reliable flag.

Scenarios covered: speech through one forced split; through multiple forced
splits; silence beginning inside trailing-context collection; endpoint silence
immediately after a forced split; short dropout around the boundary; shutdown
immediately after a forced split; partial final 512-sample frame; dropped audio
during an active utterance; merged transcript ordering.

**The gate was verified to actually catch the defect.** Reintroducing the old
behaviour made the harness fail with exactly the predicted symptom: a boundary
at decision sample 210,656 with reason `end_of_stream` became one at 189,152
with reason `endpoint_silence`. The fix was then restored and the file
confirmed byte-identical to its pre-probe state.

---

## Defect 2 — calibration audio removed from operational pre-roll

**Was:** operational framing started at 12,000, but `_activate()` computed
`segment_start = max(0, candidate_start − 6,144)`. Speech in the first
operational frame produced `segment_start = 5,856`, leaking 6,144 samples of
startup calibration noise into the first Canary segment.

**Now:** `OPERATIONAL_AUDIO_START_SAMPLE = CALIBRATION_SAMPLES` (12,000) in
`integrated_pipeline/config.py` is an explicit lower bound. A single helper
`CanaryLiveSegmenter._operational_start()` clamps:

- the first segment's pre-roll in `_activate()`;
- the retained look-back in `_check_forced_split()`;
- both candidate and idle retention branches in `_trim_store()`.

The calibration formula, the frozen gate and monotonic/UTC timing derivation
are unchanged (timing is derived from `decision_sample − utterance_start_sample`,
which stays consistent under the clamp).

Tests assert every emitted first-segment start is ≥ 12,000 and the submitted
samples contain no calibration prefix, for speech starting: immediately after
calibration; less than one pre-roll window after; more than one pre-roll window
after (where the unclamped value correctly applies); and in the same 0.1 s
callback that completes calibration. One test uses loud random calibration
noise so a leak would be detectable in the submitted array.

**Knock-on effect:** the first unique segment now spans 12,000–148,192 instead
of 5,856–142,048. `tests/test_segmentation.py::test_shutdown_after_forced_emit_uses_control_not_overlap_only_audio`
encoded the old geometry (and its comment documented the leak); its signal was
corrected from `265 × 512` to `277 × 512` samples so it still ends exactly when
the forced segment's trailing context arrives.

---

## Defect 3 — bounded memory, plus a safety limit

**Was:** `_Accumulator.transcripts` held `SegmentTranscript` objects, each
holding a `CanarySegment`, each holding a full NumPy audio array, until the
whole utterance ended. Unbounded for continuous speech on a 4 GB Pi.

**Now:**

1. `SegmentSummary` (`integrated_pipeline/schemas.py`) holds only scalars and
   text: transcript, segment index/ids, sample boundaries, timings,
   reliability/drop counters, endpoint reason and merge diagnostics.
   `submitted_audio_seconds` is computed from the sample interval, not from an
   array.
2. `_process_segment()` converts the transcript to a summary immediately after
   inference and deletes its references to the transcript and the segment.
3. `_worker_loop()` clears its own reference **before** calling `task_done()`,
   so once a segment is accounted for as done its audio is already unreachable
   rather than pinned until the next queue item arrives.
4. `_finalize()` and the failure path operate purely on summaries.
5. The failure path no longer allocates `np.zeros(union_samples)` purely to
   represent a duration: `RecordedUtterance.submitted_audio_seconds_override`
   carries the exact scalar duration with an empty array.
6. The completed-segment queue stays bounded (`maxsize` 4, unchanged).

Tests use `weakref` to prove the segment and its audio array are collected once
inference completes while the accumulator is still open, assert no `ndarray`
survives in the accumulator, and show retention across the maximum number of
segments is under 8 KiB against more than 8 MB of audio for the same span.

### Safety boundary for pathological speech

Configured in `integrated_pipeline/config.py`:

- `MAXIMUM_LOGICAL_UTTERANCE_SECONDS = 120.0`
- `MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS = 16`

Whichever is reached first closes the current **logical downstream** utterance
at a segment boundary. This deliberately lives in `LiveSTTRuntime`, not in the
segmenter, so the locked acoustic segmentation contract is untouched.

On a split: `utterance_safety_split` is emitted; the accumulated text is merged
and released through the same exact 14-word merger and the normal Onkith →
Stage 5 → PII → SQLite path; a fresh accumulator continues the same acoustic
utterance. Canary is not reloaded, no decoder state crosses the boundary, no
text is dropped, and reliability/drop metadata is preserved.

With real forced splits (8.512 s unique each) the 120 s bound trips first,
after 15 segments; the 16-segment bound is the secondary guard for shorter
segments. Onkith runs once per safety-bounded logical utterance in this
exceptional case.

Continuation identity and ordering are covered in the next section.
Utterances below both bounds are behaviourally unchanged, which is asserted
directly.

Edge case handled: if the stream ends immediately after a split, the empty
continuation accumulator is not an unterminated utterance; it emits
`utterance_end_after_safety_split` and produces no extra row and no failure.

---

## Safety-split SQLite ordering

**Was:** continuation parts took sequence numbers from a reserved high range
(`1,000,000 × (part − 1) + parent`). Since `query_session` orders by
`sequence_number`, a continuation sorted *after* every later independent
utterance — a three-way split plus one later utterance stored as
`[1, 2, 1000001, 2000001]`, returning them out of spoken order.

**Now:** `LiveSTTRuntime` owns a monotonic `_downstream_sequence` counter. Each
logical utterance takes the next chronological number at the moment it opens,
so a continuation is numbered *before* any later utterance can claim one. With
no safety split this reproduces the acoustic numbering exactly (1, 2, 3, …), so
ordinary sessions are byte-for-byte unchanged.

Because the sequence number no longer encodes the relationship, it is carried
by dedicated metadata:

| Field | Ordinary utterance | Safety-split parts |
|---|---|---|
| `sequence_number` | next chronological | next chronological, one per part |
| `batch_id` | acoustic utterance id | part 1 unchanged; later `…-part-NNNNNN` |
| `logical_utterance_id` | equals `batch_id` | shared acoustic utterance id |
| `continuation_part_index` | `1` | `1`, `2`, `3`, … |

`transcript_batches` gained two nullable columns for these, added with an
idempotent `ALTER TABLE` migration guarded by `PRAGMA table_info`, so databases
created before the change are upgraded in place. All existing columns,
constraints, WAL/NORMAL settings and the single-writer model are untouched.
`_record_failure` now takes the accumulator directly, so failure rows carry the
same identity as success rows.

**Proof.** `tests/test_safety_split_ordering.py` drives the real STT runtime
into a real SQLite database. Reintroducing the reserved range made **4 of its
6 tests fail**, with the stored order becoming `[1, 2, 1000001, 2000001]` —
the later independent utterance sorting ahead of the continuations. Coverage
includes a three-way split, the legacy-schema migration, and confirmation that
an ordinary session's rows are unchanged.

---

## Defect 4 — real Canary backend validation

**Implemented:**

- `torch.inference_mode()` now wraps `_run_model()`, covering both the single
  warmup and every segment inference. The real backend (no injected
  `model_loader`) raises rather than silently running outside inference mode
  when torch is missing; `audit()["inference_mode_active"]` records which path
  ran, and a test with a recording fake proves one enter/exit pair per model
  call.
- The exact multitask prompt is now verified after construction (task `asr`,
  `en`→`en`, `pnc=no`, `timestamp=no`, no extra slots) and recorded in
  `audit()["active_prompt"]`. A test proves a silently enabled `pnc=yes` is
  rejected.
- `describe_nemo_backend()` introspects the installed API — versions, and the
  live signatures of `EncDecMultiTaskModel.restore_from`,
  `change_decoding_strategy`, `transcribe`, and the
  `MultiTaskTranscriptionConfig` fields — reporting absence rather than raising,
  so NeMo 2.5.3 and 2.7.3 are not assumed identical.
- The existing decoder read-back (strategy, beam size 2, best hypothesis,
  temperature 1.0, generation delta 50, `TransformerAEDBeamInfer` engine) and
  the batch/workers/timestamps/chunking checks are retained.
- `scripts/validate_canary_backend.py` performs the real-model checks: artifact
  SHA-256 and manifest, load and warmup timing, decoder and prompt read-back,
  inference-mode confirmation, independent calls with a repeat proving no state
  contamination, RTF, and idempotent close. `--emit-parity` / `--parity-report`
  produce and compare per-transcript reports across NeMo versions, listing every
  differing transcript.

**Not executed — reported as NOT RUN, not PASS.** The pinned backend cannot be
installed on this development host: `torch==2.7.1` has no wheel for Python
3.14 (earliest offered is 2.9.0) and `nemo_toolkit[asr]==2.5.3` fails metadata
generation. The `.nemo` artifact is also not bundled. Actual output here:

```
[ NOT RUN  ] NeMo/torch import :: torch import failed: ModuleNotFoundError...
[ NOT RUN  ] Speech recognition accuracy :: no --parity-audio supplied...
[ NOT RUN  ] Real model inference :: NeMo/torch unavailable in this environment
Summary: 0 PASS, 0 FAIL, 3 NOT RUN, 0 UNVERIFIED
```

No mock result is reported as real-model validation, and no NeMo 2.5.3 vs 2.7.3
parity claim is made.

---

## Backend version investigation (NeMo 2.7.3 / Silero 6.2.1)

**No version was changed.** Full analysis in `BACKEND_VERSION_DECISION.md`;
`scripts/probe_backend_versions.py` reproduces the evidence and can settle the
question on the Pi with `--install --run`.

What the PyPI evidence actually shows — and it contradicts the naive assumption
that the pinned 2.5.3 is the safer ARM64 choice:

- `nemo_toolkit[asr]==2.5.3` requires **`ctc_segmentation==1.7.4`**, which
  publishes **no wheels for any platform** — pip must compile a C/Cython
  extension on the Pi. That is the most likely `install_pi.sh` failure point.
- `nemo_toolkit[asr]==2.7.3` **dropped that dependency entirely** and needs
  `torch>=2.6.0`, satisfied by the already-pinned torch 2.7.1, which ships
  `torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl`.

That is a reason to *test* 2.7.3 on the Pi, not to switch blind. Installability
is necessary, never sufficient: decoding behaviour, memory and RTF on a 4 GB Pi
are unmeasured, and a NeMo minor version can change transcripts.

Silero is a harder blocker. The `silero_vad.onnx` in 6.2.1 has the **same byte
size (2,327,524) but a different SHA-256** than the pinned 6.0.0 artifact
(`1a153a…88e3` vs `597d30…2004`). The config's `size_bytes` check would pass
while the model changed; only the pinned hash catches it. A different VAD model
changes frame probabilities, hence speech decisions, hence segment boundaries,
hence Canary inputs — bumping Silero is a segmentation change, not a dependency
refresh.

It also means the notebook's boundaries came from a *different VAD model* than
this build ships, so **exact notebook transcript parity cannot be claimed** and
`compare_notebook_parity.py` explicitly fails its "Same Silero model on both
sides" check rather than letting a parity claim be made across that gap.

Status: **UNRESOLVED**. Pins unchanged pending on-device proof.

---

## Notebook parity comparison

`scripts/compare_notebook_parity.py` compares this pipeline against a notebook
export on the same audio across six independently reported dimensions: segment
boundaries, per-segment transcripts, final merged transcript, blank segment
indices, Canary call ordering, and endpoint reasons. Segment boundaries need
only the Silero artifact, so boundary parity remains checkable where NeMo is
absent; transcript dimensions then report NOT RUN rather than being skipped.
`--print-schema` documents the reference JSON format and `--emit` writes this
side's export.

**NOT RUN against the real notebook**: no notebook reference export exists here
(the notebook was intentionally not supplied) and no real speech run is
possible without the backend. The comparison *logic* is proven by
`tests/test_notebook_parity_harness.py`, which asserts it detects boundary
drift, segment-count drift, per-segment and merged transcript differences,
blank-index differences, call-ordering differences, and a mismatched Silero
model — and that missing local transcripts report NOT RUN rather than PASS.

---

## 16 kHz capture reporting

The strict 16 kHz software contract is unchanged and still fails clearly when a
16 kHz stream cannot be opened. What changed is the honesty of the reporting:
the runtime previously emitted `direct_16000_hz: True` in both the
`microphone_open` event and `audit()`, which asserts native hardware capture
that software cannot establish.

`LiveAudioRuntime.classify_16k_capture()` now returns
`native_16000_hz_reported`, `alsa_plug_conversion_likely` or
`requested_16000_hz_unverified`, and the event/audit carry
`requested_16000_hz`, `device_default_sample_rate`, `capture_16000_hz_status`
and a permanently false `native_hardware_16000_hz_verified`. `--list-devices`
reports `accepts_16000_hz_request` plus the status instead of
`supports_direct_16000_hz`. No streaming resampler is enabled; whole-array
`resample_poly` remains for finite replay/tests only.

---

## Pi acceptance test

`scripts/check_pi.py` was rewritten. It previously crashed on Windows at
`import resource` and aborted on the first failure, and its "STT" evidence was
silence plus a synthetic tone, which proves nothing about recognition.

Now every check reports PASS / FAIL / NOT RUN / UNVERIFIED, a missing artifact
or dependency is classified as NOT RUN by walking the exception cause chain
(adapters raise `RuntimeError(...) from ImportError`), and the exit status is
non-zero only for genuine FAILs. `peak_rss_mib()` degrades to `None` where
`resource` is unavailable instead of crashing.

Added coverage: real speech via `--speech-wav` with `--expect-keyword`, a second
independent speech call, RTF marked UNVERIFIED off-ARM, queue-bound and
safety-split behaviour under sustained input, no-text-dropped verification,
forced segments never reaching downstream independently, fail-closed masking
provable without the model installed, and a live-smoke assertion that no
segment starts inside calibration audio. No copyrighted audio is bundled.

`static_self_test()` gained `require_artifacts` so the logic contract can be
checked where the large model files are absent; `--self-test` still requires
them by default.

---

## Packaging

The input archive used Windows backslashes in **all 55 member names**
(`Omit_AI_RPi_Live\app\canary_live_segmenter.py`), which is invalid packaging
for a Linux release.

Measured behaviour, so the claim is precise: `unzip -t` returns **exit 0** on
the original archive, because `-t` only verifies CRCs and never inspects member
names. The defect surfaces on extraction — `unzip` reports

```
warning: ... appears to use backslashes as path separators
```

The Info-ZIP build tested here compensates by converting them to directories,
but the archive is relying on a recovery heuristic rather than on correct
packaging, and behaviour is not guaranteed across unzip implementations. The
corrected archive extracts with **no warning at all**.

`scripts/build_release_zip.py` writes POSIX `/` members only, rejects absolute
paths and `../` traversal, excludes caches, `__pycache__`, `.pyc`, databases,
logs, secrets and the `.nemo` download, preserves the executable bit on `.sh`
files, and uses a fixed timestamp so the build is reproducible (two consecutive
builds produced an identical SHA-256). Its validator is unit-tested, and a test
checks the built archive when present.

---

## Preserved unchanged

Onkith command parsing and its separate music worker; command-only suppression;
mixed-transcript remaining-text forwarding; `stage5.mode = immediate` with the
inactive 3-second buffer unused; TinyBERT PII and structured recognizers;
fail-closed masking; masked-only terminal output; raw and masked SQLite
behaviour and schema; WAL/NORMAL/foreign-key/busy-timeout settings; the single
serialized SQLite writer; session/batch sequencing; queue-overflow reliability;
drain-first shutdown; non-daemon worker ownership; blank-segment and all-blank
handling; and the exact 14-word merger (NFKC, lowercase, apostrophe handling,
alphanumeric comparison, longest suffix/prefix overlap searched 14→1, one-word
overlap permitted, original words preserved, no fuzzy matching).

Forced Canary segment transcripts still never independently reach PII, SQLite,
the terminal or Onkith.
