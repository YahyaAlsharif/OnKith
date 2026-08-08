# Omit AI — Canary 180M Flash for Raspberry Pi 5

Omit AI is a local English speech-transcription and privacy-masking pipeline. This build replaces the former STT section with the exact pinned `nvidia/canary-180m-flash` model, external Silero VAD segmentation, and the notebook-derived 14-word overlap merger. Onkith, immediate Stage 5, TinyBERT/structured PII masking, local SQLite storage, masked terminal output, and drain-first shutdown remain in place.

> Research notice: this is not a medical device. Clinical use requires independent validation, privacy review, and an approved retention policy.

## Live pipeline

```text
16 kHz microphone callback
  -> bounded audio queue
  -> one-time 12,000-sample calibration
  -> Silero ONNX probabilities + frozen RMS gate
  -> exact sample-count segmenter (operational audio starts at sample 12,000)
  -> bounded completed-segment queue
  -> one independent NeMo Canary call per segment
  -> segment audio released immediately after inference
  -> exact 14-word transcript merger
  -> one final utterance transcript
  -> Onkith once
  -> immediate Stage 5
  -> raw SQLite enqueue + PII masking
  -> masked terminal + masked SQLite completion
```

The microphone callback only copies mono float32 audio, computes display RMS/peak, and attempts a non-blocking queue insertion. Silero, Canary, PII, SQLite, and network work never run in the callback.

## Live masked preview

An utterance is only released when the speaker stops, so someone talking for two
minutes would otherwise see nothing. `app/live_preview.py` renders a masked view
in the meantime. It is display only: it never writes to the database, never
reaches Onkith, and never changes the final transcript.

Each completed segment is added to a rolling 30-second buffer of raw text, and
the masker is re-run over the whole buffer every cycle. Masking segments
separately and appending them would be unsafe - a segment ending
`my number is 05511` followed by one starting `95719` contains no complete phone
number on either side, so neither half would be masked.

The trailing words are rendered as `●●●●●` rather than their text, because the
next segment may complete them into an entity. Blurring extends earlier when an
entity touches the end of the buffer, or when a cue such as `my phone number is`
has arrived without its value. Past `BLUR_CAP_WORDS` the run is replaced with
`[[REDACTED]]` so a long spoken number cannot stall the display. The buffer is
held in memory only, is never logged or persisted, and is wiped along with the
label salt at session end.

Preview labels number each distinct person once, so they will not always match
the final transcript's numbering. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Spoken digits to numerals at render time

Canary runs with `pnc: no`, so numbers arrive as words: `the code is four seven
two nine`. `app/digit_normalizer.py` rewrites runs of spelled-out digit words
into numerals for display only, called from `app/terminal_ui.py` (the final
transcript and the stored-session view) and from the preview render in
`app/live_preview.py`, after masking and after the preview's own blur/eviction
decisions. Storage, detection and Onkith keep receiving the original
spelled-out text; nothing about what gets masked or how the preview decides to
blur changes.

A tens-and-unit pair (`twenty five`) or a resolved `double`/`triple`
(`double four`) converts on its own. Bare single digit words only convert in
runs of three or more (`four seven two nine` -> `4729`; `seven four` alone
stays as words). `oh`, `double` and `triple` are only numeric next to an
actual digit word - `oh really` and `double check` are untouched. Text inside
an entity placeholder, the preview's blur token, or `[[REDACTED]]` is never
touched. Any internal failure falls back to the original text unchanged.

## Calibration boundary

Samples 0–11,999 are startup calibration noise and are used **only** to compute
the frozen energy gate. `OPERATIONAL_AUDIO_START_SAMPLE` (12,000) is a hard
lower bound: operational framing starts there, and every pre-roll and retained
audio calculation is clamped to it. A segment can therefore never begin inside
calibration audio, even when speech starts in the very first operational frame
(where the unclamped 6,144-sample pre-roll would otherwise reach back to 5,856).
The calibration formula and the frozen gate are unchanged.

## Bounded memory for continuous speech

Once a segment's Canary inference completes, its NumPy audio array is released.
The utterance accumulator retains a `SegmentSummary` per segment: text plus
scalar sample boundaries, timing, reliability and merge diagnostics. Retention
is O(metadata + text), never O(total audio duration), and all final metrics are
computed from scalar intervals rather than from retained audio.

Pathological never-ending speech is bounded by a **logical utterance safety
split**:

- maximum logical utterance span: 120 seconds
- maximum segments in one logical utterance: 16
- whichever bound is reached first closes the current logical downstream
  utterance at a segment boundary

With real forced splits (8.512 s unique audio each) the 120-second bound is
reached first, after 15 segments. The split merges and releases the text with
the same exact 14-word merger, emits an `utterance_safety_split` event, and
continues later audio as a new logical utterance. Canary is not reloaded and no
decoder state crosses the boundary. In this exceptional case Onkith runs once
per safety-bounded logical utterance rather than once per acoustic utterance.
### Safety-split identity and ordering

Every persisted result takes the **next normal chronological session sequence
number**, so `query_session` (which orders by `sequence_number`) always returns
rows in the order they were spoken — a continuation part appears immediately
after its own part one and before any later independent utterance. No reserved
high-number range is used.

Because the sequence number is purely chronological, the split relationship is
carried by dedicated metadata instead:

| Field | Ordinary utterance | Safety-split parts |
|---|---|---|
| `sequence_number` | next chronological | next chronological, one per part |
| `batch_id` | the acoustic utterance id | part 1 unchanged; later parts `…-part-NNNNNN` |
| `logical_utterance_id` | equals `batch_id` | the shared acoustic utterance id |
| `continuation_part_index` | `1` | `1`, `2`, `3`, … |

Both columns are additive and nullable; databases created before this change
are migrated in place with `ALTER TABLE`. With no safety split the numbering is
identical to the acoustic numbering, exactly as before. Utterances below both
bounds are behaviourally unchanged.

## Pinned STT backend

- Model: `nvidia/canary-180m-flash`
- Revision: `c770bfa0f70f66ad89dbf9827985f999675943a3`
- Artifact: `canary-180m-flash.nemo`
- SHA-256: `7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675`
- Backend: native NeMo `2.5.3`, PyTorch `2.7.1`, CPU, FP32
- Decoder: beam, beam size 2, best hypothesis, temperature 1.0, maximum generation delta 50
- Call: batch 1, English-to-English ASR, punctuation/capitalization off, timestamps off, workers 0, hypotheses on, chunking off

Startup reads back and rejects any different live decoder configuration or beam engine, and additionally verifies the exact multitask prompt (task `asr`, `en`→`en`, `pnc=no`, `timestamp=no`). The model loads once, warms up once, runs serially, and closes once after all downstream work drains. Mutable decoding state is not shared between segment calls. Warmup and every inference run inside `torch.inference_mode()`; the real backend refuses to run if torch is unavailable rather than silently downgrading.

> **Version note — unresolved.** The pinned backend is NeMo `2.5.3` with the
> Silero `6.0.0` ONNX artifact. The validated notebook used NeMo `2.7.3` and
> Silero `6.2.1`. Nothing was changed to match the notebook, because matching a
> notebook is not evidence that a version works on a Raspberry Pi. Silero 6.2.1
> ships a *different* `silero_vad.onnx` (same byte size, different SHA-256), so
> bumping it would silently change VAD behaviour and therefore segment
> boundaries. **This build cannot claim exact notebook transcript parity.**
> See [BACKEND_VERSION_DECISION.md](BACKEND_VERSION_DECISION.md) for the
> evidence gathered and the procedure to resolve it on the target device.

The large official `.nemo` is intentionally not in this ZIP. `scripts/acquire_canary.py` downloads only the pinned revision during installation, verifies the SHA-256, and atomically installs it. Normal startup is offline.

## Silero and segmentation contract

The included `silero_vad.onnx` comes from `silero-vad==6.0.0`: 2,327,524 bytes, SHA-256 `597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004`, default-domain opset 16. It receives exactly 512 samples per probability call.

Calibration consumes exactly 12,000 post-capture samples. RMS is calculated for 23 complete 512-sample frames and the final real 224-sample partial frame using float64 accumulation and `+1e-18`. The frozen gate is `max(1e-5, P95 * 2.25)`. Speech requires both Silero probability at least 0.50 and RMS at least the gate.

| Setting | Requested | Effective |
|---|---:|---:|
| Capture/replay chunk | 0.100 s | 1,600 samples / 0.100 s |
| Calibration | 0.750 s | 12,000 samples / 0.750 s |
| Pre-roll | 0.400 s | 6,144 / 0.384 s |
| Start confirmation | 0.200 s | 3,072 / 0.192 s |
| Minimum speech | 0.200 s | 3,072 / 0.192 s |
| Dropout bridge | 0.128 s | 2,048 / 0.128 s |
| Endpoint silence | 2.500 s | 39,936 / 2.496 s |
| Maximum unique segment | 8.500 s | 136,192 / 8.512 s |
| Forced look-back | 0.350 s | 5,632 / 0.352 s |
| Trailing context | 0.350 s | 5,632 / 0.352 s |
| Adjacent model-input overlap | — | 11,264 / 0.704 s |

Silero state remains continuous through candidates, short dropouts, and forced splits. It resets only after the complete utterance ends or on explicit stream shutdown. Endpoint decisions wait 39,936 samples but the submitted final audio is clipped to 5,632 samples after effective speech.

### Forced-split state transition

When a maximum-duration segment is emitted at forced boundary `B`:

| State | Transition |
|---|---|
| Emitted segment | `[segment_start, B + 5,632)`, reason `max_duration`, not final |
| Next `segment_start` | `max(12,000, B − 5,632)` (retained look-back) |
| Next `segment_unique_start` | `B` |
| Adjacent submitted overlap | 11,264 samples / 0.704 s |
| Voiced eligibility counter | **reset to 0** |
| Silero model state | continued, **not** reset |
| Acoustic utterance | kept alive |
| Raw-silence / last-speech state | untouched |

The voiced-eligibility reset is the correctness-critical part. Retained
look-back, the previous segment's trailing context, and post-boundary frames
already processed while waiting for that trailing context are all audio the
previous segment already owned. Counting them again lets the next segment
satisfy the 3,072-sample minimum-speech rule with no newly voiced audio, which
can turn a later `end_of_stream` boundary into an `endpoint_silence` one and
change the number of emitted segments. `tests/reference_segmenter.py` is a
literal, independent offline implementation of these rules, and
`tests/test_forced_split_parity.py` fails the release if the online state
machine and the reference ever disagree.

### 16 kHz capture: what is and is not verified

This build requires a 16,000 Hz mono float32 stream and fails clearly if one
cannot be opened; no streaming resampler has been parity-qualified. The exact
whole-array `scipy.signal.resample_poly` reference remains available for finite
replay/tests and is never called on live chunks.

Accepting a 16 kHz request through sounddevice/ALSA does **not** prove the
microphone samples natively at 16 kHz — an ALSA `plug` device resamples
silently. Logs, `--list-devices` and the runtime audit therefore distinguish:

| Reported value | Meaning |
|---|---|
| `requested_16000_hz: true` | The application opened a 16 kHz stream |
| `native_16000_hz_reported` | The driver reports 16 kHz as the device default |
| `alsa_plug_conversion_likely` | An ALSA plug/dmix wrapper, which resamples |
| `requested_16000_hz_unverified` | Stream opened, device default differs |
| `native_hardware_16000_hz_verified: false` | Always false in software |

Native hardware capture stays **UNVERIFIED** until confirmed on the physical
device with the actual microphone.

## Raspberry Pi 5 installation

Target: Raspberry Pi 5 with 4 GB RAM, 64-bit Raspberry Pi OS (`aarch64`), Python 3.11, and at least 6 GiB free disk.

```bash
unzip Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL.zip
cd Omit_AI_RPi_Live
chmod +x install_pi.sh run_live.sh check_pi.sh
./install_pi.sh
./run_live.sh --list-devices
./check_pi.sh --with-mic --device 1
./run_live.sh --device 1
```

Select a device whose listing says `accepts_16000_hz_request: true`, and prefer one whose `capture_16000_hz_status` is `native_16000_hz_reported` over `alsa_plug_conversion_likely`. `install_pi.sh` downloads the pinned Canary artifact; after setup, live startup does not require the internet.

## Verification

```bash
.venv/bin/python -m compileall -q .
```

```bash
.venv/bin/python -m unittest discover -s tests -v
```

```bash
./run_live.sh --self-test
```

```bash
./run_live.sh --print-config
```

```bash
python scripts/check_pi.py --speech-wav my_speech.wav --expect-keyword hello
```

```bash
python scripts/validate_canary_backend.py --json
```

```bash
python scripts/probe_backend_versions.py --json probe.json
```

```bash
python scripts/compare_notebook_parity.py --audio speech.wav --reference notebook_reference.json
```

`--self-test` validates artifacts, active decoder state, model warmup, Silero
loading, merger behavior, structured PII, Onkith interception, and fail-closed
privacy behavior.

`scripts/check_pi.py` reports each check as **PASS**, **FAIL**, **NOT RUN** or
**UNVERIFIED** and never converts NOT RUN into PASS. It exits non-zero only for
genuine FAILs, so it can be run off-target to exercise the portable checks while
still refusing to claim Raspberry Pi results. Supply `--speech-wav` (16 kHz mono
16-bit) and `--expect-keyword` to validate real speech recognition; no
copyrighted audio is bundled, so record your own or pass a licensed file.

`scripts/validate_canary_backend.py` performs the real-model checks that mocks
cannot prove: NeMo/torch API introspection, decoder and prompt read-back,
`torch.inference_mode()` confirmation, two independent calls proving no state
contamination, and NeMo version parity reporting.

`scripts/probe_backend_versions.py` decides, with evidence, whether a candidate
backend version is usable on the target. `--install --run` builds a throwaway
virtualenv on the Pi and proves installability and importability without
touching the working install. PyPI metadata alone is never treated as proof.

`scripts/compare_notebook_parity.py` compares this pipeline against a validated
notebook export on the same audio across six dimensions: segment boundaries,
per-segment transcripts, final merged transcript, blank segments, Canary call
ordering, and endpoint reasons. Segment boundaries are comparable using only
the Silero artifact, so boundary parity is checkable even where NeMo is
unavailable; the transcript dimensions then report NOT RUN. Run
`--print-schema` for the reference JSON format.

See `PI5_VALIDATION_REPORT.md`: physical Pi 5 RSS, RTF, long-run microphone, and
command-latency validation are pending until run on the target hardware.

## Release packaging

```bash
python scripts/build_release_zip.py --output ../Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL_FIXED.zip
```

The builder writes POSIX (`/`) member paths only, rejects absolute paths and
`../` traversal, excludes caches, `__pycache__`, `.pyc`, databases, logs,
secrets and the large `.nemo` download, and preserves the executable bit on
shell scripts. Verify on Linux with:

```bash
unzip -t Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL_FIXED.zip
```

```bash
python -m zipfile -t Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL_FIXED.zip
```

## Privacy and Onkith

Normal terminal output contains only the final masked transcript. A masking error fails closed as `[BLOCKED: PII MASKING FAILED]`. Raw and masked transcripts remain in local `session_transcript.db*` files and must not be uploaded.

Optional Onkith YouTube playback reads only these local `.env` keys:

```text
YOUTUBE_API_KEY=your_key_here
YOUTUBE_REGION=SA
```

Examples: `Onkith play Adele easy on me`, `Onkith stop the music`. Onkith sees only the complete merged utterance, once. Only the music query goes to YouTube; the STT and privacy models remain local.

## Limitations

- Physical Raspberry Pi 5 validation is pending; no measured Pi RSS, model RTF, or live command latency is claimed.
- Real-model validation (loading the actual `.nemo` under NeMo) has **not** been executed; it is implemented and pending.
- NeMo 2.5.3 vs 2.7.3 transcript parity has **not** been measured, and the version question is **unresolved**; see [BACKEND_VERSION_DECISION.md](BACKEND_VERSION_DECISION.md).
- This build runs a different NeMo version *and* a different Silero VAD model than the validated notebook, so **no exact notebook transcript parity is claimed**.
- Notebook comparison across segment boundaries, per-segment transcripts, merged transcript, blank segments and call ordering is implemented but **NOT RUN** (no notebook reference export and no real speech run).
- Native hardware 16 kHz capture is UNVERIFIED; only the 16 kHz software request is enforced.
- English-only STT; punctuation/capitalization are deliberately disabled.
- A 16 kHz stream is mandatory in this release; no streaming resampler is enabled.
- PII detection can produce false positives or false negatives.
- SQLite intentionally retains sensitive raw transcripts locally.
- YouTube playback requires network access, API quota, and a supported browser.

