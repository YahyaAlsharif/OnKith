# Backend version decision — NeMo and Silero

## Decision

**No version was changed. The pins remain NeMo 2.5.3 / torch 2.7.1 and the
Silero 6.0.0 ONNX artifact.**

> ### Update — real model executed on x86-64 Windows, 2026-08-05
>
> The pinned stack was built and run for the first time. Results:
>
> * The real `canary-180m-flash.nemo` **loads and transcribes correctly**.
>   Warmup 16.6 s; on a 9.155 s English sample the transcript was exact and
>   model time was 1.981 s (RTF 0.22 **on a laptop — not a Pi measurement**).
> * The active decoder was verified against the live model: `beam`, size 2,
>   best hypothesis, temperature 1.0, generation delta 50,
>   `TransformerAEDBeamInfer`, with `torch.inference_mode()` active.
> * **A real defect was found that mocks could not catch.** The adapter sent
>   the prompt as a *list*; NeMo 2.5.3's `parse_multitask_prompt` calls
>   `.items()` on it and requires a *mapping*, raising
>   `AttributeError: 'list' object has no attribute 'items'`. Fixed by sending
>   the single-turn dict form. Slot values, task, decoder and every other
>   contract are unchanged.
> * `nemo_toolkit[asr]==2.7.3` **installed cleanly with no compiler**, while
>   2.5.3 required working around two source-only packages. This strengthens
>   the case for 2.7.3 but is still x86 evidence, not ARM64 evidence.
>
> The version pin was deliberately left alone: switching it remains a decision
> for real Raspberry Pi results.

The validated Kaggle notebook used **NeMo 2.7.3 and Silero 6.2.1**. Matching a
notebook is not by itself a reason to change a deployed backend. The pins stay
until a candidate is proven to install *and run* on a physical Raspberry Pi
ARM64 device and a real-model transcript comparison shows an acceptable
difference. Neither has been done.

Status of the version question: **UNRESOLVED.**

Consequence, stated plainly: **this build cannot claim exact notebook
transcript parity.** It runs a different NeMo version and a different Silero
VAD model than the notebook that produced the validated results.

---

## Evidence actually gathered (PyPI metadata, executed 2026-08-05)

Produced by `python scripts/probe_backend_versions.py`. This is **installability
and artifact evidence only** — it proves nothing about runtime behaviour on a
Pi.

| Finding | Result |
|---|---|
| `nemo_toolkit` 2.5.3 wheel | `nemo_toolkit-2.5.3-py3-none-any.whl` — platform independent |
| `nemo_toolkit` 2.7.3 wheel | `nemo_toolkit-2.7.3-py3-none-any.whl` — platform independent |
| `nemo_toolkit[asr]==2.5.3` extra | requires **`ctc_segmentation==1.7.4`** |
| `ctc_segmentation` 1.7.4 distributions | **source tarball only — no wheels for any platform** |
| `nemo_toolkit[asr]==2.7.3` extra | **no `ctc_segmentation` dependency** |
| `nemo_toolkit` 2.7.3 torch requirement | `torch>=2.6.0` — satisfied by the pinned 2.7.1 |
| `nemo_toolkit` 2.5.3 torch requirement | `torch` (unpinned) |
| torch 2.7.1 Linux ARM64 wheels | `torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl` present (also cp39/310/312/313) |

### What that implies — and does not

The naive assumption is that the currently pinned 2.5.3 is the safer choice on
ARM64. The metadata points the other way:

- **2.5.3 is the version with the ARM64 installation risk.** Its `[asr]` extra
  pulls `ctc_segmentation==1.7.4`, which ships no wheels at all, so pip must
  compile a C/Cython extension on the Pi. That needs a toolchain and build
  time, and is the most likely failure point of `install_pi.sh`.
- **2.7.3 removed that dependency entirely** and is compatible with the already
  pinned torch 2.7.1, which does publish a `cp311` `manylinux_2_28_aarch64`
  wheel (Raspberry Pi OS Bookworm ships glibc 2.36, so `manylinux_2_28` is
  satisfied).

This is a reason to **test** 2.7.3 on the Pi, not a reason to switch to it
blind. Installability is a necessary condition, never a sufficient one:
decoding behaviour, memory footprint and RTF on a 4 GB Pi are all still
unmeasured, and a NeMo minor-version change can alter transcripts.

---

## Silero: a version bump would silently change the VAD model

| Package | `silero_vad/data/silero_vad.onnx` size | SHA-256 |
|---|---:|---|
| silero-vad **6.0.0** (pinned) | 2,327,524 | `597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004` |
| silero-vad **6.2.1** (notebook) | 2,327,524 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |

**Identical byte size, different content.** The config's
`silero.size_bytes = 2_327_524` check would still pass while the model changed
underneath; only the pinned SHA-256 catches it.

A different VAD model produces different frame probabilities, therefore
different speech/silence decisions, therefore **different segment boundaries**,
therefore different Canary inputs and different transcripts. Bumping Silero is
a segmentation change, not a dependency refresh, and it invalidates the
locked segmentation contract until re-validated end to end.

This also means the notebook's segment boundaries were produced by a *different
VAD model* than this build ships. Boundary differences against the notebook are
expected, and `scripts/compare_notebook_parity.py` fails the
"Same Silero model on both sides" check to stop a parity claim being made
across that gap.

---

## How to resolve this on the target device

Run on the physical Raspberry Pi 5, in this order.

1. Establish the current baseline works:

```bash
python scripts/probe_backend_versions.py --json probe_current.json
```

2. Prove (or disprove) that the candidate versions install and import on the Pi.
   This builds a throwaway virtualenv and does not touch the working install:

```bash
python scripts/probe_backend_versions.py --install --run --json probe_candidate.json
```

3. Only if every "installs on target" and "runs on target" check PASSes,
   capture a real-model parity report under each version:

```bash
python scripts/validate_canary_backend.py --parity-audio speech.wav --emit-parity parity_253.json
```

```bash
python scripts/validate_canary_backend.py --parity-audio speech.wav --emit-parity parity_273.json --parity-report parity_253.json
```

4. Compare the full pipeline against the notebook export:

```bash
python scripts/compare_notebook_parity.py --audio speech.wav --reference notebook_reference.json --json parity_report.json
```

5. Decide with evidence in hand:
   - if 2.7.3 installs, runs, and transcripts match or differ acceptably →
     update `requirements-rpi.txt`, `config/live_config.json`
     (`canary.nemo_version`) and the manifest together, then re-run the full
     suite and the Pi acceptance test;
   - if it does not → keep 2.5.3, and record the notebook version difference as
     an accepted, measured deviation.

Do not update `config/live_config.json` or `requirements-rpi.txt` before step 5.
The configuration validator intentionally rejects a mismatched
`canary.nemo_version`, so a partial change fails fast rather than running an
unverified backend.

---

## Current status summary

| Question | Status |
|---|---|
| Does NeMo 2.7.3 install on Pi ARM64? | **NOT RUN** — no ARM64 host available here |
| Does NeMo 2.7.3 run on Pi ARM64? | **NOT RUN** |
| Does NeMo 2.5.3 install on Pi ARM64? | **UNVERIFIED** — needs to compile `ctc_segmentation` from source |
| Do 2.5.3 and 2.7.3 produce identical transcripts? | **NOT RUN** — neither version installable on this host |
| Is Silero 6.2.1's ONNX the same model as 6.0.0? | **NO — confirmed different SHA-256** |
| Does this build match the validated notebook exactly? | **NO / UNRESOLVED** — different NeMo version and different Silero model |
