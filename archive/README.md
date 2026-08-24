# Archive — superseded work, kept as evidence

> **Nothing in this folder is current. Do not run any of it.**
> The system to run is [`FullPipeline_V6/`](../FullPipeline_V6/), at the
> repository root. Everything here is a superseded snapshot, kept because it is
> how the project got there.

Nothing in this folder is maintained. Several of these snapshots describe
themselves as "the current deployment package" in their own READMEs — they were,
when they were written. They are not now.

Two of these snapshots are not merely history. **V4** is the system the written
academic report evaluates, and its workstation and Raspberry Pi results are that
report's evidence; V6's recognizer and privacy core are V4's files unchanged, so
a V4-against-V6 comparison on one device is a check on that claim. It is no
longer a stand-in: V6 has its own measurements on both devices. **V5** is the
experiment that produced the assistant layer V6 ships.

Each directory is a frozen snapshot: its implementation and result artifacts
are preserved as they were when archived, with only navigation notes adjusted
when repository paths move. Model files inside them are copies
of artifacts that also live under [`model/`](../model/), so a snapshot stays
runnable on its own terms.

| Directory | What it was | Speech-to-text | Privacy model | Superseded by |
|---|---|---|---|---|
| [`moonshine_pi_project/`](moonshine_pi_project/) | First Raspberry Pi speech-to-text experiment. Captured audio directly so RNNoise denoising could be inserted ahead of the recognizer, then used Moonshine's own integrated VAD/segmenter. | Moonshine | none | `moonshine-pi-v2/` |
| [`moonshine-pi-v2/`](moonshine-pi-v2/) | Second speech-to-text harness, and the one that produced the project's Moonshine word-error-rate numbers across recorded and live-microphone conditions. RNNoise ended up **off** and the VAD/segmentation parameters were tuned by hand. | Moonshine v2 Medium Streaming (`moonshine-voice==0.0.69`) | none | `FullPipeline_V1/` |
| [`FullPipeline_V1/`](FullPipeline_V1/) | First pipeline that put speech recognition and privacy masking in one process: segmentation, a 3.0 s transcript inactivity buffer, structured recognizers, whole-word span merging, masked terminal output. | Moonshine v2 Medium Streaming | TinyBERT-4 INT8 (Model V1) | `FullPipeline_V2/` |
| [`FullPipeline_V2/`](FullPipeline_V2/) | Same pipeline plus local SQLite transcript storage and the terminal UI. | Moonshine v2 Medium Streaming | TinyBERT-4 INT8 (Model V1) | `FullPipeline_V3/` |
| [`FullPipeline_V3/`](FullPipeline_V3/) | Raspberry Pi pipeline with Silero VAD, overlap segmentation, Canary transcription, DeBERTa masking, local storage and UI. | NVIDIA Canary 180M Flash | DeBERTa-v3-xsmall INT8 (Model V2) | `FullPipeline_V4/` |
| [`FullPipeline_V4/`](FullPipeline_V4/) | **The system the written academic report evaluates.** V3's pipeline with faster-whisper in place of Canary; everything downstream of the recognizer is unchanged. Its workstation and Raspberry Pi results are the report's evidence and are frozen. | faster-whisper `base.en` (CTranslate2, CPU, INT8, beam 5) | DeBERTa-v3-xsmall INT8 (Model V2) | `../FullPipeline_V6/` |
| [`FullPipeline_V5/`](FullPipeline_V5/) | Parakeet recognizer experiment, and where the assistant layer was built: Groq over masked text only, local placeholder rehydration, TTS, the localhost presentation surface, wake-word routing and graceful no-key behaviour. The assistant layer was kept; the recognizer was not. **Never taken to the Raspberry Pi** - see below. | NVIDIA Parakeet 110M TDT (NeMo, CPU, FP32) | DeBERTa-v3-xsmall INT8 (Model V2) | `../FullPipeline_V6/` |

## What changed in V3 and V4

`FullPipeline_V3` replaced Moonshine with **NVIDIA Canary 180M Flash**, replaced
Moonshine's integrated segmentation with **Silero VAD** plus overlap-based
segmentation and a 14-word transcript merger, and replaced the TinyBERT-4 masker
with **DeBERTa-v3-xsmall INT8** (Model V2, 31 entity types / 63 BIO labels).

`FullPipeline_V4` changed exactly one component: the speech recognizer is
**faster-whisper `base.en`** on CTranslate2 (CPU, INT8, beam 5, with a decoding
hint built from the device's configured names). Everything downstream of the
recognizer is V3's code, unchanged, which is what makes a Canary-against-Whisper
comparison on the same fixtures meaningful.

## What changed in V5, and what V6 kept

`FullPipeline_V5` did two separate things, and only one of them survived.

The recognizer swap - Whisper out, **NVIDIA Parakeet 110M TDT** in - brought a
torch and `nemo_toolkit[asr]` dependency tree of about 2 GB, several packages
without aarch64 wheels, and a 438 MiB checkpoint pinned by hash. Workstation
testing did not show a privacy or deployment trade-off good enough to justify
that on a 4 GB Raspberry Pi, so V5 was **deliberately not taken through Pi
evaluation**. There are no V5 Pi results, and their absence is a decision
rather than a gap.

The assistant layer was the part worth keeping: Groq over masked text only, a
local placeholder map that never leaves the device, local rehydration of the
reply, local text to speech, a localhost presentation surface, wake-word
routing, and a degraded mode that still runs everything local when no API key
is configured.

`FullPipeline_V6` is that assistant layer built on V4's core rather than V5's.
The recognizer, the segmenter configuration, the privacy model and its decoder
are V4's files byte for byte -
[`FullPipeline_V6/tests/test_v6_independence.py`](../FullPipeline_V6/tests/test_v6_independence.py)
fails if any of them drifts - and
[`model/privacy_evaluation/tests/test_masking_parity.py`](../model/privacy_evaluation/tests/test_masking_parity.py)
runs fixed text through V3, V4 and V6 in three isolated interpreters and
requires identical masked output, spans, released text and wake-word repair
from all three.

**The written academic report stays at V4**, which is a scoping decision about
the report rather than a gap in the evidence. V6 has six benchmark runs of its
own in
[`research_results/v6_benchmarks/`](../research_results/v6_benchmarks/) —
audio, CPU/text and the assistant round trip, on the workstation **and on a
Raspberry Pi 5**. On the same device and the same fixtures, every V6 quality
metric is identical to V4's to machine precision, so the two describe the same
local core; the run ids are in
[`research_results/README.md`](../research_results/README.md).

### Why V3 remains reproducible

The archived V3 snapshot is the only place the Canary implementation exists.
Every Canary number in [`research_results/`](../research_results/) - workstation
and Raspberry Pi - was produced by it, and those saved numbers remain live
evidence in the V3-against-V4 comparison. A masking-parity test reads the privacy
path from this snapshot and V4 in isolated interpreters to verify the recognizer
swap did not silently change downstream masking. The Canary result folders are
frozen and never regenerated.

The measured consequences of those changes are in the root
[`README.md`](../README.md) and in [`research_results/`](../research_results/).

### Why V4 remains reproducible

The archived V4 snapshot is runnable on its own terms and is the reference the
parity tests compare against. Its result folders under
[`research_results/cpu_benchmarks/`](../research_results/cpu_benchmarks/) -
workstation and Raspberry Pi alike - are frozen, are the written report's
evidence, and are never regenerated. V6 writes to a different results root so a
V6 run can never be written into, or read as, a frozen historical one.

## Reading these snapshots

- The word-error-rate tables in `moonshine-pi-v2/README.md` describe **Moonshine**
  and say nothing about the current recognizer.
- `FullPipeline_V1` and `FullPipeline_V2` document a 67-label TinyBERT contract.
  Production is 63 labels and has been since Model V2 shipped.
- Install and run instructions in these folders refer to release ZIPs and
  environments that no longer exist. Use `FullPipeline_V6` instead.
- `FullPipeline_V5`'s `README.md`, `START_HERE.md` and `ASSISTANT_ADDON.md`
  describe the Parakeet install, including a `nemo_toolkit[asr]` dependency
  tree and a hash-pinned checkpoint fetch. V6 needs none of it: faster-whisper
  resolves `base.en` through the CTranslate2 cache on first use.
- V5's assistant documentation is otherwise still accurate about the assistant,
  because V6 ships that code. The current copy is
  [`FullPipeline_V6/ASSISTANT_ADDON.md`](../FullPipeline_V6/ASSISTANT_ADDON.md).
- `FullPipeline_V3` is a historical snapshot. Its install scripts still build
  NeMo and PyTorch and fetch the 700 MB `.nemo` artifact; they are correct for
  V3 and record what the Canary deployment cost. V4 and V6 need none of it.

Additional pre-V5 development snapshots are retained locally, outside the tracked
public tree. Nothing depends on them: `FullPipeline_V5/` above is the
maintained-as-history version of that work, and it is the one to read.
