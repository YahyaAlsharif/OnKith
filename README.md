# OnKith

**What happens at home, stays at home.**

[onkith.online](https://onkith.online/)

OnKith is a privacy-first voice pipeline that runs entirely on a small edge
device. It listens, transcribes speech locally, removes personally identifiable
information from the transcript, and only then lets the text go anywhere else.
No audio and no raw transcript ever leaves the device.

---

## The problem

Voice assistants are useful because they understand natural speech. They are
uncomfortable because understanding natural speech has, until now, meant
streaming your living room to someone else's servers.

The usual privacy controls are all-or-nothing. You either send the whole
utterance to the cloud, or you send nothing and lose the capability. But most
sentences are not uniformly sensitive:

```text
"Remind me to call Dr. Sarah Chen at 555-0147 about my appointment on Tuesday."
```

The useful intent — *set a reminder about an appointment on Tuesday* — carries
almost no privacy risk. The name and the phone number carry nearly all of it.
Discarding the whole sentence throws away the intent; forwarding the whole
sentence exposes the identifiers.

OnKith separates the two. Sensitive spans are detected and replaced locally,
and only the redacted remainder is available for any downstream use:

```text
"Remind me to call [TITLE_1] [GIVENNAME_1] [SURNAME_1] at [TELEPHONENUM_1]
 about my appointment on [DATE_1]."
```

## Why local processing matters

Redaction is only a privacy guarantee if it happens *before* the data leaves
your control. A cloud service that promises to delete identifiers has already
received them. OnKith performs speech recognition and PII removal on the device
itself, so:

- raw audio is never transmitted, stored, or written to disk;
- the unredacted transcript exists only in memory, only briefly;
- the pipeline keeps working with no network connection;
- if masking fails for any reason, the text is **blocked rather than released**.

That last point is the design rule the rest of the system is built around: the
pipeline fails closed. A masking error never degrades into passing the original
text through.

---

## The pipeline

```text
microphone
   │
   ▼
local speech segmentation        energy gate, 400 ms pre-roll, endpoint detection
   │
   ▼
Moonshine v2 Medium Streaming    on-device speech-to-text, fresh stream per utterance
   │
   ▼
inactivity buffer                merges fragments spoken within 3.0 s
   │
   ▼
TinyBERT-4 INT8 + structured     67-label BIO token classification, ONNX Runtime
   │
   ▼
whole-word span merge            deterministic overlap resolution
   │
   ▼
masked text
```

### Speech-to-text — Moonshine

Speech recognition uses [Moonshine](https://github.com/moonshine-ai/moonshine)
v2 Medium Streaming, an on-device English model. Audio is captured in 100 ms
mono chunks; a local energy gate with pre-roll decides when speech begins and a
0.85 s silence rule decides when it ends. Every utterance gets its own
recognition stream, so no transcript state leaks between utterances.

Measured word error rate on the project's evaluation set was **12.10% on live
microphone input**, with decode real-time factors between 0.60 and 0.72 on
recorded conditions.

### PII masking — TinyBERT

The masking model is a 4-layer TinyBERT token classifier fine-tuned for PII
detection and quantized to INT8 ONNX (14 MB). It predicts 67 BIO labels across
entity types including names, emails, phone numbers, addresses, account and
identification numbers, dates, and organisations.

Three details make the output trustworthy rather than merely plausible:

- **Whole-word consolidation.** Transformer models see subwords. Masking at the
  subword level produces leaks like `[EMAIL]@example[EMAIL]`, where the domain
  survives. OnKith promotes subword evidence to complete words before masking,
  so an email address is removed in full.
- **Sliding windows.** Text longer than the 512-token context is re-encoded as
  overlapping windows with 128-token stride, and every token is scored. Long
  input is never silently truncated, which would leave later sensitive text
  unmasked.
- **Structured recognition.** Regular-expression recognizers catch explicitly
  cued values — email addresses, and phone, account, postal, and identifier
  numbers introduced by a cue phrase. Exact structured spans take priority over
  overlapping model predictions.

### Edge deployment — Raspberry Pi

The target device is a Raspberry Pi 5 with 4 GB RAM running 64-bit Raspberry Pi
OS. Both models must coexist within that budget, so the runtime avoids heavy
dependencies: tokenization uses the low-level `tokenizers` library instead of
`transformers`, which keeps PyTorch out of the process entirely.

Measured on an x86 development laptop, that choice reduces resident memory from
**999 MiB to 324 MiB** and model load time from **30.5 s to 1.4 s**.

---

## Current status

The live pipeline is implemented and validated end to end on a Windows laptop.

| Component | Status |
|---|---|
| Canary 180M Flash STT (V3) | Implemented, benchmarked |
| DeBERTa-v3-xsmall INT8 PII masking (Model V2) | Implemented, held-out typed F1 0.9588 |
| Inactivity buffering | Implemented, unit-tested |
| Fail-closed privacy boundary | Implemented, tested |
| Windows support | Implemented, validated |
| Raspberry Pi 5 deployment | Implemented; **not yet validated on hardware** |

Speech recognition accuracy, PII detection quality, and laptop latency have been
measured. Sustained throughput, thermal behaviour, and live latency on Pi
hardware have not yet been measured and are the immediate next step.

---

## Results and benchmarks

Two notebooks at the repository root produce every number that goes into the
report. Neither trains a model.

| Notebook | Purpose |
|---|---|
| [`Research_Paper_Figures.ipynb`](Research_Paper_Figures.ipynb) | Turns the saved evaluation evidence into publication-quality figures and CSV tables under `research_results/`. |
| [`Full_Pipeline_CPU_Benchmark.ipynb`](Full_Pipeline_CPU_Benchmark.ipynb) | Runs the real pipeline and the accepted INT8 privacy model on **any CPU**, including the Raspberry Pi, and records latency, memory and quality. |

Run the benchmark notebook on a device, commit the folder it writes under
`research_results/cpu_benchmarks/`, then rerun the figures notebook: it picks up
every device it finds and adds the comparison and prediction-parity figures
automatically. Full instructions:
[research_results/README.md](research_results/README.md).

---

## Repository structure

```text
OnKith/
├── FullPipeline_V3/           Current deployment package — the live pipeline
│   ├── app/                   Entry point, audio runtime, terminal UI
│   ├── integrated_pipeline/   Buffering, PII masking, span merging, config
│   ├── models/canary/         Canary 180M Flash manifest (weights acquired on device)
│   ├── models/deberta_pii/    Model V2 INT8 ONNX, tokenizer, label map, manifest
│   ├── models/silero/         Silero VAD ONNX
│   ├── scripts/               Environment checks and release packaging
│   ├── run_live.sh            Linux / Raspberry Pi launcher
│   └── install_pi.sh          Raspberry Pi installer
├── FullPipeline_V1/           History: first integrated pipeline (Moonshine STT)
├── FullPipeline_V2/           History: second integrated pipeline (Moonshine + SQLite/UI)
├── model/                     Research: training notebooks and artifacts
│   ├── DeBERTa-v3-xsmall/     Model V2 — the deployed privacy model
│   └── TinyBERT-4/            Model V1 — superseded, kept for provenance
├── moonshine-pi-v2/           Research: STT benchmarking harness
├── tinybert_pii_rpi/          Model V1 exported PII artifacts
├── research_results/          Report figures, tables, and per-device CPU benchmarks
└── context/                   Coursework and background notebooks
```

The deployment package is self-contained. Everything outside
`FullPipeline_V3/` is research and experimentation history.

`FullPipeline_V1/` and `FullPipeline_V2/` are the earlier integrated pipelines,
kept unchanged as evidence of how the design evolved. They still run Moonshine
STT and the TinyBERT Model V1 masker; neither is maintained, and neither should
be used for deployment. Only `FullPipeline_V3/` is current.

---

## Running the pipeline

Full instructions are in
[FullPipeline_V3/README.md](FullPipeline_V3/README.md); the quick start is
[FullPipeline_V3/START_HERE.md](FullPipeline_V3/START_HERE.md).

### Windows

WSL is not required.

```powershell
cd FullPipeline_V3
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-windows.txt

python -m app.omit_ai_live --print-config
python -m app.omit_ai_live --text-test "My name is Rayyan Khan and my email is rayyan@example.com"
python -m app.omit_ai_live --list-devices
python -m app.omit_ai_live --device 2
```

### Raspberry Pi OS (64-bit)

```bash
cd ~/FullPipeline_V3
chmod +x install_pi.sh run_live.sh check_pi.sh
./install_pi.sh

./check_pi.sh
./run_live.sh --list-devices
./run_live.sh --device 2
```

### Verify without a microphone

```bash
cd FullPipeline_V3
python -m pytest -q
```

`--text-test` above also exercises the whole privacy path without audio: it
loads the Model V2 INT8 ONNX, masks the text, and prints the released
transcript.

---

## Known limitations

- **Not yet validated on Raspberry Pi hardware.** All measurements reported here
  were taken on an x86 laptop. Pi latency, sustained throughput, and thermal
  behaviour are unmeasured.
- **English only.** Both the speech model and the PII model are English.
- **PII, not contextual privacy.** The model detects identifiable entities. It
  does not recognise information that is sensitive because of its meaning rather
  than its form — a medical worry or a family conflict containing no identifiers
  passes through unmasked.
- **Recognition errors propagate.** A misheard name may not be detected as a
  name. Masking quality is bounded by transcription quality.
- **Structured recognition is cue-dependent.** The regex recognizers fire on
  explicit cues such as "my phone number is" or "call me at". A number
  introduced differently — for example `"Call Dr Sarah Chen at 555 123 4567"` —
  is left to the model alone, and is currently missed. Detection recall for
  uncued numeric identifiers is a known weak point.
- **Over-masking occurs.** The whole-word rule occasionally absorbs an adjacent
  word into a span. This errs toward privacy, but it costs some fluency.
- **64-bit ARM only.** There is no 32-bit `moonshine-voice` wheel.
- **Single speaker assumed.** Diarisation is not used.

## Future work

- Validate latency, memory, and thermal behaviour on Raspberry Pi 5 hardware.
- Measure end-to-end latency from end-of-speech to masked output.
- Extend beyond entity-level PII toward contextual privacy classification.
- Add a confidence-based escalation path so uncertain spans can be handled more
  conservatively or deferred to a stronger local model.
- Investigate selective offloading: forwarding only masked text to a larger
  model while sensitive spans remain local.
- Explore Arabic language support.
- Evaluate speaker diarisation for multi-speaker households.

---

## Acknowledgements

Speech recognition uses [Moonshine](https://github.com/moonshine-ai/moonshine)
by Moonshine AI. The PII model is fine-tuned from
[TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) on the
[ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)
dataset. Inference runs on [ONNX Runtime](https://onnxruntime.ai/).
