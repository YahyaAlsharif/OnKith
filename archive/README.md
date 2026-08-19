# Archive — superseded work, kept as evidence

Nothing in this folder is maintained, and nothing here should be deployed. It is
kept because it is how OnKith arrived at the pipeline in
[`FullPipeline_V4/`](../FullPipeline_V4/), which is the only current one.

Each directory is a frozen snapshot: it was working when it was archived, and it
has been moved unchanged rather than edited. Model files inside them are copies
of artifacts that also live under [`model/`](../model/), so a snapshot stays
runnable on its own terms.

| Directory | What it was | Speech-to-text | Privacy model | Superseded by |
|---|---|---|---|---|
| [`moonshine_pi_project/`](moonshine_pi_project/) | First Raspberry Pi speech-to-text experiment. Captured audio directly so RNNoise denoising could be inserted ahead of the recognizer, then used Moonshine's own integrated VAD/segmenter. | Moonshine | none | `moonshine-pi-v2/` |
| [`moonshine-pi-v2/`](moonshine-pi-v2/) | Second speech-to-text harness, and the one that produced the project's Moonshine word-error-rate numbers across recorded and live-microphone conditions. RNNoise ended up **off** and the VAD/segmentation parameters were tuned by hand. | Moonshine v2 Medium Streaming (`moonshine-voice==0.0.69`) | none | `FullPipeline_V1/` |
| [`FullPipeline_V1/`](FullPipeline_V1/) | First pipeline that put speech recognition and privacy masking in one process: segmentation, a 3.0 s transcript inactivity buffer, structured recognizers, whole-word span merging, masked terminal output. | Moonshine v2 Medium Streaming | TinyBERT-4 INT8 (Model V1) | `FullPipeline_V2/` |
| [`FullPipeline_V2/`](FullPipeline_V2/) | Same pipeline plus local SQLite transcript storage and the terminal UI. | Moonshine v2 Medium Streaming | TinyBERT-4 INT8 (Model V1) | `FullPipeline_V3/` |
| [`FullPipeline_V3/`](FullPipeline_V3/) | The Canary pipeline, and the **frozen historical baseline** every Canary measurement in `research_results/` was taken on: Silero VAD, overlap-based segmentation, a 14-word transcript merger, wake-word repair, DeBERTa-v3-xsmall INT8 masking, spoken-digit rendering, SQLite storage and the terminal UI. Benchmarked end to end on a workstation and on a Raspberry Pi 5. | NVIDIA Canary 180M Flash (NeMo, CPU, FP32, beam 2, `pnc: no`) | DeBERTa-v3-xsmall INT8 (Model V2) | `FullPipeline_V4/` |

## What changed after this folder

`FullPipeline_V3` replaced Moonshine with **NVIDIA Canary 180M Flash**, replaced
Moonshine's integrated segmentation with **Silero VAD** plus overlap-based
segmentation and a 14-word transcript merger, and replaced the TinyBERT-4 masker
with **DeBERTa-v3-xsmall INT8** (Model V2, 31 entity types / 63 BIO labels).

`FullPipeline_V4` changed exactly one component: the speech recognizer is
**faster-whisper `base.en`** on CTranslate2 (CPU, INT8, beam 5, with a decoding
hint built from the device's configured names). Everything downstream of the
recognizer is V3's code, unchanged, which is what makes a Canary-against-Whisper
comparison on the same fixtures meaningful.

### Why V3 is kept live-readable rather than trimmed

`FullPipeline_V3/` is the only place the Canary implementation still exists, and
every Canary number in [`research_results/`](../research_results/) - workstation
and Raspberry Pi - was produced by it. Those result folders are frozen: they are
read by the comparison tooling and are never regenerated. The adapter, the config
contract, the acquisition script, the backend validator and the Canary tests are
all still here so that the baseline stays auditable.

The measured consequences of those changes are in the root
[`README.md`](../README.md) and in [`research_results/`](../research_results/).

## Reading these snapshots

- The word-error-rate tables in `moonshine-pi-v2/README.md` describe **Moonshine**
  and say nothing about the current recognizer.
- `FullPipeline_V1` and `FullPipeline_V2` document a 67-label TinyBERT contract.
  Production is 63 labels and has been since Model V2 shipped.
- Install and run instructions in these folders refer to release ZIPs and
  environments that no longer exist. Use `FullPipeline_V4` instead.
- `FullPipeline_V3`'s install scripts still build NeMo and PyTorch and fetch the
  700 MB `.nemo` artifact. They are correct for V3 and are the record of what the
  Canary deployment cost; V4 needs none of it.
