# Model V1 vs Model V2: the privacy classifier on OnKith

**Date:** 2026-08-19 · **Corpus:** `eval/fixtures/ood_dev.jsonl`, 360 speech-like rows
**Reproduce:** `.venv/bin/python eval/compare_classifiers.py --json results.json`

| | Model V1 | Model V2 |
|---|---|---|
| Architecture | TinyBERT-4, INT8 ONNX | DeBERTa-v3-xsmall, INT8 ONNX |
| Ontology | 67 BIO labels | 63 BIO labels, 31 entity types |
| On disk | 14 MB | 78 MB |
| Tokenizer | WordPiece | SentencePiece |

Both run on `CPUExecutionProvider`, through the same span merger and the same
structured regex layer. The only variable is the classifier.

---

## Headline

**Model V2 should ship.** It masks three times more of the private text and
raises false alarms a quarter as often. It costs 5.7× the disk, 2.6× the
memory, and about 2.6 ms more per utterance on this machine.

The recommendation is conditional on one thing that has not been done:
**nothing has been measured on the Pi 5.** See *What is still unknown*.

---

## Accuracy

Two arms, because this device deliberately releases some of what the corpus
marks private. The **deployment** arm applies our `never_mask` list — CITY,
COUNTRY and TIMEZONE are released on purpose so "what time is it in Riyadh"
still works — and the corpus marks 40 cities private, so both models are
penalised identically for a decision that is ours. The **model-only** arm lifts
that policy, and is the number that compares the two classifiers.

### As deployed

| Metric | Model V1 | Model V2 | Change |
|---|---|---|---|
| Typed entity F1 | 0.5439 | **0.6638** | +0.120 |
| Typed precision | 0.6189 | **0.6645** | +0.046 |
| Typed recall | 0.4852 | **0.6631** | +0.178 |
| Private character recall | 0.5079 | **0.8537** | +0.346 |
| **Private character leakage** | 0.4921 | **0.1463** | **−0.346** |
| Over-masking rate | 0.0012 | **0.0009** | −0.0003 |
| Exact-case rate | 0.4028 | **0.6083** | +0.206 |
| **Hard-negative false-positive rate** | 0.6389 | **0.1667** | **−0.472** |

### Model only

| Metric | Model V1 | Model V2 | Change |
|---|---|---|---|
| Typed entity F1 | 0.5724 | **0.7118** | +0.139 |
| Private character leakage | 0.4553 | **0.1115** | −0.344 |
| Hard-negative false-positive rate | 0.6528 | **0.2361** | −0.417 |

The two rows that matter most for a privacy device:

- **Leakage** is the fraction of private characters left in the released text.
  Model V1 leaks about half of them. Model V2 leaks about an eighth.
- **Hard-negative false-positive rate** is how often a sentence with nothing
  private in it got masked anyway. Model V1 does this to nearly two sentences in
  three. Model V2 does it to one in six. This is the difference between a device
  that redacts ordinary conversation and one that mostly does not.

---

## Cost

| | Model V1 | Model V2 | Ratio |
|---|---|---|---|
| Model on disk | 14 MB | 78 MB | 5.7× |
| Load time | 0.45 s | 0.71 s | 1.6× |
| Classify p50 *(utterance-sized)* | 0.8 ms | 3.4 ms | 4.4× |
| Classify p95 *(utterance-sized)* | 1.3 ms | 5.0 ms | 3.9× |
| Classify p95 *(all rows)* | 46.4 ms | 141.2 ms | 3.0× |
| Peak RSS | 285 MB | 753 MB | 2.6× |

**Quote the utterance-sized latency.** Capture stops at 8.5 seconds of speech
and the buffer releases one utterance at a time, so nothing over ~400 characters
can reach the masker on this device. The corpus holds 24 documents of about
5,000 characters to stress the 512-token windowing; they are 6.7% of the rows,
so they land exactly on the all-rows p95 and make it a number about a document
rather than about anything this device will be asked to mask.

Each arm was measured in its own process. `ru_maxrss` is a high-water mark that
never falls, so two models in one process report the second as costing whatever
the first did plus its own growth.

---

## Where the change lands, by entity type

Model-only typed F1, ordered by how much moved.

| Entity | V1 | V2 | Support | |
|---|---|---|---|---|
| ORGANISATION | 0.0000 | **0.5882** | 38 | V1 never gets one right |
| DATE | 0.3600 | **0.9394** | 32 | |
| ZIPCODE | 0.3000 | **0.8000** | 6 | |
| CITY | 0.5867 | **0.9070** | 40 | released by policy anyway |
| CREDITCARDNUMBER | 0.2857 | **0.5882** | 12 | |
| DRIVERLICENSENUM | 0.1333 | **0.5000** | 6 | |
| TAXNUM | 0.1667 | **0.5000** | 6 | |
| PASSPORTNUM | 0.2667 | **0.5000** | 6 | |
| IDCARDNUM | 0.1111 | **0.3750** | 12 | |
| ACCOUNTNUM | 0.2000 | **0.4348** | 18 | |
| SURNAME | 0.7385 | **0.8529** | 72 | |
| GIVENNAME | 0.5976 | **0.6879** | 78 | |
| TELEPHONENUM | 0.7234 | **0.8095** | 24 | |
| EMAIL | 0.8214 | 0.8214 | 66 | regex layer, not the model |
| BUILDINGNUM | 0.8000 | 0.7907 | 24 | flat |
| **STREET** | **0.8136** | **0.5814** | 32 | **regression** |

By category:

| Category | V1 | V2 | |
|---|---|---|---|
| contextual_multi_entity (48) | 0.5852 | **0.8981** | +0.313 |
| long_window (24) | 0.8372 | **1.0000** | +0.163 |
| numeric_structured (84) | 0.3309 | **0.4184** | +0.087 |
| names (72) | 0.6920 | **0.7599** | +0.068 |
| email (60) | 0.7018 | 0.6015 | −0.100 — *see below* |
| hard_negatives (72) | 0.6528 FP | **0.2361 FP** | −0.417 |

---

## Two things not to read off the table

**The email row is not a regression in detection.** Written addresses score
1.0000 on both models — they are caught by the structured regex layer, not by
either classifier. Spoken addresses — *"team dot bot at deepfield dot
engineering"* — score **0.0000 on both**: neither model produces an `EMAIL` span
for them at all. Model V2 scores lower on the category only because it guesses
more, and its wrong guesses still cover most of the address, which is why its
character leakage is a third of Model V1's. On the case above, Model V1 masks
nothing and leaks the whole address; Model V2 masks most of it under the wrong
label.

**Spoken email is unsolved, and it is what a microphone actually produces.**
This is the single largest remaining privacy gap in the pipeline and neither
model closes it. It is a training-data question for Model V3, or a regex
question for the structured layer — "X dot Y at Z dot W" is a tractable pattern.

**STREET is the one real per-entity regression**: 0.814 → 0.581 across 32 cases.
On a device that hears addresses, that is worth a look before the demo, though
character leakage still falls overall.

---

## What is still unknown

**Nothing here was measured on the Raspberry Pi 5.** Every latency and memory
number above is from an Apple Silicon Mac and is only useful as a ratio. The
three questions the Pi has to answer:

1. **Does 753 MB peak RSS fit** alongside Whisper, the surface, and Chromium?
   This is the question most likely to say no.
2. **What does 3.4 ms become** on four Cortex-A76 cores? A 10–20× slowdown is
   plausible, which is still tens of milliseconds per utterance and fine.
3. **Does the 78 MB model change boot time** enough to matter for a demo.

Yahya ships `Full_Pipeline_CPU_Benchmark.py` for exactly this, and his manifest
records `physical_pi5_validation: pending_not_run`.

**The corpus is 360 rows.** Categories of 6–24 cases move visibly on one row
changing. TAXNUM, ZIPCODE, PASSPORTNUM and DRIVERLICENSENUM have 6 gold entities
each; treat those rows as directional, not measured.

**`ood_final.jsonl` was not run.** It is a one-shot frozen holdout that has
already been spent once on this exact artifact. Re-running it would quietly turn
a holdout into a dev set, so it is deliberately not vendored into this repo.

---

## Rollback

Model V1 is still packaged and still loadable. Either of these puts it back:

```bash
ONKITH_PII_ENGINE=tinybert .venv/bin/python -m app.omit_ai_live
```

or set `pii.engine` to `"tinybert"` in `config/live_config.json`. To go back
further, the tree before any of this work is tagged `pre-deberta-20260818` and
branched as `rollback/tinybert-classifier`, and the V1 artifacts are copied
outside the repo under `~/onkith-backups/tinybert-20260818/` with checksums.

---

## Provenance

Model V2 artifacts come from `YahyaAlsharif/OnKith`, `FullPipeline_V3`, commit
`86bb142`. The packaged ONNX is verified against the sha256 pinned in
`config/live_config.json` on every load; it matches the manifest exactly.

The manifest's hashes for the tokenizer, tokenizer config and label map are
**stale upstream** — they match neither the deployment copy nor the canonical
one. Those three files are byte-identical between deployment and canonical, and
the ONNX entry, the one that decides behaviour, is correct. Worth a one-line fix
in Yahya's repo.

Metrics come from Yahya's `privacy_eval/metrics.py` and `schema.py`, vendored
unmodified into `eval/privacy_eval/`, so "typed F1" here means what it means in
his model card. For reference, his held-out IID numbers on 40,909 rows were
0.9588 (V2) against 0.9502 (V1) — much higher than anything here, because that
split is in-distribution written text. This corpus is deliberately the hard,
speech-like one.
