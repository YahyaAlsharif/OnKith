# Source dataset builders

> The assembled Model V2 corpus these builders feed into is published as
> [`KhalidAlharbi377/pii-detection-multisource-en-saudi-arabic`](https://huggingface.co/datasets/KhalidAlharbi377/pii-detection-multisource-en-saudi-arabic)
> (CC BY 4.0) — 284,619 rows in the same four splits recorded in
> `model/DeBERTa-v3-xsmall/config/model_v2_data_manifest.json`.

Two notebooks that produce local PII corpora. Their **outputs are gitignored**;
only the generators and this file are tracked.

| Notebook | Output | Role |
|---|---|---|
| `OpenPII_1_5M_Unified_PII_Builder.ipynb` | `unified_pii_dataset.csv` + summaries | Multi-source exploration and audit workbook |
| `saudi_PII_Builder.ipynb` | `saudi_mena_pii_dataset.csv` + summaries | Saudi/MENA supplementary generator |

## How these relate to Model V2

The Model V2 pipeline lives in `model/privacy_data_v2/` and is driven from
`model/DeBERTa-v3-xsmall/model_v2_deberta_v3_xsmall.ipynb`. It consumes the
Saudi CSV as one supplementary source and reads the other corpora directly from
their pinned upstream revisions.

Two things from these notebooks are deliberately **not** used by Model V2:

- **their split columns.** Both assign their own `final_split`. The centralised
  Model V2 pipeline owns every split decision, so `final_split` is carried only
  as provenance and never acted on.
- **the unified builder's embedded Saudi generator.** It duplicates
  `saudi_PII_Builder.ipynb`. Model V2 uses the dedicated generator so there is
  one Saudi provenance chain, not two.

Their audited logic *is* reused: the source registry, the label-mapping tables
and drop reasons, offset repair, Gretel entity-string anchoring, structural
SimHash deduplication and the Saudi entity pools were all ported into
`model/privacy_data_v2/`.

## Running them

```powershell
$env:PYTHONUTF8 = "1"
$env:PII_OUTPUT_DIR = "$PWD\model\data"  # optional; this is already the default
$env:PII_RUN_MODE  = "smoke"          # structural test; "full" streams >1.8M rows
```

A full run of the unified builder takes hours. `PII_CACHE_DIR` moves the
Hugging Face cache off this volume; by default it lands in `.pii_hf_cache/`,
which is gitignored.

Optional credentials: `HF_TOKEN` for gated datasets, and
`NATIONAL_ADDRESS_API_KEY` for the official Saudi city lookup (there is a
documented public fallback without it).

## Why the outputs stay local

The unified corpus is hundreds of megabytes and is regenerable from pinned
source revisions. The Saudi corpus is smaller but is likewise regenerable, and
keeping one policy for all generated data means a new output filename cannot be
committed by accident. `.gitignore` allows the builders, documentation, and the
compact Saudi hash/provenance manifest; corpora and summary workbooks stay ignored.

What makes that safe is
`model/DeBERTa-v3-xsmall/config/model_v2_data_manifest.json`, which records the
pinned revisions, mappings, seeds, split configuration, expected counts and
SHA-256 fingerprints needed to regenerate an identical supplementary holdout.
