# OnKith Model V2 data preparation

This folder builds the auditable training and validation data for Model V2
(`microsoft/deberta-v3-xsmall`, 63 BIO labels). It does not train a model and it
does not modify the live pipeline.

## The ontology

`ontology.py` is the single source of truth and self-checks at import.

**31 entity types, 63 BIO labels.** Derived from the historical TinyBERT
67-label head (33 entities):

| Change | Label | Reason |
|---|---|---|
| removed | `ALLERGIES` | 3 spans in 147,366 pinned training rows |
| removed | `USERNAME` | 16 spans; not an OnKith product need |
| removed | `WEIGHT` | 1 span; no source maps to it |
| stays out | `HEIGHT` | never in the 67-label head; 1 span exists only in validation |
| **added** | `PASSWORD` | protects disclosed credentials in conversational text |

`HEIGHT` was never in the head, so removing three labels nets 33 − 3 = 30, and
adding `PASSWORD` gives **31 entities / 63 labels**. Anyone reasoning from
"67 − 3" will get the wrong number.

## PASSWORD

`PASSWORD` exists to protect a *disclosed secret*, not the word "password".
`password_rules.py` makes that distinction explicit and auditable.

Admitted — the span covers the secret itself:

```
My password is BlueHorse!92        The Wi-Fi password is kaust2026
My phone password is 482913        I changed my password to MoonLight44
Use Summer#2026 to log in
```

Rejected, and valuable as hard negatives instead:

```
I forgot my password               How do I reset my password?
Never share your password          The password must contain eight characters
Password authentication is enabled
```

Also rejected: credential columns in synthetic document dumps, which look
nothing like OnKith speech input. Every decision is categorised and counted in
the manifest, so the label is trained on the right distribution rather than on
raw volume.

## What is implemented

| Module | Responsibility |
|---|---|
| `ontology.py` | The canonical 63-label space, weak-label lists, measured baselines |
| `records.py` | One canonical record shape, strict validation, JSONL IO, adapters for the other two shapes in this repository |
| `label_mapping.py` | Source-label to OnKith mapping; every decision carries a written reason and nothing is guessed |
| `password_rules.py` | `PASSWORD` admission rules |
| `sources.py` | Pinned source registry, offset repair, Gretel entity-string anchoring, script filtering |
| `selection.py` | Targeted gap filling: volume for empty labels, novelty for OOD-weak labels |
| `dedup.py` | Exact + structural + near-duplicate removal, and the contamination firewall |
| `grouped_splits.py` | Leakage-aware splitting, one factor or many, label-aware, with component diagnostics |
| `pools.py` | Partition assembly, the contamination filter and the disjointness proof |
| `augment_pipeline.py` | Training-only augmentation with caps, plus PASSWORD and hard-negative generation |
| `manifest.py` | The committed reproducibility manifest and its verifier |
| `data_audit.py`, `augmentation.py` | Unchanged from the previous phase, except for a widened mutation guard |

## The contamination filter

The firewall rejects supplementary rows as they are *selected*. That is not
enough on its own, because the AI4Privacy component is a **pinned** split rather
than a selected one: whatever upstream put in the official train split arrives
unscreened.

It does contain duplicates. AI4Privacy English ships zero-length `source_text`
rows in both its official train and its official validation split - seven landed
in the pinned project-train split, one in project-test, two are in
`val_original` - so eight training-side rows were byte-identical to a protected
validation row.

`pools.filter_contamination` is the deterministic sweep that removes them,
before the final split and again after augmentation, before anything is written.
The resolution rule never varies:

> The protected validation/OOD row is kept. The colliding training-side copy is
> removed. Protected sets are never modified.

`val_original` therefore stays at 40,909 rows and the frozen fixtures are never
touched. Every removal names both sides and is recorded in the manifest under
`contamination_filter`. `firewall.assert_clean` still runs afterwards - the
filter is not a replacement for the assertion, and the assertion is not a
warning.

`manifest.DATA_PIPELINE_VERSION` guards the caches: a corpus built before this
filter existed reports an older version and is rebuilt automatically rather than
reused.

## Split policy

The supplementary pool is split 80/20 by connected components over four
*enforced* factors: `open_entity_value`, `email_domain`, `canonical_template`,
`organisation_family`. Rows sharing any of them stay on the same side.

Two of those keys are deliberately **narrow**, because a key that connects
unrelated rows protects nothing and starves the holdout:

- `email_domain` skips public mail providers and the RFC 2606 reserved
  `example.*` names. Sharing `@gmail.com` (1,495 rows in the measured pool) says
  nothing about two records; the whole address is still grouped by
  `open_entity_value`, and 4,070 of the pool's 4,105 domains remain enforced.
- `open_entity_value` skips `DATE` and `BUILDINGNUM`. One date value is shared
  by 603 rows drawn from four independent generators, and `BUILDINGNUM:12` by
  347. A calendar coordinate and a house number are public facts, exactly like
  the closed vocabularies below. Date and identifier *formats* are still
  measured by the `numeric_pattern_family` diagnostic.

Names are **not** narrowed. `GIVENNAME:emily` recurring across 508 rows is the
memorisation risk the split exists to control, so it stays enforced even though
it is the single largest source of connectivity. `split_by_factors` reports
`component_diagnostics` - each factor alone, each factor removed, and the
highest-degree keys - so the cause of a large component is measured rather than
guessed.

Three factors are deliberately **not** enforced, because enforcing them destroys
the split rather than protecting it:

- `formatting_family` collapses every run of letters to a single `A`, so all
  alphabetic names share `GIVENNAME:A` and the whole pool becomes one component.
- `numeric_pattern_family` groups every Saudi IBAN under one shape and every
  five-digit postcode under another. Holding those out would leave `ACCOUNTNUM`
  and `ZIPCODE` with no validation at all.
- `entity_value` over **closed vocabularies** (`BANKNAME`, `COUNTRY`,
  `CURRENCY`, `TIMEZONE`, `GENDER`, `SEX`, `TITLE`) has the same problem: there
  are only a dozen currencies, so the same values recur throughout the corpus
  and weld it together. Both `model/data` builders reached this conclusion
  independently and recorded it as an audited split exception.

All three are still **measured and reported** as diagnostics, so format leakage
stays visible without collapsing the pool.

`split_by_factors` refuses to return an empty holdout. If the data really is one
connected component, that is a data problem to fix, not a split to accept.

## Rare-label policy

Unchanged and binding: duplicating one or two examples does not create coverage.
Labels below 100 training spans are reported explicitly, and labels without
enough independent groups to validate are named in the split audit rather than
forced into the holdout.

## Augmentation policy

Training split only. The guard in `augmentation._assert_mutable` rejects every
protected partition — `test`, `val_original`, `val_supplementary`, `ood_dev`,
`ood_final` — at the mutation boundary.

Only conservative, span-safe transforms are used: entity-value diversification,
casing, punctuation-free and whitespace variants, conversational framing,
spoken structured forms, hard negatives and PASSWORD disclosures. Deliberately
excluded: paraphrasing, LLM rewriting, synonym substitution and backtranslation,
all of which can move an entity boundary silently.

Caps keep "diversity" honest: at most 25% of the split augmented, at most 8% per
family, and each family must draw on at least 50 distinct parent rows. Entity
values that appear in the holdout are removed from the inventories first.

## Evaluation isolation

Six sets, kept separate: training-side `test`, `val_original` (40,909),
`val_supplementary` (20% grouped holdout), the combined validation view, OOD dev
and frozen OOD final.

The combined validation set is reconstructed at evaluation time rather than
materialised, because `val_original` is already deterministically addressable
from the pinned upstream revision.

Records marked `partition="ood_final"` or `frozen=true` are rejected by every
mutation and split entry point. The frozen fixtures are read for fingerprints
only, and scored exactly once at the end of the notebook.

## Run the tests

```powershell
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Re-run the cached dataset audit

```powershell
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
..\..\.venv\Scripts\python.exe data_audit.py --output reports/current_dataset_audit.json
```

## Reproducing the supplementary holdout

Corpora are gitignored. `model/DeBERTa-v3-xsmall/config/model_v2_data_manifest.json`
carries the pinned revisions, mappings, filters, seeds, split configuration,
expected counts and SHA-256 fingerprints needed to rebuild them. Run sections
1–11 of `model/DeBERTa-v3-xsmall/model_v2_deberta_v3_xsmall.ipynb`, then:

```python
from manifest import read_manifest, verify_manifest
verify_manifest(read_manifest("committed.json"), read_manifest("regenerated.json"))
```

A difference in only `artifacts.val_supplementary.sha256` means the selection or
split logic changed and the committed holdout is no longer valid.

## Intentionally postponed

- the production migration from the TinyBERT 67-label contract to this 63-label
  model — a separate reviewed release;
- Raspberry Pi latency and memory benchmarking, which requires a trained INT8
  model first;
- collecting or licensing external real-person data.
