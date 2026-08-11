# OnKith Model V2 data preparation

This folder prepares auditable data for a future privacy model. It does not
train a model, change the 67-label ontology, or modify either existing model.

## What is implemented

- `data_audit.py` reproduces the pinned TinyBERT split and reports entity,
  empty-row, source, region, name-case, unique-value, and email-domain support.
- `augmentation.py` performs deterministic entity substitution, case variants,
  conversational framing, punctuation/whitespace perturbation, spoken e-mail
  and digit conversion, and realistic hard-negative generation.
- `grouped_splits.py` creates one-factor holdouts by entity value, e-mail
  domain, formatting family, canonical template, region, or source dataset.
- `resources/` contains synthetic/curated values and hard negatives. Reserved
  example domains are used so generated e-mail addresses do not target people.
- `prepare_model_v2_data.ipynb` is a readable, cell-by-cell development flow.
- `reports/current_dataset_audit.json` records the current pinned-data audit.

All text-changing operations rebuild the row and recalculate offsets. They
validate every resulting span. They never silently discard an annotation,
assign an invalid span to `O`, or relax the strict alignment contract.

## Evaluation isolation

Records marked either `partition="ood_final"` or `frozen=true` are rejected by
all mutation and split entry points. The frozen Model V2 OOD-final fixtures
must remain outside this directory and must never be loaded as training input.

The official English validation split remains the historical IID test. Grouped
holdouts are development diagnostics, not replacements for that continuity
baseline or the independently frozen OOD-final evaluation.

## Run the tests

From this directory, using the repository environment:

```powershell
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Re-run the cached dataset audit

The dataset revision is pinned in `data_audit.py`. The command is offline-first
and fails instead of silently downloading a different dataset:

```powershell
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
..\..\.venv\Scripts\python.exe data_audit.py --output reports/current_dataset_audit.json
```

Review changes to the report before committing it. A changed count means the
cache, revision, audit logic, or historical split no longer matches the frozen
baseline and must be investigated.

## Split policy

One split cannot generally be disjoint by every surface factor at once. Shared
names, domains, and templates create connected components that can become very
large. `split_by_factor` therefore holds out exactly one declared factor and
reports component size and overlap. The caller must record the factor and use
the corresponding audit in experiment metadata.

## Rare-label policy

Duplicating one or two examples does not create coverage. Labels below 100
training spans are reported explicitly. They require diverse acquisition or
generation before per-label performance claims are allowed. This phase adds
the mechanism and inventories, but it intentionally does not claim that every
rare label has been solved.

## Intentionally postponed

- full Model V2 training;
- selection of augmentation ratios or a final training budget;
- importing the frozen OOD-final suite into any data workflow;
- changing the ontology to include production-only `NUMERICID`;
- collecting or licensing external real-person data;
- choosing one grouped holdout as the universal validation split.
