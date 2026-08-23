# Per-label masking policy

OnKith separates two questions that used to be one:

- **what the model detects** — decided by the trained ontology;
- **what OnKith masks** — decided by configuration, at runtime.

Keeping them apart means an ordinary request such as *"what time is it in
Riyadh"* still works while the model correctly recognises `Riyadh` as a `CITY`.
The detection is preserved and reported; only the text substitution is
suppressed.

This is why label removal is never the answer to "we don't want to mask X".
Training still evaluates whether the model detects `CITY` correctly, regardless
of what production chooses to do with that detection.

## Configuration

`config/live_config.json`, under `pii`:

```json
{
  "pii": {
    "mask_policy": {}
  }
}
```

An empty policy — the shipped default — masks everything, exactly as before.
To stop masking a label while keeping its detection:

```json
{
  "pii": {
    "mask_policy": {
      "CITY": false,
      "COUNTRY": false,
      "EMAIL": true
    }
  }
}
```

Result:

| Input | `masked_text` |
|---|---|
| `What time is it in Riyadh?` | `What time is it in Riyadh?` |
| `Email me at ana@example.com` | `Email me at [EMAIL_1]` |

In both cases the entity still appears in `result.entities`.

## Behaviour

Each entity in `PIIResult.entities` carries a `masked` flag:

```python
{
  "start": 19, "end": 25,
  "entity_type": "CITY",
  "text": "Riyadh",
  "placeholder": None,      # no placeholder when not masked
  "masked": False,          # detected, reported, deliberately left visible
  "source": "model_whole_word",
  "confidence": 0.93,
}
```

Placeholder numbering counts only masked spans, so disabling `CITY` in
`Riyadh and Jeddah and ana@example.com` yields
`Riyadh and Jeddah and [EMAIL_1]` rather than a gap in the sequence.

The span trace records `mask_policy_disabled` on any span the policy exempted,
and `raw_model_output` carries `mask_policy` and `policy_exempt_span_count`.

## Where it is applied

`integrated_pipeline/span_merger.py`, in `merge_pii_spans` — after model
inference, BIO repair, whole-word consolidation and overlap resolution, and
immediately before spans become placeholders in `masked_text`.

This is the single choke point where a chosen span becomes a replacement. The
policy is deliberately **not** applied earlier in the inference path: the model
must still be able to detect and report every supported label.

`SupplementalPIIMasker` re-runs `merge_pii_spans`, so it inherits the policy
from the masker it wraps. Without that, a disabled label would be re-masked by
the supplemental layer.

## Safety

`integrated_pipeline/mask_policy.py` fails closed, in priority order:

1. **Default to masking.** A missing policy, a missing label, an empty policy or
   an unreadable policy all mask exactly as before. A deployment that has never
   heard of `mask_policy` is unaffected.
2. **Only an explicit boolean `false` disables masking.** `"false"`, `0`,
   `null` and `[]` are configuration errors, not permission to release data.
   Note that `isinstance(False, int)` is true in Python, so the check tests the
   type exactly — `0` does not disable anything.
3. **Errors are recorded, never silently applied.** Rejected entries land in
   `MaskPolicy.errors` and in the span trace, so a typo surfaces instead of
   quietly becoming a privacy hole.

`FailClosedMasker` semantics are untouched. The policy filter cannot raise, so
it cannot turn a masking failure into a release.

`mask_policy` is deliberately outside `PipelineConfig.validate`'s exact-match
contract: it is operator-tunable configuration, not a pinned deployment
invariant like the model revision or the label count.

## Model independence

The policy operates on normalised entity-type strings and contains no
label-specific branching — a test asserts that no label name appears in
`mask_policy.py`. It works unchanged for any label the model emits, including
labels that do not exist yet.

That was proved by the migration itself. The mechanism carried over from the
67-label Model V1 (TinyBERT) contract to the 63-label Model V2
(DeBERTa-v3-xsmall) ontology without a single change to `mask_policy.py`: a
label absent from the policy — such as the new `PASSWORD` — simply defaults to
masked.

## Tests

`tests/test_mask_policy.py` — 23 tests covering the default path, explicit
enable, explicit disable, mixed policies, placeholder numbering, fail-closed
handling of malformed input, the supplemental wrapper, deployment config and
model independence.

```powershell
& ..\.venv\Scripts\python.exe -B -m unittest tests.test_mask_policy -v
```
