# TinyBERT OOD baseline findings

This is the first frozen comparison for the current TinyBERT checkpoint. It is
not a Model V2 result and it is not a Raspberry Pi benchmark.

## Reproducibility boundary

- Development fixture: 360 cases, SHA-256
  `eb43a4a3d0b6d22afac9aa85ba3925bdbccf0350916bb0c62fcea8ff9789db47`.
- Frozen final fixture: 360 cases, SHA-256
  `2acdd4022adc0b74659d5823e93bf2b5a9e97e96eeb8ca772a6f7f1ab4f4e438`.
- FP32 ONNX: 57,141,381 bytes, SHA-256
  `a1c46df39ab6b8321988ef1ce6ec398b288f140f679a6b76f108b0adfd10e36c`.
- INT8 ONNX: 14,386,813 bytes, SHA-256
  `9efee1e3a339a7b0fb0e68df414f0aa773456543a5cb7c7cf619974ff6beceee`.
- Host: Windows x86-64, Python 3.11.9. Latencies below are not Raspberry Pi
  measurements.

The frozen IID baseline remains separate: INT8 token micro-F1 0.9696, typed
entity F1 0.9496, private-character recall 0.9969, and leakage 0.0031 on the
same-source synthetic final test.

## OOD headline results

| Partition and layer | Typed entity F1 | Private-char recall | Leakage | Hard-negative FP case rate |
|---|---:|---:|---:|---:|
| Dev FP32 model-only | 0.4229 | 0.3515 | 0.6485 | 0.6250 |
| Dev INT8 model-only | 0.4406 | 0.3500 | 0.6500 | 0.6389 |
| Dev production combined | 0.5529 | 0.6856 | 0.3144 | 0.6389 |
| Final FP32 model-only | 0.4666 | 0.3083 | 0.6917 | 0.6944 |
| Final INT8 model-only | 0.4649 | 0.3209 | 0.6791 | 0.6528 |
| Final production combined | 0.5665 | 0.6819 | 0.3181 | 0.6528 |

The result replicates across the independently generated partitions. Strong IID
scores do not predict robust OOD privacy behavior. Structured and supplemental
recognizers roughly double character coverage relative to model-only inference,
but they do not fix hard-negative overmasking because those false positives come
from the neural model.

The frozen structured recognizer alone had typed precision 1.0 and recall
0.1102 on both partitions. Adding the supplemental recognizer raised character
recall to 0.4313 dev and 0.4399 final, but its deliberately generic
`NUMERICID` fallbacks lower typed precision. This is why privacy-character and
typed-entity metrics must remain separate.

## Production behavior by category

| Category | Dev typed F1 | Dev char recall | Final typed F1 | Final char recall |
|---|---:|---:|---:|---:|
| Names | 0.694 | 0.963 | 0.735 | 0.964 |
| Email | 0.702 | 0.664 | 0.650 | 0.659 |
| Numeric/structured | 0.318 | 0.748 | 0.317 | 0.770 |
| Contextual/multi-entity | 0.592 | 0.491 | 0.636 | 0.504 |
| Long window | 0.558 | 0.665 | 0.333 | 0.524 |
| Hard negatives | 0.000 | n/a | 0.000 | n/a |

The weakest supported final types in combined production evaluation were
`ORGANISATION` (38 gold spans, F1 0.000), `IDCARDNUM` (12, 0.111), `ZIPCODE`
(6, 0.118), `TAXNUM` (6, 0.143), `DRIVERLICENSENUM` (6, 0.143), and `DATE`
(32, 0.275). Email reached F1 0.821 only because the deterministic recognizer
handled every syntactically valid written address.

## Controlled findings

### Capitalization is not the general explanation

For all 12 development names, title-case, lowercase, and uppercase variants
had identical combined-production F1 (0.711) and character recall (0.959).
On final, title-case and lowercase were again identical (F1 0.857, recall
0.986). Uppercase differed only for the Turkish pair `Ayşe Yılmaz`, where the
uppercased Unicode surface caused the model to merge both names as
`GIVENNAME`. The evidence supports Unicode/casing edge tests, not a blanket
claim that ordinary lowercase breaks this uncased model.

Initials remain difficult: final names/initial typed F1 was 0.457 and no case
was an exact typed match. Reordered and conversational names were substantially
better but still not stable across both partitions.

### Written unseen-domain email is a postprocessor success

All 40 written email cases per partition—unseen domains, uppercase, subdomains,
and plus addressing—were exact production matches. Model-only email typed F1
was 0.000, with just 0.152/0.114 character recall on dev/final. The production
regex, not TinyBERT generalization, deserves credit for the written result.

Spoken email stayed weak: production typed F1 was 0.000 for both `spoken` and
`asr_fragment` cases. Character recall was only 0.153/0.185 on dev and
0.147/0.139 on final. A future model/data pipeline needs explicit spoken-email
semantics; the current syntactic email recognizer cannot supply them.

### Numeric privacy coverage and numeric typing diverge

Compact numeric forms had perfect character recall, but spaced and spoken
forms often received generic or wrong labels. On final production:

- compact: typed F1 0.643, character recall 1.000;
- spaced: typed F1 0.000, character recall 0.745;
- spoken: typed F1 0.417, character recall 0.809;
- uncued contextual: typed F1 0.061, character recall 0.884;
- ASR filler: typed F1 0.455, character recall 0.629.

The supplemental layer protects many characters, but it is not a substitute
for contextual type learning.

### Hard negatives expose serious shortcut learning

The final production model masked at least one span in 47 of 72 no-PII cases
(65.28%). Product codes, model numbers, temperatures, fictional reference IDs,
and code fragments triggered in every final example. The most frequent false
labels were `BUILDINGNUM` (11 spans), `AGE` (11), `ZIPCODE` (9), `PASSPORTNUM`
(5), and `SEX` (5). Generic calendar dates and short benign sequences did not
trigger in this fixture version.

This is direct evidence that numeric/pattern shortcuts are not only a recall
problem. Future augmentation needs realistic hard negatives and a split that
holds formatting families out.

### Long windows use the actual 512/128 path

The packaged TinyBERT tokenizer audit measured 725–732 tokens for all 48 long
cases. Every case produced exactly two production windows and every gold entity
was fully covered by at least one window. Therefore the poor results are not
silent truncation.

On frozen final, model-only long-window typed F1 was 0.000 and character recall
0.109. Production caught every written email but no organisation, and it
covered 0.811 of phone characters without assigning an exact correct phone
span. The long-window organisation F1 and character recall were both 0.000.

The full-set median model latency was 26–38 ms, while P95 was approximately
0.97–1.08 seconds because the 24 deliberately long cases occupy more than five
percent of each suite. That is useful relative evidence on this host, but not a
Pi latency claim.

## Quantization

INT8 did not produce one consistent OOD direction. Relative to FP32 it slightly
improved development typed F1 but slightly reduced character recall; on final
it slightly reduced typed F1 while improving character recall and hard-negative
false-positive rate. INT8 produced 57 BIO repairs on both partitions versus 49
dev and 35 final for FP32. Quantization therefore changes individual decisions
and must remain a separate go/no-go comparison, even though it is not the main
cause of the large OOD gap.

## Limits of this baseline

- Fixtures are controlled synthetic text, not recorded speech or measured
  Canary error distributions.
- They intentionally stress policy choices such as whether public dates/times
  and domain-like strings should be masked; those choices must remain explicit.
- OOD token-level metrics are not reported because the fixtures are
  character-span annotations. Historical token metrics remain in the IID file.
- The final set has now been used once for the declared current-TinyBERT
  baseline. Its failures must not be used to tune Model V2.
- Development-host latency includes a mixture of short and deliberately long
  inputs and cannot establish Raspberry Pi feasibility.
