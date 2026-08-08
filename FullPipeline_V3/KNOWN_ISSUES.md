# Known issues

Open problems that are understood but deliberately not fixed. Each one says why
it was left alone, so it does not get rediscovered from scratch.

---

## Residual PII risk after the supplemental recognizers

**Where:** `integrated_pipeline/supplemental_pii.py`.

**Status:** the two confirmed leaks are fixed and covered by tests. What is
listed here is what the fix does *not* cover, so nobody assumes more than was
actually verified.

The confirmed defects were:

```
raw    : phone number is a zero five five a six seven three four four two one
before : released verbatim, 11 digits in plaintext
after  : phone number is a [TELEPHONENUM_1]

raw    : bank id is four five three three two one one one six five
before : released verbatim, 10 digits in plaintext
after  : bank id is [IDCARDNUM_1]
```

Both were structured-recognizer misses that TinyBERT did **not** cover; the
entity list came back empty in each case.

Still open, by design:

- **Cue coverage is a list, not an understanding.** Spellings outside
  `_CUES`/`_FROZEN_CUE_RESCAN` ("my digits are ...", "the long number is ...")
  still depend on TinyBERT alone.
- **The filler allowlist is seven words.** A different disfluency inside a
  sequence ("phone number is zero five, sorry, five one one") still breaks it.
- **The cue-less floor is nine digits.** A 7- or 8-digit identifier with no
  recognised cue is not protected by the floor and falls back to TinyBERT.
  The threshold is a deliberate trade against masking ordinary speech; lowering
  it would start eating room numbers and counts.
- **`NUMERICID` is a new entity type.** Anything downstream that enumerates
  entity types will see it. Nothing in this repo validates entity types, but an
  external consumer might.

---

## The same person can be labelled as two different people

**Where:** `integrated_pipeline/span_merger.py`, in `merge_pii_spans`.

**Status:** open, not fixed, out of scope when found. Do not fix this as a side
effect of unrelated work.

### What happens

Placeholder numbers are allocated by position within a single `mask_text` call:

```python
counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
placeholder = f"[{span.entity_type}_{counts[span.entity_type]}]"
```

So `[GIVENNAME_1]` means "the first given name in this string", not "this
particular person". If one person is mentioned twice, they get two numbers.

Observed on a real run, one speaker, one name spoken twice:

```
raw     : hello my name is rayan and my email address is rayan at example dot
          com please call me tomorrow morning
masked  : hello my name is [GIVENNAME_1] and my email address is [GIVENNAME_2]
          at example dot com please call me tomorrow morning
```

### Why it matters

The masked transcript is the record that gets stored and read later. As written
it says two different people were named. Anyone reading the stored transcript,
or counting distinct entities from it, will draw the wrong conclusion. The
privacy guarantee is intact - nothing leaked - but the meaning is wrong.

It also breaks the reverse direction: you cannot tell from the masked text
whether two mentions were the same person, which is exactly the question a
transcript reader is most likely to ask.

### How it surfaced

The live preview layer (`app/live_preview.py`) needed labels that stay stable
across repaints, so it keys a per-session registry on a salted hash of the
surface form. That gives one person one label for as long as they are on screen.
Running both side by side made the difference visible:

```
preview : ... [GIVENNAME_1] ... [GIVENNAME_1] ...
final   : ... [GIVENNAME_1] ... [GIVENNAME_2] ...
```

The preview is the one behaving sensibly here. It was not built to fix this and
its numbering deliberately does not feed the authoritative path.

### Why it was not fixed

It was found while building a display-only layer under an explicit constraint
that the authoritative transcript, Onkith and the database path stay
byte-identical. Changing placeholder numbering changes stored output, so it was
out of scope.

### What fixing it would involve

Give `merge_pii_spans` a per-call identity map from normalised surface form to
label, so a repeated mention reuses its number. Roughly what
`_LabelRegistry` in `app/live_preview.py` already does, but scoped to the single
call rather than the session, since the authoritative path masks one finalised
utterance at a time and has no session-level state.

Things to decide before doing it:

- Normalisation. `rayan` and `Rayan` are the same person; `rayan` and `rayyan`
  are two spellings Canary may produce for one person, and treating those as
  one requires fuzzy matching, which is a much bigger change.
- Scope. Per call is easy and fixes the case above. Per session would also make
  numbering consistent across utterances, but needs state the masker does not
  currently carry.
- Tests. `tests/test_pii_regression.py` and `tests/test_storage.py` assert
  masked output; both will need updating, and the change alters stored text for
  any transcript with a repeated entity.

### Not affected

Detection, span boundaries, `safe_to_release`, the fail-closed behaviour and the
structured recognizers are all unaffected. This is purely how chosen spans are
numbered when rendered.
