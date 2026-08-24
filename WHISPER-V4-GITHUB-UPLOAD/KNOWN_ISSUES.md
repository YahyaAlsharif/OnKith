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

Both were structured-recognizer misses that the token classifier did **not**
cover either; the entity list came back empty in each case. (Measured against
Model V1, TinyBERT-4. These are regex-layer gaps, so the supplemental layer is
still required under Model V2.)

Still open, by design:

- **Cue coverage is a list, not an understanding.** Spellings outside
  `_CUES`/`_FROZEN_CUE_RESCAN` ("my digits are ...", "the long number is ...")
  still depend on the token classifier alone.
- **The filler allowlist is seven words.** A different disfluency inside a
  sequence ("phone number is zero five, sorry, five one one") still breaks it.
- **The cue-less floor is nine digits.** A 7- or 8-digit identifier with no
  recognised cue is not protected by the floor and falls back to the token
  classifier.
  The threshold is a deliberate trade against masking ordinary speech; lowering
  it would start eating room numbers and counts.
- **`NUMERICID` is a new entity type.** Anything downstream that enumerates
  entity types will see it. Nothing in this repo validates entity types, but an
  external consumer might.

---

## Spoken identifiers can be rebuilt as digits after masking

**Where:** `app/digit_normalizer.py`, downstream of `integrated_pipeline`.

**Status:** open under V3, and **not reproduced under V4**. Kept because the
mechanism is intact and a recognizer change is what suppressed it.

### What happens

Canary ran with `pnc: no`, so numbers arrived as digit words. If the privacy
model does not mask a spoken identifier - phrasing and cue structure decide this,
and the miss is already counted as a privacy-model failure - the display
renderer, which runs *after* masking by design, rewrites the surviving words into
a numeral:

```
recognized : call me back on five five five zero one four seven
released   : call me back on five five five zero one four seven
displayed  : call me back on 5550147
```

The same effect reproduced on a spoken card-style number
(`four one one two six six eight three` -> `41126683`), and it reproduced
identically on the development workstation and on the Raspberry Pi 5. Measured
frequency in the V3 audio benchmark: 2 identifiers across 2 of 32 clips.

**Under V4 the count is zero, and that is not a fix.** Whisper writes numerals
itself, so a number that survives masking is already a numeral in the released
text and the renderer has nothing left to assemble - the leak arrives in the
compact form directly instead of being built downstream. The renderer is
unchanged and would do exactly the same thing to any digit-word run reaching it,
which a different recognizer, a different decoding setting, or a spelled-out
reading would still produce. `audio_postprocessing_amplification.csv` keeps
reporting it, and V4 runs report zero rows.

### Why it matters, and how much

No additional information is released. The digit words were already in the
released text, so no per-entity privacy metric moves. What changes is the *form*:
a copyable seven- or eight-digit numeral is materially more useful to a reader
than the same digits spelled out, and no entity-counting metric would ever show
it. That is why the audio benchmark reports it separately, in
`audio_postprocessing_amplification.csv`.

The reconstruction is display-only. Storage, Onkith and the preview's own
arithmetic all keep the original text, so it reaches the terminal and the JSON
`display_text` field, not the database.

### Why it was not fixed here

Two candidate fixes exist and both change production behaviour:

- normalise spoken numbers *before* the privacy model, so digit sequences reach
  it in the form it handles best;
- re-screen the text after every post-masking transform, so a render step can
  never assemble something the privacy layer declined to protect.

Either is a real change to the release path and belongs in the next model
generation with its own evaluation, not in a documentation freeze.

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

---

## Numeral dates and times are missed where the spelled-out forms were masked

**Where:** the privacy model itself (DeBERTa-v3-xsmall INT8), reached through
`integrated_pipeline/pii_adapter.py`.

**Status:** open, measured, deliberately not fixed. It is a model false
negative, not an interface defect, and the fix is a training change.

### What happens

Canary wrote dates and times as they were spoken; Whisper writes numerals. The
same clip therefore reaches the same model in two different surface forms, and
the model masks one and not the other:

```
V3 (Canary)  ... before the thirtieth of september   ->  before [DATE_1]
V4 (Whisper) ... before the 30th of September.       ->  before the 30th of September.
```

In the committed audio benchmark this accounts for three of the twelve entities
V4 still leaks: `privacy_007` (`12th of March`), `privacy_013`
(`30th of September`) and `privacy_021` (`4th of June`).

### Why the transcript normalizer does not cover it

`spoken_form_normalizer.py` exists to repair forms that are **not written
English** - "billing at example.org" is neither what was said nor what anyone
writes. `12th of March` is ordinary written English, and the model was trained
on ordinary written English. Rewriting it would be papering over a model gap
with a rule, and the rule would have to guess which written form the model
prefers. That is training data, not preprocessing.

Two further DATE/TIME misses in the same benchmark - `Friday`, `6 in the
evening` - are missed under **both** recognizers, so they are not a V4 effect at
all; they are the same model gap seen without any surface change.

### What fixing it would involve

Numeral date and time forms in the Model V3 training mix, then a re-evaluation.
Not a deployment change, and explicitly out of scope for a recognizer swap.

---

## The spoken-address rewrite can absorb one ordinary word

**Where:** `integrated_pipeline/spoken_form_normalizer.py`.

**Status:** open, bounded, accepted.

### What happens

The rewrite turns `<token> at <domain>` into an address so the existing EMAIL
recognizer can see it. The left-hand token is whatever word precedes " at ", and
nothing distinguishes an email local part from an ordinary word by shape alone:

```
Look at example.com for details.   ->  Look@example.com for details.
```

The projection then widens any span found there back over `Look at example.com`,
so the word `Look` is masked along with the domain.

### Why it is accepted

The cost is bounded and one-sided. It can only ever mask *more* of a sentence
that already contains a domain the model would have masked anyway, so it cannot
turn a protected sentence into a leaking one. Scanning every text this
repository has - the audio corpus, `ood_dev`, `ood_final` and the 2,000-row
deployment benchmark, 3,072 texts in total - the rules fire on four regions, all
four of them the spoken email and URL forms they exist for, and this shape does
not occur.

The guards that keep it that narrow are the top-level-domain list and the
lookahead that refuses a name continuing into `@`; without the second one the
rule would bite the front off an address that was already written correctly.
Tightening further - requiring the local part to be lowercase, or checking it
against a word list - trades a bounded overmask for a rule that is harder to
explain and that breaks on a recognizer capitalising differently.

The 2,000-row text benchmark and the `ood_dev` cross-check both score
**identically** with and without the stage (typed F1 0.9452, `ood_dev` 0.6882),
and the audio benchmark's hard-negative false-positive count is unchanged at
1 clip of 7.

---

## Whisper emits U+FFFD for some non-ASCII symbols

**Where:** faster-whisper / CTranslate2, upstream of everything in this package.

**Status:** open, upstream, not worked around.

### What happens

On `privacy_023` the recognizer writes a currency symbol as the Unicode
replacement character:

```
spoken     : the excess is two hundred and fifty pounds
recognized : The excess is <U+FFFD>250 as she will write to us again ...
```

It reproduces byte for byte on Windows x86-64 and on Raspberry Pi OS aarch64,
which rules out a console encoding or a locale on either device: the character
is already U+FFFD when the recognizer hands the segment over. Nothing in this
package re-encodes recognizer output - the text goes from faster-whisper into
the merger as a `str` and is written with `ensure_ascii=False`.

### Why it is not worked around

Guessing which symbol a replacement character used to be is not recoverable
information, and a rule that mapped `<U+FFFD>250` to `£250` would be inventing
content in the one place where the device must not. The entity in question
(`AMOUNT`) is separately destroyed by the recognizer merging two sentences, so
no privacy metric hinges on it. It is recorded here so the mojibake in the
committed V4 artifacts is not read as a bug in this repository.
