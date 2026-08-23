# Known issues

Open problems that are understood but deliberately not fixed. Each one says why
it was left alone, so it does not get rediscovered from scratch.

Sections that name Whisper document behavior inherited from the original V4
baseline. Where a section has since been re-measured under Parakeet it says so;
where it has not, it must not be read as a current Parakeet result.

The first four sections below were **measured on 2026-08-23** on the committed
44-clip audio benchmark, on a Windows workstation, against the committed V4
Whisper run of the same clips through the same benchmark code
(`research_results/cpu_benchmarks/20260823T092547Z_windows-amd64_audio-parakeet_full/`
against `20260820T145659Z_windows-amd64_audio-whisper_full_workstation/`).

---

## Parakeet drops repeated tokens in a spoken digit run

**Where:** the recognizer, reached through
`integrated_pipeline/parakeet_adapter.py`.

**Status:** open, measured, not worked around. It is a recognizer defect and the
fix is a different recognizer or a decoding change, neither of which belongs in
a documentation freeze.

### What happens

A telephone number read aloud digit by digit loses its repeats:

```
spoken     : five five five, zero one four seven
Whisper V4 : 555-0147
Parakeet V5: five zero one four seven
```

The same shape on `privacy_012` (`five five five, zero one six two` ->
`five zero one six two`) and on `privacy_015`
(`four one one two, six six eight three` -> `four one one two six eight three`).

### Why it matters

It is the single reason V5 scores **0 of 5** long identifiers transcribed
exactly where V4 scored 3 of 5. A mangled identifier is not a leaked
identifier - the digits that reach the privacy model are already wrong - but it
is also not a usable one, and this device exists to be useful about exactly
these strings. It also destroys the surface the privacy model was trained to
recognise, which is how it turns into the leakage in the next section.

### What fixing it would involve

Measuring the same clips against `parakeet-tdt-0.6b-v2`, or against the same
110M checkpoint with a larger `greedy_max_symbols`, and re-running the audio
benchmark. That is an STT evaluation, not a deployment change.

---

## Parakeet's number and address orthography costs the privacy model

**Where:** the seam between the recognizer and
`integrated_pipeline/pii_adapter.py` / `spoken_form_normalizer.py`.

**Status:** open, measured, and the reason V5 has not replaced V4.

### What happens

The privacy model and the structured recognizers were trained and tuned on
written English. Whisper writes written English. Parakeet does not, in two
specific shapes:

```
spoken     : five five five, zero one four seven
Whisper V4 : 555-0147                  -> [TELEPHONENUM_1]
Parakeet V5: five zero one four seven  -> released in the clear

spoken     : billing at example dot org
Whisper V4 : billing at example.org    -> [EMAIL_1]
Parakeet V5: billing at example. org   -> released in the clear
```

The second one is the more damaging of the two, because
`spoken_form_normalizer.py` exists precisely to repair `<local> at <domain>` and
it needs a *dotted* domain to fire. Parakeet's space before the TLD means the
rule never matches, so a spoken email address that V4 masked completely is
released completely. In the 44-clip run the spoken-address rewrite fired on
**four** regions under V4 and on **zero** under V5.

### How much it costs

| | V4 Whisper | V5 Parakeet |
|---|---:|---:|
| Character recall after STT | **0.8314** | 0.7299 |
| Character leakage after STT | **0.1686** | 0.2701 |
| Fully sanitized clips | **16 / 25** | 12 / 25 |
| Entities attributed to a privacy-model failure | **7** | 11 |

The control arms move by almost nothing across the two runs (reference_text
typed F1 0.5254 against 0.5299), which is what establishes that the privacy
model itself is unchanged and the difference is entirely what the recognizer
handed it.

### Why it was not fixed

Two candidate fixes exist and both are real changes with their own evaluation:

- extend `spoken_form_normalizer.py` to close `example. org` into
  `example.org`. Cheap, but it widens a rule whose whole safety argument is how
  narrow it is, and it would need re-scoring against the 3,072-text corpus the
  current rules were validated on;
- fold spoken digit runs to numerals *before* the privacy model rather than
  after it, which is the fix the "spoken identifiers can be rebuilt as digits"
  section below has wanted since V3.

Neither is a final-day change.

---

## Parakeet needs ~577 MiB more resident memory than Whisper

**Where:** the NeMo/torch runtime.

**Status:** open, measured on a workstation, **unmeasured on the Pi**, and the
first thing to check there.

```
                    V4 Whisper    V5 Parakeet
stt_loaded            713.6 MiB     1557.6 MiB
all_models_loaded    1146.5 MiB     1791.9 MiB
peak during run      1269.5 MiB     1846.3 MiB
```

faster-whisper runs on CTranslate2; Parakeet runs on torch plus the NeMo ASR
stack, and that is where the difference lives. On a 15.65 GiB workstation it
does not matter. On a 4 GB Pi 5 with a browser UI and a music player also
resident, 1.85 GiB is most of the budget, and nothing in this repository has
yet observed what happens when it is not enough.

---

## The recognizer swap has not been measured on the Pi at all

**Where:** everywhere.

**Status:** open by necessity. Every V5 number in this repository was produced
on a Windows workstation.

The two runs being compared also used different Python environments - the
Whisper stack and the NeMo stack cannot share one - so onnxruntime 1.27.0 sat
under V4 and 1.22.1 under V5, with matching differences in `transformers` and
`tokenizers`. The bound on what that contributes is visible in the CPU
benchmark, where the *same* privacy artifact scores the *same* 2,000 rows:
typed F1 0.9452 against 0.9450, `ood_dev` 0.6882 against 0.6881. Two ten-
thousandths. Every difference reported above is one to three orders of magnitude
larger than that, so the environment is not what produced them - but it is not
nothing, and a Pi run of both would settle it properly.

---

## The rehydration map is per conversation, the placeholder numbers are per utterance

**Where:** `app/rehydrate.py` (`RehydrationMap`) and
`integrated_pipeline/span_merger.py` (`merge_pii_spans`).

**Status:** open, understood, out of scope. Related to - and downstream of - the
"same person labelled as two different people" section further down.

### What happens

`merge_pii_spans` numbers placeholders by position within a single `mask_text`
call, so `[GIVENNAME_1]` means "the first given name in *this* utterance".
`RehydrationMap` holds a rolling map across the whole conversation so that a
reply can refer to a placeholder introduced several turns earlier - which it
genuinely can, because `GroqClient` keeps six turns of masked history.

Those two are in tension. Two utterances mentioning two different people both
produce `[GIVENNAME_1]`, and the later one overwrites the earlier in the map:

```
turn 1  masked : My name is [GIVENNAME_1].          map: [GIVENNAME_1] -> Alex
turn 2  masked : Ask for [GIVENNAME_1] instead.     map: [GIVENNAME_1] -> Priya
turn 2  reply  : I will ask [GIVENNAME_1].
turn 2  shown  : I will ask Priya.                  correct
turn 3  reply  : ...as [GIVENNAME_1] said earlier.
turn 3  shown  : ...as Priya said earlier.          WRONG: that was Alex
```

### Why it matters, and how much

Nothing leaks: the wrong value substituted is another value the same device
already holds, and the model still only ever saw placeholders. What breaks is
truthfulness on screen and in the spoken answer, in a conversation that
mentions two different people of the same entity type across turns. A
single-turn demonstration cannot hit it.

### Why it was not fixed

The fix is per-session placeholder identity, which means changing how
`merge_pii_spans` numbers spans, which changes the stored SQLite transcript for
any utterance with a repeated entity. That is explicitly out of scope for the
same reason the section below says it is.

The cheaper half - clearing the rehydration map between utterances - is worse
than the bug: it would break the legitimate case where a reply refers back to a
placeholder from the previous turn, which is most of what a conversation does.

---

## The wake word is masked, and the assistant works around it

**Where:** `app/post_mask_assistant.py` (`strip_masked_wake_word`).

**Status:** worked around in the additive layer, not fixed at the source.

The privacy model reads "Onkith" as a given name, which it reasonably is, so an
addressed utterance masks to `[GIVENNAME_1], remind me to call [GIVENNAME_2]
tomorrow.` Nothing leaks - the wake word is masked, not released - but two
things were wrong before the workaround: the question went to the model with a
placeholder for a vocative, and a reply that echoed it was rehydrated into the
device addressing the person by its own name ("I will message Onkith on Fahad").

`strip_masked_wake_word` removes exactly the placeholder whose span lies
entirely inside the wake word's own span in the raw transcript, and drops it
from the map handed to rehydration. The containment guard is what keeps it from
deleting real content when the masker merges the wake word into a longer span;
in that case it does nothing.

What it does **not** do is stop the wake word consuming `[GIVENNAME_1]` in the
canonical masked transcript, so the first real name in an addressed utterance is
numbered `_2`. Fixing that means either not masking the wake word - a change to
the release path - or renumbering after the fact, which would make the stored
transcript disagree with itself. Neither is worth it for a number nobody reads.

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
