# Recogniser evaluation

A fixed corpus and a fixed score, so two recognisers can be compared on
the same evidence and the comparison still means something in a month.

```bash
.venv/bin/python eval/run_eval.py                   # the two shipped engines
.venv/bin/python eval/run_eval.py --engine whisper  # one
.venv/bin/python eval/run_eval.py --verbose         # every miss, with what was heard
.venv/bin/python eval/run_eval.py --no-record       # do not append to history

# Canary needs torch and NeMo, which this project does not install, so it
# is asked for by name and scored from its own interpreter.
ONKITH_CANARY_BOOST=2.0 ~/.onkith-canary-eval/.venv/bin/python \
    eval/run_eval.py --engine canary

.venv/bin/python eval/report.py                     # latest scores as markdown
.venv/bin/python eval/report.py --split all         # recorded, synthetic and both
.venv/bin/python eval/report.py --history wake_recall
```

Every run appends a line to `results/history.jsonl` with the git revision,
so a change that helps one number and quietly ruins another is visible
rather than argued about. `report.py` reads that file back and prints the
table, so the numbers in a document and the numbers on the machine cannot
drift apart by retyping.

## The two halves are scored apart

`run_eval.py` prints recorded first, synthetic second, and both together
last. They are not averaged into one row, because that is how a
recogniser that cannot hear a person in a room gets carried by 24 clips
of clean machine speech.

The split has already caught a wrong conclusion. On synthetic audio
Moonshine kept 100% of the digits and Whisper kept 67%, which was written
down as Whisper being the one that invents digits. On real voice it
reverses exactly: Whisper 100%, Moonshine 67%. Neither number was wrong,
and the one-line average of the two says nothing.

## What is scored, and why these six

Word error rate alone would have said Moonshine was fine. Measured on
ordinary speech it scores about 10%, which is respectable, and the device
still never once answered to its own name. Each of these exists because it
caught a failure the others hid.

| | |
|---|---|
| **wake** | Of the clips that called it, how many woke it. The one that matters most. |
| **false** | Of the clips that did not call it, how many woke it anyway. Kept separate because either number can be driven to perfect by ruining the other. |
| **intent** | Whether the words routed to the lane they asked for. Waking up and then playing music at somebody who asked a question is still a failure. |
| **digits** | Whether every digit of a phone number or an ID survived. A recogniser that invents digits is working against a masker whose whole job is catching them. Separators are ignored; an inserted or dropped digit is not. |
| **WER/w, WER/q** | Transcription quality, split by whether the clip contains the name. The split is the point: one recogniser scored 1.9% on ordinary speech and 43% on the same voice the moment the name appeared. |
| **RTF** | Seconds of compute per second of audio. Meaningless off the target machine. Measure it on the Pi. |

## The corpus

`corpus/manifest.json` is the source of truth. Each clip carries its
transcript, whether it should wake the device, which lane it should reach,
and any string that must survive verbatim.

24 synthetic clips, built by `corpus/build_synthetic.py` from a fixed
script with macOS `say`. They are reproducible and cheap, and they are a
**floor on difficulty**: one voice, one accent, no room, no noise. Nothing
is decided on them alone.

The recorded half is the real evidence:

```bash
.venv/bin/python eval/record.py --speaker turki
.venv/bin/python eval/record.py --speaker faris --only wake
```

It reads the same script back to you a line at a time, records until you
stop talking, and lets you keep or redo each one. Recorded clips are never
touched by `build_synthetic.py`, so the synthetic half can be regenerated
without losing them.

**A result with zero recorded clips is a smoke test, not a decision.**

## What the recorded half says today

24 clips, one speaker, 112 seconds. Measured 2026-08-08 at `0c73102`, all
three at the same revision with a clean tree. `report.py` prints this table
from the history file, so it can be regenerated rather than trusted.

| | Moonshine 0.0.69 | Whisper base.en int8 | Canary 180M flash |
|---|---|---|---|
| Wake recall | 25% (3/12) | **100% (12/12)** | 83% (10/12) |
| False wake | **0%** | 8% (1/12) | **0%** |
| Intent | 25% | **100%** | 75% |
| Digits kept | 67% | **100%** | 0% |
| WER with the name | 39.5% | **5.0%** | 16.3% |
| WER without it | 14.2% | 10.5% | **2.2%*** |

\* Canary on prose with no number in it. Its published figure over all the
quiet clips is 50.2%, and the gap between those two numbers is the whole
story. See below.

Moonshine heard the name as Onceth, Anketh, Ankath, Oncus, Anke and
Orcath. It is not mishearing the sentence: WER without the name is 14.2%,
which is fine. It has no way to be told the word exists. Whisper's
`initial_prompt` is that way, and it is the whole difference.

Whisper's single false wake is the `ankith` entry described at the bottom
of this file. It is a correct transcription of a sentence about a person
called Ankith, so it is a routing fault and not a recogniser one. Every
other clip in the corpus, both halves, now passes.

### What the hint has to contain

Naming the device was not enough on its own. It got the name right and
then ran the next word into it, so "Onkith play bones" came back as
"Onkith Playbons": the wake word perfect, the command gone, and a request
for music answered as conversation. Showing whole commands in the hint
instead of just the name is worth a great deal:

| recorded half | name only | + commands | + a number |
|---|---|---|---|
| Intent | 83% | **100%** | **100%** |
| Digits kept | **100%** | 67% | **100%** |
| WER with the name | 20.7% | **2.9%** | 5.0% |

The middle column is why the digit example is there. Biasing toward four
short command words cost a digit out of a phone number, `0501234567` came
back as `050123457`, and one full-length number in the hint puts it back.
It costs 2.1 points of WER to buy back the digit, which for this device is
not a trade at all.

Two things measured and not adopted. `condition_on_previous_text=False`
changed nothing on any clip. `small.en` reached 1.7% WER on the name and
then scored 52% on the quiet clips, which is hallucination, and its RTF is
three times base.en before the Pi is even involved.

## Canary, and the one number that decides it

Canary 180M flash is the recogniser in FullPipeline_V3. Scored here on the
pinned artifact, sha256 checked against that build's
`models/canary/model_manifest.json`, with its beam decoder and prompt.

**It is the best prose recogniser of the three by a wide margin and it
cannot be used, and both of those come from the same measurement.**

| recorded quiet clips | n | mean WER |
|---|---|---|
| no number in the sentence | 18 | **2.2%** |
| a phone number or an id | 6 | **201.9%** |

Whisper scores about 10% on the first group. Canary is four times better
at ordinary speech than anything else measured here. Then:

```
wanted : my number is 0501234567
heard  : my number is zero five zero one two three four five six seven

wanted : my id card number is 1098765432
heard  : my id card number is one billion and ninety eight million
         seven hundred sixty five thousand four hundred and thirty two
```

That is `pnc: "no"` in the prompt, which turns off punctuation and with it
digit formatting. It was the first thing tested and `pnc: "yes"` does not
fix it: the numbers still come back as words. FullPipeline_V3 compensates
downstream, and `supplemental_pii.py` there exists to catch spelled-out
digit sequences, which is direct confirmation that this behaviour is known.

It is still the wrong recogniser for this device. Every digit is a word to
be reassembled before the masker can see it, on the one device whose entire
claim is that it catches phone numbers. The second example is worse than a
formatting difference: the digit string has been read as a quantity.

### Canary can be told the name, and the switch has a cliff

Canary's prompt slots are fixed and there is nowhere in them to put a name.
Its biasing lives somewhere else: `TransformerAEDBeamInfer`, the exact
decoder that build selects, accepts a boosting tree of key phrases, and the
build leaves it at alpha 0.

Turning it on is worth a great deal, and then it falls off a cliff:

| boosting alpha | recorded wake recall | what it writes |
|---|---|---|
| 0 (as shipped) | 17% | `uncle's play bonds`, `aunt kith`, `oncuth` |
| 1.0 | some fixed | `onkith stopped the music` |
| **1.5 to 2.0** | **83%** | correct on most |
| 3.0 | collapses | every clip returns `onk` |
| 6.0 | collapses | every clip returns `on` |

At 3.0 the decoder stops transcribing and emits the boosted phrase alone.
The usable band is narrow and 83% is the best it reaches, against 100% for
a Whisper prompt that has no cliff in it at all.

The 83% row is the one in the table above, because scoring Canary with its
biasing off against a Whisper with its hint on would not be a comparison.

### What it costs to run

`torch==2.7.1` and `nemo_toolkit[asr]==2.5.3`, which is 177 packages and
pulls lightning, wandb, datasets and bitsandbytes onto a 4 GB Pi 5. The
model is 700 MB of fp32 weights with no quantised path. That is why it is
scored from `~/.onkith-canary-eval` and why nothing in this repo imports it.

Its own `PI5_VALIDATION_REPORT.md` says `PENDING - NOT RUN ON PHYSICAL
RASPBERRY PI 5 HARDWARE`, and its model manifest carries
`physical_pi5_validation: pending_not_run`. The only timing that build
reports is RTF 0.22 on an x86 laptop, which is roughly three times what
Whisper base.en measures here.

RTF here is on a Mac and means nothing for the device. It has to be
measured on the Pi before Whisper can be chosen on those grounds.

## Adding an engine

Anything with `load()`, `transcribe_recording(recording, partial) -> STTResult`
and `close()` can be scored. Register it in `engines_for()` in
`run_eval.py`. That is the same seam the runtime uses, so whatever wins
here is what ships.

An engine that does not take a vocabulary hint starts the wake column at
whatever the name happens to sound like to it, which the table above
puts at 25%. Ask for the hint before asking for accuracy.

## Known trade-off this corpus found

`quiet-near-miss-1` is "someone named Ankith called me this morning".
Whisper writes the name Ankit with an h, and `ankith` is in the wake
spellings while only `ankit` is in `never`. So today a person called Ankit
wakes the device.

Blocking `ankith` fixes that and costs a real mishearing: "Ankith play
bones" then stops working. Measured both ways, neither was free, because
one string means two things.

**This is now free, but only on Whisper, and that is the whole point.**
Measured on all 48 clips:

| moving `ankith` to `never` | wake recall | false wake |
|---|---|---|
| Whisper with the hint | 100%, unchanged | 8% to **0%** |
| Moonshine | 25% to **17%** | 0%, unchanged |

`ankith` is in the wake list to catch a recogniser that cannot write the
name. Whisper with the hint never writes `Ankith` when somebody is talking
to the device, so the entry only ever fires on the person. Moonshine
writes it constantly and genuinely needs it.

So the entry is not a wake-list question. It is a consequence of which
recogniser is running, and it should be removed in the same change that
makes Whisper the default, not before. It is still in the list today
because the config still says `moonshine`, and that stays true until
base.en has been timed on the Pi.
