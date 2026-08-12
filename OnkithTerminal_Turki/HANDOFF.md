# Handoff

Updated 2026-08-12. The demo is 2026-08-20 on a Raspberry Pi 5, which is
8 days out, and nothing in this project has ever run on that hardware.
This is written for whoever picks it up next, and the Pi section at the
end is written for Faris.

## Rules that are not up for discussion

1. **Do not push.** The full sequence is reviewed before anything leaves
   the machine. Commit freely, push nothing.
2. **Never list an AI as co-author, contributor or trailer** in a commit
   message.
3. **Do not add `torch`, or anything that pulls it in.** This is why
   Abdullah's recogniser is scored from a separate interpreter and why
   nothing in this repo imports it.
4. **Confirm the OS release and the audio stack on the device itself**
   rather than assuming what a Pi 5 runs.
5. Secrets live in `.env`, which is gitignored: `GROQ_API_KEY`,
   `GROQ_MODEL`, `YOUTUBE_API_KEY`, `YOUTUBE_REGION`. Names only, never
   values, anywhere.
6. Brand: no em dashes anywhere, including code comments. No eyes,
   padlocks, shields, gradients, glow, or a second hue. `--void #080706`,
   `--ink #F5F2EE`, `--lift #FF5A36`.

## The environment

The venv is uv-created. There is **no `pip`** and **no bare `python`**.

```bash
.venv/bin/python                                  # the interpreter
VIRTUAL_ENV="$PWD/.venv" uv pip install <pkg>     # how to install
.venv/bin/python -m unittest discover -s tests -q # the tests, 351 of them
```

There is no pytest. `unittest discover` is the runner.

## What was decided, and on what evidence

The device's problem was never general accuracy. It was that it did not
answer to its own name. The thing that separates the three recognisers is
whether they can be told a word exists.

Scored on 24 clips of Turki's real voice, 112 seconds, plus 24 synthetic
clips kept separate:

| recorded half | Moonshine 0.0.69 | Whisper base.en int8 | Canary 180M flash |
|---|---|---|---|
| Wake recall | 25% (3/12) | **100% (12/12)** | 83% (10/12) |
| False wake | **0%** | **0%** | **0%** |
| Intent | 25% | **100%** | 75% |
| Digits kept | 67% | **100%** | 0% |
| WER with the name | 39.5% | **2.9%** | 16.3% |
| WER without it | 14.2% | 9.7% | 2.2% on prose, 50.2% overall |

**Whisper base.en is the recogniser to ship.** Two reasons the other two
are out:

- **Moonshine** has no vocabulary hint of any kind. It wrote the name as
  Onceth, Anketh, Ankath, Oncus, Anke and Orcath. Its WER on ordinary
  speech is 14.2%, which is fine, so this is not a general accuracy
  problem and cannot be tuned away.
- **Canary** is the best prose recogniser measured by a wide margin, 2.2%
  WER against Whisper's 10%, and it writes every digit as a word:

  ```
  wanted : my number is 0501234567
  heard  : my number is zero five zero one two three four five six seven

  wanted : my id card number is 1098765432
  heard  : my id card number is one billion and ninety eight million ...
  ```

  That is `pnc: "no"` turning off digit formatting, and `pnc: "yes"` does
  not fix it.

### Why Canary was rejected, in full

It was not rejected for being inaccurate. It won the accuracy metric,
decisively. Four separate things ruled it out, and the first one alone
would have been enough.

**1. It cannot keep a digit, on the device whose claim is keeping
digits.** Split the recorded quiet clips by whether a number is in the
sentence:

| recorded quiet clips | n | mean WER |
|---|---|---|
| no number in it | 18 | **2.2%** |
| a phone number or an id | 6 | **201.9%** |

0 of 3 verbatim strings survived. This is not a formatting difference
that a regex fixes. `1098765432` came back as "one billion and ninety
eight million seven hundred sixty five thousand four hundred and thirty
two": the digit string was parsed as a **quantity**, so the original
digits are not recoverable from the output in any general way. The masker
downstream is looking for phone numbers and ID numbers. Feeding it prose
numerals defeats the entire product. FullPipeline_V3 knows this, which is
why `supplemental_pii.py` exists in that build to catch spelled-out digit
sequences, which means adopting Canary means adopting and validating that
reconstruction layer too, in the days left, on the feature the demo
is about.

**2. Its wake ceiling is lower and sits next to a cliff.** 83% against
Whisper's 100%. Worse than the gap is the shape of it. Canary's prompt
has five fixed slots (`source_lang`, `target_lang`, `task`, `pnc`,
`timestamp`) and no room for a name, so biasing has to go through
`TransformerAEDBeamInfer`'s boosting tree, which his build ships at alpha
0. Measured:

| alpha | recorded wake | what it writes |
|---|---|---|
| 0 (as shipped) | 17% | `uncle's play bonds`, `aunt kith`, `oncuth` |
| 1.5 to 2.0 | **83%** | correct on most |
| 3.0 | collapses | every clip returns `onk` |
| 6.0 | collapses | every clip returns `on` |

At 3.0 the decoder stops transcribing and emits the boosted phrase alone.
The usable band is narrow and the operating point sits right beside a
total failure mode. Whisper's `initial_prompt` reaches 100% and has no
cliff anywhere in it.

**3. It costs more than the target has.** `torch==2.7.1` plus
`nemo_toolkit[asr]==2.5.3` is 177 packages, pulling lightning, wandb,
datasets and bitsandbytes onto a 4 GB Pi 5. The weights are 700 MB of
fp32 with no quantised path. Whisper base.en is int8 through CTranslate2
and faster-whisper does not use torch at all. This also runs straight
into the standing rule against adding torch to this project.

**4. There is no evidence it runs on the target, and it is not a swap.**
Abdullah's own `PI5_VALIDATION_REPORT.md` says `PENDING - NOT RUN ON
PHYSICAL RASPBERRY PI 5 HARDWARE` and his model manifest carries
`physical_pi5_validation: pending_not_run`. The only timing that build
reports is RTF 0.22 on an x86 laptop, roughly three times what Whisper
base.en measures here. And its adapter answers
`transcribe_segment(CanarySegment) -> SegmentTranscript`, not the
`transcribe_recording` seam the runtime calls, so adopting it is an
integration project rather than a config change.

**How the comparison was kept fair.** Same pinned artifact, sha256
checked against his `models/canary/model_manifest.json` and refused if it
does not match. Same beam decoder, same single-turn prompt. Its boosting
tree was turned **on** for the reported row, because scoring it with its
biasing off against a Whisper with its hint on would not have been a
comparison. Its own shipped configuration scores 17% wake, not 83%.

**Where Canary would win.** If this device were a dictation or meeting
transcription tool with no PII requirement and mains power, Canary is the
right answer by a wide margin: 2.2% WER on prose is four times better
than anything else measured here. Keep `eval/canary_engine.py`. If the
product ever changes shape, the comparison is already built.

Full write-up with the reasoning behind each metric: `eval/README.md`.

## What Whisper is told before it listens

The single biggest code change. `integrated_pipeline/whisper_adapter.py`
builds this from the configured wake names, never hardcoded beside them:

> Onkith is the name of this device. It also answers to Sada, Sitr, Echo.
> Onkith, play Bones by Imagine Dragons. Onkith, stop the music. Onkith,
> show the player. Onkith, hide the player. Onkith, turn the volume up.
> Onkith, turn the light off. Onkith, set the light to 40 percent. My
> number is 0512345678. My id card number is 1234567890.

Naming the device alone was not enough. It got the name right and ran the
next word into it, so "Onkith play bones" came back as "Onkith Playbons":
wake word perfect, command gone, music request answered as conversation.
Showing whole commands took intent from 83% to 100%. Biasing toward short
command words costs a digit out of a long number, so the numbers on the
end buy it back, and the two sides have to be measured together every
time either grows. Adding volume and the lamp took the list to eight
examples and lost a digit out of an id number until the list came back
down to six, one per lane rather than one per wording. The comment on
DIGIT_SHAPE has the measurements, including the variant that says part of
why this works is the examples resembling the answer.

Measured and rejected: `condition_on_previous_text=False` changed nothing
on any clip. `small.en` reached 1.7% WER on the name and then 52% on the
quiet clips, which is hallucination, at three times the RTF.

## Commands

```bash
cd "$HOME/Documents/Onkith Website/onkith-terminal"

# run the app on the recogniser we are shipping
./run_live.sh --stt whisper

# confirm which recogniser a run will use before starting it
./run_live.sh --stt whisper --print-config

# pick a microphone if the default is wrong
./run_live.sh --list-devices
./run_live.sh --stt whisper --device 2

# score the two shipped engines, recorded and synthetic scored apart
.venv/bin/python eval/run_eval.py --verbose

# score Abdullah's Canary from its own interpreter
ONKITH_CANARY_BOOST=2.0 ~/.onkith-canary-eval/.venv/bin/python \
    eval/run_eval.py --engine canary

# the results as markdown, generated from the history file, not retyped
.venv/bin/python eval/report.py --split all

# record more corpus clips
.venv/bin/python eval/record.py --speaker faris

# a clickable icon, a menu entry and start on boot
./install_desktop.sh
./install_desktop.sh --status
./install_desktop.sh --remove
./install_desktop.sh --no-autostart   # the icon without the boot entry

# the tests, 351 of them
.venv/bin/python -m unittest discover -s tests -q
```

`--stt whisper` matters. Without it `./run_live.sh` runs Moonshine,
because `config/live_config.json` still says `"engine": "moonshine"`.

Everything the device answers to, by voice:

```
Onkith, play Bones by Imagine Dragons     Onkith, louder
Onkith, stop                              Onkith, quieter
Onkith, show the player                   Onkith, turn the light on
Onkith, hide the player                   Onkith, turn the light off
                                          Onkith, brighter
Anything else is a question.              Onkith, dimmer
                                          Onkith, set the light to 10 percent
```

Anything the device does not place goes to the assistant, which may
answer with the instruction instead of prose. That is a fallback, not the
route: a recognised command is carried out locally, instantly, offline,
and never sent anywhere. Only what the first word after the name did not
match reaches the model, which is where a phrasing nobody enumerated, a
mangled first word, or Arabic lands.

The model is given the intent to name and never its arguments, because it
reads masked text where "set the light to 10 percent" has already become
"[AGE_1] percent" and "play Adele" has become "play [GIVENNAME_1]". The
number and the title are re-read from the raw transcript on the device.

Three files have to agree on that list and a test walks every wording
the prompt offers through the lane. `app/onkith_music.py` recognises it,
`SYSTEM_PROMPT` in `app/llm.py` offers it out loud when somebody asks the
wrong way, and `COMMAND_SHAPES` in
`integrated_pipeline/whisper_adapter.py` demonstrates it to the
recogniser. A command the hint does not show is one the recogniser is not
biased toward, which is how "play bones" became "Playbons".

## Git state

Fourteen commits on top of `bf197b2`, none pushed:

```
9b6e856 fix: check_pi.sh has been failing since Whisper was offered
0e0afd8 docs: bring the handoff up to what the device now does
a38af14 feat: start Onkith when the Pi boots
a3c5694 feat: the lamp, on and off and a level, by voice
06dda5d fix: centre the playing record in the panel instead of clamping it away
625b9d6 feat: louder and quieter by voice
4f0a075 fix: the assistant thought the person was called Onkith
5b468ec fix: stop the device answering itself
19b4c5e docs: say why Canary was rejected, and what the Pi install misses
9a54331 docs: hand off the state of the device in one file
e87dc77 fix: --print-config showed the file, not the run
1989fec docs: rescore all three at one clean revision
0c73102 fix: the dirty flag was on for every run after the first
5cfefbb feat: score Abdullah's Canary on the same corpus as the other two
```

324 tests pass.

## What changed since the last handoff

Everything below was seen failing on screen or found by reading, and is
now fixed with a test that fails without the fix.

**The device answered itself.** Its own reply came back through the
microphone and was answered, and nothing stopped it: the loop ended
because the language model remarked that the two of them were repeating
each other. `SpeechEngine.speaking` was a live property nothing read.
Capture now reads it and drops chunks while the device talks, holds the
gate open past the last word because synthesis finishing is not the room
going quiet, and clears the pre-roll so the tail of a reply does not lead
the next utterance. Talking over a reply drops that recording rather than
finishing it, which costs barge-in the device never had.

**The assistant thought the person was called Onkith.** The whole masked
text went to the model, name and all. The name comes off first now, from
the masked text and never the raw, and only the occurrence that did the
summoning, so "what does Onkith mean" survives being asked.

**That last one does not work in production, and ISSUES.md #1 says
why.** The masker redacts the device's own name as a person's given
name, so by the time the strip looks for it there is a `[GIVENNAME_1]`
where the name was and nothing to remove. The code is right and it never
runs. Read ISSUES.md before trusting this section.

**Louder and quieter by voice.** The mechanism was all there and there
was no lane to reach it. Read before the transport, because "turn it
down" and "turn it off" share a verb.

**The lamp.** Faris's PWM flow, wired to voice. See below.

**The playing record was cut off in the panel.** `centerActiveCard` used
`offsetLeft`, which is relative to the offsetParent, and the rail is not
positioned, so in the side panel it carried the whole width of the
conversation beside it: it computed 1159 against a maximum scroll of 425,
the browser clamped it, and the rail sat jammed at its right stop.
Measured from the rail now. Checked in a browser at two widths.

**A clickable icon, and starting on boot.** `./install_desktop.sh`.

## The lamp

Faris's terminal script drove PWM0 on the Pi 5 and worked. The hardware
path is kept as he had it and the program around it is not: his was a
curses UI with Solid, Strobe, Breathing and Heartbeat modes, and what
this device needs is a level that voice can move. `app/light.py` opens
one hardware PWM channel at 1 kHz, writes brightness as a duty cycle and
stops the channel on the way out.

Say any of these:

```
Onkith, turn the light on          Onkith, brighter
Onkith, turn the light off         Onkith, dimmer
Onkith, lights on                  Onkith, dim the light
Onkith, close the light            Onkith, turn the light up
Onkith, open the light             Onkith, lower the lights
```

The lamp insists on being named. It shares "turn", "off" and "lower"
with both the music and the volume, so "turn it off" is still the music
and only "turn the light off" is the lamp. The two comparatives are the
exception, since nothing else here gets brighter or dimmer.

Off is remembered, so on returns to the level somebody chose rather than
to a default, and asking for brighter at a dark lamp turns it on.

**It needs a boot overlay, and `install_pi.sh` now writes it.** That
used to be a printed instruction, which was the wrong call: hand-editing
a boot config has no backup, no check for an overlay already there, and
nothing to notice that the end of `config.txt` is inside a `[pi4]`
section, which would scope the line to one model and fail silently on
every other.

`scripts/enable_pwm.sh` handles all three. It backs the file up and
prints where, it is idempotent, and it writes under an explicit `[all]`
header. It refuses rather than guesses when a different pwm overlay is
already set, including `pwm-2chan` and `pwm-gpio`, which are the same
hardware under other names. It can be run on its own:

```bash
./scripts/enable_pwm.sh --check     # say what the state is
./scripts/enable_pwm.sh             # add it, for GPIO12
./scripts/enable_pwm.sh --pin 18    # a different pin
./scripts/enable_pwm.sh --remove    # take it back out
sudo reboot                         # the one part no script decides
```

`./install_pi.sh --no-pwm` skips it, for a board with no lamp wired or
whose boot config belongs to somebody else.

After the reboot, `ls /sys/class/pwm/`. If the chip is not `pwmchip0`,
set `light.chip` in `config/live_config.json` to match. Faris confirmed
`pwmchip0` for GPIO12 on his board and the kernel is free to number it
otherwise, which is why it is configurable rather than assumed.

Without any of that the lamp does nothing and says why, once. Everything
else on the device works regardless: a lamp that cannot be driven never
reaches back into the voice path, because it is on the same thread as a
transcript and it is the least important thing here.

## Open issues

The running list, with what has been diagnosed on each, is in
[ISSUES.md](ISSUES.md). Read that one before picking anything up: the
first entry there undoes a fix this document describes as done.

Ranked by what would embarrass the demo.

### 1. Music requests can still reach the assistant

Seen live: "Onkith, they found to see earth wind" was answered as a
question about a discovery. `_lane_for` in `app/onkith_music.py` reads
only the first word after the name, which is deliberate, since scanning
the whole sentence would send "Onkith what time do the shops close" to
the transport. A mangled first word falls through to conversation.
Whisper's hint has made this much rarer and not impossible. Offered
three times, never taken, and it is still the right call to leave it
until somebody decides otherwise.

Downstream of the same thing: the player showed "The Marías, Sienna" for
a request that was not that, which is a bad search from a mangled query.

### 2. The `ankith` wake entry

`quiet-near-miss-1` is "someone named Ankith called me this morning", and
it wakes the device. `ankith` is in the wake spellings to catch Moonshine
writing the name badly, which it needs. Whisper with the hint never
writes `Ankith` when somebody is talking to the device. Measured on all
48 clips: removing it takes Whisper's false wake from 8% to 0% with wake
recall unchanged at 100%, and costs Moonshine 25% to 17%.

So it is not a wake-list question, it is a consequence of which
recogniser is running. **Remove it in the same commit that flips
`config/live_config.json` to `whisper`, not before.**

### 3. Groq rate limiting

The primary model `groq/compound-mini` saturates the whole account's TPM.
A 429 fallback is configured. Worth a look before a demo where several
people may talk to it.

### 4. Orphaned mpv processes

They survive some exits. The temp directories they left behind are swept
at startup now, but the processes themselves are not, and on a device
that is power-cycled this is worth a look before the demo.

### Decided, on 2026-08-12, by the Pi

The rule was that the default did not move until base.en had been timed
on the target. The Pi 5 measured it:

| half | Mac | Pi 5 | headroom |
|---|---|---|---|
| recorded | 0.17x | **0.45x** | 2.2x faster than real time |
| synthetic | 0.32x | 0.81x | 1.2x faster than real time |

Whisper costs roughly three times Moonshine's compute and keeps up on
the device with room to spare, so `config/live_config.json` now says
`"engine": "whisper"` and a bare `./run_live.sh` runs it.

`ankith` moved to `never` in the same change, because the two were
coupled: Moonshine needed that spelling to hear its own name and pays
25% to 17% wake recall for losing it, while Whisper with the hint never
writes Ankith when somebody is talking to the device. Deleting it from
the spellings was not enough on its own. It is six letters and one edit
from `onkith`, so the fuzzy matcher caught it anyway and the false wake
stayed at 8%. In `never`, which is checked first, it goes to 0%.

Whisper is now clean on every metric on both halves: wake 100%, false
wake 0%, intent 100%, digits 100%.

## For Faris: bringing this up on the Pi

None of this has run on the target. Everything below is untested on the
hardware it has to work on, which is the whole reason it is going to you.

The installer now does the two things it used to miss. It installs
`faster-whisper`, which is the recogniser this device ships on and which
`requirements-rpi.txt` never carried, and it pulls `base.en` down during
install rather than on first use, because it comes from HuggingFace and a
venue is the wrong place to discover that.

```bash
# 1. everything. Python 3.11 exactly, 2 GiB free disk, sudo.
#    apt packages, both recognisers, base.en cached, yt-dlp, piper.
#    Ends by reporting whether hardware PWM is available.
./install_pi.sh

# 2. confirm what the board actually is, rather than assuming
./run_live.sh --device-report

# 3. readiness check, then again with the microphone
./check_pi.sh
./run_live.sh --list-devices
./check_pi.sh --with-mic --device N

# 4. the number everything is waiting on
.venv/bin/python eval/run_eval.py --engine whisper

# 5. talk to it
./run_live.sh --stt whisper --device N

# 6. the clickable icon and start on boot, once the rest works
./install_desktop.sh
```

Step 6 puts an Onkith icon on the desktop and in the applications menu,
and makes the device come up on its own after a power cut. Clicking the
icon keeps a terminal so you can see what it is doing. The boot entry
does not, so it waits for the audio server and writes to
`~/.local/state/onkith/onkith.log`, keeping the previous run as `.log.1`.
`./install_desktop.sh --remove` takes all three back out, and none of it
needed a password to install.

**Step 4 is the decision.** RTF for `base.en` on the Pi. On a Mac it is
0.07x against Moonshine's 0.06x, which says nothing about a 4 GB Pi 5.
The config stays on `moonshine` until that number exists, even though
every accuracy measurement says Whisper. Copy `eval/corpus/` across so it
is the same clips. Expect wake and intent to reproduce and RTF not to.

### What to try, and what would be a real finding

Worth saying out loud, because most of these are fixes that have never
been tested against a room:

- **Does it still answer itself?** Ask it something, let it reply, and
  say nothing. On a Pi with the speaker in the same enclosure this is a
  harder test than any laptop gives it. If it answers its own reply, the
  gate is too short: raise `audio.speech_tail_seconds` in
  `config/live_config.json` from 0.35 and say so.
- **Does it call you Onkith?** It should not, any more.
- **Does the lamp respond**, and does the level survive off and on.
- **Louder and quieter**, and confirm "turn it off" still stops the music
  rather than dimming anything.
- **The playing record**, centred in the panel on the Pi's actual screen
  size. It was fixed against a browser at two widths, not against that
  display.
- **Boot the device cold** with autostart installed and see whether it
  comes up unattended. The log is at
  `~/.local/state/onkith/onkith.log`, and the previous run is kept as
  `.log.1`, which is usually the one worth reading.

### Things that are only true on the device

- **Piper needs a voice file, not just the binary.** `install_pi.sh`
  installs `piper-tts` best-effort, because its wheels are not always
  published for aarch64. `build_piper_backend` (`app/speech.py:217`)
  returns `None` unless it finds both a `piper` binary **and**
  `ONKITH_PIPER_MODEL` pointing at a voice. Missing either, the chain
  falls silently through `piper, kokoro, espeak, say`. Nobody gets an
  error, the device just sounds wrong.
- **The lamp needs a reboot after the install.** `install_pi.sh` writes
  the overlay, and an overlay only takes effect on boot, so the lamp
  does nothing until then. The installer says so at the end when it has
  added it.
- **The audio stack.** `--device-report` prints the board, OS release,
  kernel, the detected stack, chromium, mpv and yt-dlp. Read it rather
  than assuming ALSA or PulseAudio. It does not report the TTS backend or
  the recogniser, so check those separately.
- **`check_pi.sh` still verifies Moonshine only.** It has not been
  taught about Whisper. A pass is not proof the shipping recogniser
  loaded; step 4 is. It was also failing outright on its second step,
  since the day Whisper was offered, because it grepped every runtime
  file for the word torch and the Whisper adapter's docstring says that
  it does not use torch. Fixed, and covered by the suite now, so it
  fails on a laptop rather than on the Pi next time.
- **Chromium and the display.** The player surface needs a browser and a
  display, or `--headless` and a second machine to view it from.
- **The network.** Groq for the assistant and YouTube for playback. Both
  need internet at demo time. Decide now what the device does on the
  venue's wifi, or without it.
- **Thermals and power.** A 4 GB Pi 5 running a recogniser, a browser and
  mpv together will throttle on a bad power supply. Watch it during a
  long session rather than a single utterance.

## Later, not before the demo

openWakeWord as a two-stage architecture: a tiny always-on wake detector
in front of the recogniser, so the expensive model only runs on speech
that was addressed to the device. This is the correct long-term shape and
it is not an eight-day change.

Acoustic echo cancellation, which would give back the barge-in that the
self-hearing gate costs. Today talking over a reply drops that recording.
That is the right trade while the alternative is a device that answers
itself, and it is not the right trade forever.

The lamp's other modes. Faris's script had Strobe, Breathing and
Heartbeat and they all worked; they were left out because voice has no
use for them yet. The hardware path here is the same one, so adding them
back is a waveform on a thread, not a rewrite.
