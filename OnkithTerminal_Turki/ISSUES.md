# Running issues

Things seen and diagnosed, not yet fixed. Each one says what was
observed, what the cause turned out to be, and what fixing it would
involve, so picking one up does not mean starting the investigation
again.

`HANDOFF.md` is the state of the device and the decisions behind it.
This is the list of what is still wrong with it.

---

## 0. The Pi cannot speak

**Seen** in the first report back from the device, 2026-08-12:

```
  piper                -
  piper-tts            -
  espeak-ng            -
```

None of the three. `detect_backend` in `app/speech.py` walks piper,
kokoro, espeak, say and falls off the end to `NullSpeechBackend`, which
logs one line and returns. So the device hears, masks, routes, answers
and shows the answer on screen, and says nothing out loud. Nothing
crashes, which is why the report is otherwise all green.

**Cause.** `install_pi.sh` installs piper best-effort, because its wheels
are not always published for aarch64, and falls back to
`apt-get install -y espeak-ng || true`. Both failed and the `|| true`
swallowed the second one, so the install finished reporting success.

**Fix, in order of how quickly it gets a voice back:**

```bash
sudo apt-get install -y espeak-ng      # and read the error if it fails
```

That alone makes the device speak, badly. For the real voice, piper needs
both its binary and a voice file, and `ONKITH_PIPER_MODEL` pointing at
the voice, or `build_piper_backend` returns None and the chain falls
through again with no error.

The installer should stop hiding this. `|| true` on the last fallback in
a chain means the one case where nobody gets a voice is the one case that
reports success.

---

## 1. The masker redacts the device's own name

**Seen** in a live session on 2026-08-12: every utterance in the
transcript reads `[GIVENNAME_1]` where the person said Onkith.

```
You    on. [GIVENNAME_1], turn the light off.
You    [GIVENNAME_1] also has a way to go and place to go.
You    turn the volume up. [GIVENNAME_1], turn the volume up.
```

**Cause**, reproducible in one line:

```python
FailClosedMasker(PIIMaskerAdapter(PipelineConfig())).mask_text("Onkith, turn the light off")
# -> '[GIVENNAME_1], turn the light off'
```

TinyBERT reads the device's name as a person's given name and redacts
it. Every name in the wake list does it: Onkith, Sada, Sitr, Echo.

**Why it matters more than it looks.** It silently undoes the fix in
`4f0a075`. `without_wake_word()` runs on the masked text, by design, so
the raw transcript never reaches a network model. By the time it looks
for the name, the name is a placeholder, so it finds nothing and strips
nothing, and the assistant is handed `[GIVENNAME_1], turn the light off`.
The reply is rehydrated on the way out, so a model that answers "users
like you, [GIVENNAME_1]" says "users like you, Onkith" out loud. That is
the exact behaviour that fix was written to stop, arriving by a
different route. **Treat that fix as not working in production.**

It also burns a placeholder slot on every utterance and puts the
device's name in the database as though it were somebody's personal
detail.

Waking up is unaffected: `intercept_onkith_command` reads
`result.pii.original_text`, which still has the real word in it.

**Independently confirmed from the other pipeline.** FullPipeline_V3's
`app/wake_word_normalizer.py` hit the same wall and says so in its
docstring: *"once the masker has turned it into 'aunt [GIVENNAME_1]' the
wake word is gone for good"*. They solved the ordering by normalising
before masking.

**Fix direction.** The device's name is not private data, and the masker
should know its own name. A never-mask list seeded from `wake.names` and
`wake.spellings` is the honest shape, next to the existing
`pii.never_mask` which works on labels rather than strings. Doing it by
reordering instead, stripping the name before masking, would mean
touching text on the raw side of the privacy boundary, which is not a
trade worth making for this.

---

## 2. The recogniser could not keep up

**Seen** in the same session, four times in one short conversation:

```
Completed-recording STT backlog reached its configured bound;
1 recording(s) retained in lossless spill memory
```

That is `completed_recording_queue_max: 4` filling up: utterances
finished faster than Whisper transcribed them. Nothing is lost, the
spill is lossless, but the device is running behind the person talking
and replies arrive late.

**Context worth keeping.** That session had a Zoom call running for 32
minutes on the same machine and the battery was at 4%, so the CPU was
being shared and very likely throttled. This may be entirely explained
by that and not be a property of the device at all.

**Mostly settled now.** The Pi has since measured base.en at 0.45x on
the recorded half, 2.2 times faster than real time, and 0.81x worst case
on the short synthetic clips. It keeps up on the device with room to
spare, so the backlog on that laptop was the video call, not the
recogniser.

What is left of this: 0.81x is the worst case on a corpus of clean
clips, and a command is short, so the fixed cost per utterance is what
bites. Worth watching the same warning during a long live session on the
Pi. If it appears there, with nothing else running, that is real.

---

## 3. Transcription in that session was poor

Fragments and mishearings throughout: `Gonna smoke sure.`, `Our names as
the oh yeah but. turn the volume up. Oh.`, and `Oh, it's in a lamp.`
where somebody almost certainly said turn on the lamp.

Same caveat as above, and probably the same cause: a machine sharing its
CPU with a video call. Worth reproducing on an idle machine before
reading anything into it. If it survives that, the corpus is the way to
chase it, not another live session.

---

## 4. The Pi's microphone is quiet enough to miss speech

Not seen yet, found by putting Faris's measurement next to our
thresholds.

He measured the Pi's mic, a `UACDemoV1.0: USB Audio`, with `sox
stat`: speech RMS 0.0123, peak 0.10, roughly ten times of headroom
before clipping. Our trigger threshold is
`max(minimum_start_rms, room_noise * noise_multiplier)`, which is
`max(0.006, noise * 4)`:

| room noise RMS | threshold | speech margin | |
|---|---|---|---|
| 0.0010 | 0.0060 | 2.05x | fine |
| 0.0020 | 0.0080 | 1.54x | tight |
| 0.0030 | 0.0120 | 1.03x | **will not trigger** |
| 0.0040 | 0.0160 | 0.77x | **will not trigger** |

A quiet room is fine. A room with people in it, which is what a demo is,
may not be. The failure is the bad one: the device sits there saying
"listening" and never hears anybody, which is precisely what the
`maximum_start_rms` ceiling was added to prevent in the noisy direction.

The device reports it at index **0**, `hw:0,0`, 48 kHz, mono. Faris's
earlier note said `hw:2,0`, so the number moves: read it from
`./run_live.sh --list-devices` rather than from any document.

**Cheapest fix first, and it is not in this repo:** turn the mic gain up
with `alsamixer`, pick the USB card with F6, and re-measure. Ten times of
headroom means there is plenty to take. Only if that cannot be done
should `minimum_start_rms` come down, because lowering it trades against
false triggers.

**Still not measured.** The microphone section of the first two reports
never ran: `--device N` was typed with the letter N, the report went on
to fail on it, and the summary said `ok` anyway because of a bug in the
script. Both are fixed. Somebody needs to run
`./scripts/pi_report.sh --device 0` in the actual demo room and send the
calibration numbers.

---

## 5. Music requests can still reach the assistant

Carried over from `HANDOFF.md`, unchanged. `_lane_for` reads only the
first word after the name, deliberately, so a mangled first word falls
through to conversation. Intent measures 100% on all 48 corpus clips
with Whisper, so there is no measurement showing a change would help.

---

## 6. Orphaned mpv processes

They survive some exits. The temp directories are swept at startup now
(`c45702d`); the processes are not.
