# Onkith terminal

This folder contains the standalone Onkith terminal product. It does not import, build, start, or modify the website in `../onkith-web`.

The runtime records short utterances, transcribes them locally with Moonshine, masks private text with a local DeBERTa classifier and structured recognizers, and renders the result through the original Onkith ANSI terminal UI.

```text
  You      Please call [GIVENNAME_1] at [TELEPHONENUM_1]

  Onkith   Opening Dancing with Monsters
```

## Install

Python 3.11 is required. If `uv` is installed, the installer can fetch and manage the correct Python version automatically. On macOS, install PortAudio first:

```sh
brew install portaudio
```

Then install Onkith from this folder:

```sh
cd onkith-terminal
./install.sh
```

The installer creates `onkith-terminal/.venv`, installs the pinned runtime dependencies, registers the `onkith` command in that environment, and runs the built-in self-test.

## Run

List microphone inputs:

```sh
./onkith --list-devices
```

Start the terminal UI with the default microphone:

```sh
./onkith
```

From the repository root, the complete command is:

```sh
cd onkith-terminal && ./onkith
```

Or choose a microphone by its listed ID:

```sh
./onkith --device 1
```

After it reports that it is listening, speak commands such as:

```text
Onkith open Imagine Dragons Bones
Onkith play Adele Easy on Me
Onkith stop the music
Onkith close it
```

Press `Ctrl+C` once for a drain-first shutdown.

## Voice

Kokoro is the voice the device ships with. It needs two downloaded files,
which `install_pi.sh` fetches into `~/onkith-voices/`, and two lines in
`.env` pointing at them:

```text
ONKITH_KOKORO_MODEL=/home/pi/onkith-voices/kokoro-v1.0.onnx
ONKITH_KOKORO_VOICES=/home/pi/onkith-voices/voices-v1.0.bin
```

Only the ONNX build is allowed. Plain `kokoro` pulls torch, which this
deployment forbids and `check_pi.sh` rejects.

**It is not free.** On an M-series MacBook, Kokoro holds about 600 MB
resident on top of everything else and takes one to three seconds a
sentence. With the recogniser and the classifier alongside it the process
sits at roughly 1.2 GB. None of that has been measured on the Pi. If the
board cannot hold it, or answers too slowly to use, one line in `.env`
moves to a lighter voice and nothing else changes:

```text
ONKITH_TTS=piper
```

The order without an explicit choice is kokoro, piper, espeak, say — first
one installed wins.

## Privacy classifier

Two are packaged. **Model V2** (DeBERTa-v3-xsmall INT8, 63 labels) is what
runs. **Model V1** (TinyBERT-4 INT8, 67 labels) is the one it replaced, kept
loadable so the two can be measured against each other and so a bad result on
the Pi is a config key rather than a redeploy.

Switch with `pii.engine` in `config/live_config.json`, or without editing
anything:

```sh
ONKITH_PII_ENGINE=tinybert .venv/bin/python -m app.omit_ai_live
```

Measure them against each other on the speech-like corpus:

```sh
.venv/bin/python eval/compare_classifiers.py
```

The findings are in [eval/results/CLASSIFIER_COMPARISON.md](eval/results/CLASSIFIER_COMPARISON.md).
In short: Model V2 leaks a third as much private text and raises a quarter as
many false alarms, for 5.7x the disk and 2.6x the memory. Nothing has been
measured on the Pi 5 yet, which is the open question.

## Session report

`Ctrl+C` prints a session report and then a hardware usage report, after
the live screen has released the terminal. The session report covers what
the device heard, latency at p50/p95/max for every stage, backlog
high-water marks, what was masked, and how the assistant did. Add
`--report-file run.json` to keep the same numbers as JSON.

It deliberately contains no accuracy. Word error rate, wake recall and
intent accuracy all need ground truth, and a live session has none. Those
come from the corpus:

```sh
.venv/bin/python eval/run_eval.py --engine whisper --no-record
```

Two lines in it are worth reading first on new hardware. **loudest /
threshold** is the peak input level divided by the speech threshold the
calibration chose; under 2x on a session where people did speak means the
microphone barely clears its own gate. **near misses** lists how the
recogniser actually spelled the device's name when it nearly matched, and
those spellings go straight into `wake.spellings` in
`config/live_config.json`.

If you prefer an activated environment, run:

```sh
source .venv/bin/activate
onkith --device 1
```

## Music setup

Music search is optional. Create `onkith-terminal/.env` with your own YouTube API key:

```text
YOUTUBE_API_KEY=your_key_here
YOUTUBE_REGION=SA
```

Only a selected music query is sent to YouTube. The microphone, speech recognition, privacy masking, and transcript storage stay local.

Audio does not play in a browser. The core process plays music in a headless `mpv` child driven over a JSON IPC socket, with `yt-dlp` resolving the stream and only the audio track requested, so the Pi never decodes video for a track nobody is watching. Both are installed by `install_pi.sh`. `yt-dlp` is deliberately unpinned because it has to track YouTube extraction changes; refresh it with:

```sh
.venv/bin/python -m pip install --upgrade yt-dlp
```

Because the core owns audio rather than the page, the screen can be closed or killed without interrupting playback, music can play with no UI at all, and the core can duck music while Onkith speaks.

Onkith requests up to 25 music results, verifies that each result is public, processed, embeddable, and available in the configured region, then keeps the first five verified results. The five result cards retain YouTube's verified titles, channels, and thumbnails. Selecting a card changes the active song. The tab title follows `Music name \ Player \ Onkith`.

## Screen and headless operation

The core serves the player on localhost and publishes its state over a websocket. Surfaces subscribe to that feed; none of them owns audio, and none is required for audio to work.

With a display attached, Chromium is started as the display surface only, with `--kiosk --app` against localhost: no tabs, no address bar, and no visible pointer. It is being used as a GUI toolkit, not as a browser.

Run the whole stack with no display and no Chromium:

```sh
./onkith --headless
```

Headless is also selected automatically when no display server is present, so a Pi with nothing plugged in still plays and still answers commands. `--port` pins the surface port for a systemd unit.

Confirm what a device actually is, rather than assuming it:

```sh
./onkith --device-report
```

That prints the OS release, kernel, board model, sound server, default sink, output monitor, and whether `mpv`, `yt-dlp`, and Chromium are present. Run it on the Pi before a demo.

## Reviewing the player

To look at the screen without starting the microphone or loading any model:

```sh
./onkith --surface-only --port 8787
```

On a machine with a display that opens the kiosk window. Headless, or over SSH, it prints the URL to open from another machine. Add a search to give the player real content to show:

```sh
./onkith --surface-only --port 8787 --play "imagine dragons bones"
```

That needs `YOUTUBE_API_KEY`; without it the surface still loads and reports that playback is unavailable.

## Assistant

Masked speech is answered by Groq. Configure it in `onkith-terminal/.env`:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
```

The model is whatever Groq currently serves; change `GROQ_MODEL` without touching code. With no key set, Onkith transcribes and masks as before and sends nothing anywhere. `--no-assistant` disables it even when a key is present.

**Check the name against the live catalogue before a demo.** Groq withdraws models, and a withdrawn name fails as HTTP 404, which on the screen reads like the device losing its network. Every Llama model is gone, including both names this repo used to ship:

```bash
curl -sH "Authorization: Bearer $GROQ_API_KEY" \
     -H "User-Agent: Onkith/1.0" \
     https://api.groq.com/openai/v1/models | grep '"id"'
```

The `User-Agent` is required. Without one Cloudflare answers 403 `error code: 1010` and the request never reaches Groq.

On a rate limit or a withdrawn name the device drops to `GROQ_FALLBACK_MODEL` and sends the next minute of questions straight there, rather than spending a round trip rediscovering the problem. `groq/compound` and `groq/compound-mini` can search the web on the server, which costs around 3400 tokens a question against a free tier of 8000 a minute: two questions before the device starts refusing people, which is why they are not the default.

The `gpt-oss` models reason before answering and the reasoning is billed against `max_tokens`, so the client sends them `reasoning_effort: low`. Without it the whole 160-token budget goes to reasoning and the reply comes back empty.

The path is:

```text
mic -> VAD -> STT (local) -> masker (local, keeps the span map)
    -> Groq -> rehydrate (local, from that map) -> TTS -> speaker
```

**Only masked text leaves the device.** The masker runs first and fails closed. An utterance it could not clear is displayed as blocked and dropped, never sent. What Groq receives has private values already replaced by placeholders such as `[GIVENNAME_1]`, and the system prompt tells it to treat those as values it simply cannot see rather than guessing at them.

**The reply is rehydrated locally before it is spoken.** The masker records what it replaced; that map stays on the device and is never sent anywhere. When the reply comes back containing `[GIVENNAME_1]`, the original value is put back just before speech, so the room hears the real name while the model never saw it. The map rolls across turns, because a reply can refer to a placeholder introduced several turns earlier. A placeholder with no recorded value is spoken as it is rather than guessed at, and the gap is logged.

**The screen shows what the room hears; the database keeps the placeholders.** A surface here is served from localhost to a browser on this device, in the room with the person who just spoke, so reading `[GIVENNAME_1]` back to somebody about their own name is theatre rather than privacy. The stored transcript is a different question and keeps the masked text: it is the only artifact that outlives the process, and therefore the only one that can be carried off on a card. None of this changes what leaves the device, which is masked either way.

One consequence worth knowing before you look at a screen: the map lives in memory and dies with the process, so conversation history from before a restart can only ever be redisplayed masked.

The request happens on its own worker, so a slow network never stalls masking or the microphone.

## Speech

Text to speech lives in the core process, not in a surface, so it behaves identically with a screen, with no screen, and in terminal mode. Every utterance ducks music to roughly a fifth of the chosen level and restores it afterwards, including when synthesis fails.

Backends are local processes only. Pick one explicitly, or leave it unset to take the first that is installed, in the order piper, kokoro, espeak-ng, macOS say:

```text
ONKITH_TTS=piper
```

Piper is the lighter of the two and the default plan for the Pi:

```text
ONKITH_PIPER_MODEL=/path/to/voice.onnx
```

Kokoro sounds better and costs more, so it is worth trying only if the Pi has headroom:

```text
ONKITH_KOKORO_MODEL=/path/to/kokoro-v1.0.onnx
ONKITH_KOKORO_VOICES=/path/to/voices-v1.0.bin
ONKITH_KOKORO_VOICE=af_sarah
```

Install it as `kokoro-onnx`, never the plain `kokoro` package. The plain package pulls a full deep-learning framework onto the device, which this runtime does not allow; `kokoro-onnx` runs on the same onnxruntime as the masker and stays inside that rule.

With no voice installed, the ducking cycle and speaking state still run, and the reason is reported once.

## The screen

The screen is a conversation, not a chat app. There is no keyboard, no text field and no send button, and it is read from across a room, so it drops most of the conventions that go with those: no bubbles, no per-message timestamps, no scrollbar, no scrollback. What it shows is one exchange at a time, with the turns before it receding above until they fade out.

**The creature is the mark, alive.** The brand's mark is an eave: a cover, a wall, and a sheltered block beneath. The block is the creature and the eave is where it lives, so the cover and the wall never move and the block is what has a life. It breathes at rest, leans out toward the room while listening, paces while thinking, rides the voice while speaking, and goes back under the cover and waits when the gate stops something. Eyes were never an option: the brand rules them out along with padlocks, shields and fingerprints, and it bans gradients, shadows and glow, so the presence comes from scale and motion rather than from light.

**Colour says the device is doing something, and nothing else.** At rest it is colourless, which is the rule the concept hangs on. Listening, thinking and speaking take the accent, on the creature, on the rings that arrive while it listens, and on the state word. What became of the words is a separate channel and lives on the chip, in the fixed vocabulary: `Held` stayed here, `Sent` went to the model as placeholders, `Blocked` was stopped by the masker. One hue is never asked to carry two meanings, so the gate stopping something is told apart by behaviour rather than colour.

Four signals drive all of it, and every one comes from the core rather than being guessed at on the page:

| Topic | Says |
|---|---|
| `listening` | the microphone is holding an utterance open right now |
| `assistant` | idle, thinking, speaking |
| `gate` | what became of the last utterance |
| `conversation` | the rolling turns, rehydrated |

Hearing somebody outranks everything else. While a person is mid sentence, nothing the device is doing about the last one matters more than showing that this one is landing.

**The player is a companion panel.** It arrives beside the conversation when there are results and leaves when there are none, and it follows results rather than whether audio happens to be running: a paused track is still the thing on the deck, and closing the panel underneath somebody who just pressed pause would be the screen arguing with them. Below a square aspect it becomes a sheet underneath instead. Everything inside it is sized against the panel rather than the window, so the same markup works in both.

The track itself is not animated. Measured in a browser, a transition on `grid-template-columns` never progresses: it reports the closed value forever and leaves the panel a sliver wide, with `fr` tracks and with percentages alike. The panel snaps to width and carries its own arrival with a fade, which is also the cheaper of the two on a Pi given there is a WebGL canvas inside the track being resized.

## Interactive recorder

The physical player is a Spline scene, published from the same source the earlier GLB was exported from and shipped finished: its own textures, its own contact shadows, its own printed symbols and a lit display. Nothing in this repository builds, shades or lights it, and nothing rebuilds it in CSS, SVG, canvas or Three.js.

**It is served from this device.** The viewer runtime and the scene are in `assets/player/spline`, and the page loads them from localhost. Confirmed by watching the network across a full load and every interaction: four files are fetched and nothing else, no unpkg, no `prod.spline.design`, no Draco decoder from gstatic. The scene's textures travel inside the `.splinecode` file as blobs. That is about 10 MB on disk, and it is the price of the object appearing on a Pi with no network at all.

The viewer is pinned at 1.12.98. It is a vendored file rather than a dependency, so it cannot change underneath the parts of this module that read the scene graph to place controls.

The model controls map directly to playback:

- `play` toggles play and pause.
- `stop` stops playback and returns to the beginning.
- `record` toggles mute.
- `-` selects the previous verified result.
- `+` selects the next verified result.
- Dragging `disk` seeks through the track.
- Turning `bottom_knob_stick`, the orange knob, changes volume.

The knob is turned rather than pushed: the level follows the angle the hand sweeps around it, clockwise to raise, over the same 270 degrees of a real pot that the knob itself turns through. A hand that moves clear of the knob before turning gets a wider arc and finer control than a 44 pixel target could ever offer, and turning past either end stops there rather than banking rotation that has to be unwound before the level moves back. The platter turns at 33 1/3 rpm while audio plays, a revolution every 1.8 seconds, and stands still when it stops. Pressing a key rocks it.

Each control is identified by its printed symbol on the object, so no written label is drawn at rest; the label appears on hover and on keyboard focus, and screen readers read it either way. If the scene cannot be drawn, the player reveals conventional play, stop, mute, seek, volume, previous, and next controls, and says why.

### Placing the controls on the object

The scene is composed at three quarters, which is a good photograph and a bad control panel: it foreshortens the keys, hides one side of the deck, and skews every projected target away from the face it belongs to. The device is turned to face the room instead, square on, from the distance the scene already chose.

Square on is also what makes the controls a measurement rather than a guess. The camera is orthographic, so a control's near and far faces project to the same rectangle and its whole bounding box is its footprint. At three quarters this had to take only the corners nearest the camera, and on a transport key those belong to the printed symbol rather than the cap it is printed on.

The viewer only reports clicks for the parts the scene's author gave an interaction to, which here is three keys and two knobs. The page needs seven controls, so each part is projected out of the scene and the page's own button is laid over it. Three more things had to be true:

- **Empty bounding boxes are skipped.** The scene's text meshes carry a box whose minimum is positive infinity. One of those anywhere under a control turns the measurement into `NaN`, and the control is then silently never placed: it is why the platter and the two small keys had no target at all until it was guarded.
- **Smaller controls stack above larger ones.** The platter is most of the front of the device and the `-` and `+` keys sit inside its box, so without ordering by area they cannot be clicked.
- **The camera has to stop moving first.** The scene animates its camera into position on load, so the placement waits for the matrix to go quiet rather than for a fixed delay. Quiet means *within a tolerance*, not identical: a camera that has stopped still dithers in the last bits of a double, and the z it sits at is in the hundreds, so the value walks a few ten-trillionths every frame forever. Compared exactly it never stopped, the wait rode out its full six-second timeout on every load, and the object was held back for nine seconds when the scene itself was ready in one.

The camera is then held still. Orbit is left switched on as a controller, because it is what runs that opening move and what the zoom goes through, but rotation, panning and wheel zoom are all switched off. On an appliance a stray drag is a way to lose the front of the device and never find it again; wheel zoom is worse, because a scroll over the stage shrinks the object away instead of scrolling the page. With it off the wheel falls through and the stage scrolls like the rest of the page, and a finger does the same: the viewer takes vertical panning, and only the two controls that are dragged keep the gesture.

The viewer's "Built with Spline" badge and its drag hint are removed at load. The hint describes a gesture the object no longer answers, and the badge is not part of this device.

The object is a display surface: it takes no pointer input at all, and the page's own controls over it take every click. That is not only about who owns input. The scene carries a click sample its author added, and it plays whenever one of the authored parts is pressed, whether by a click that missed a control or by asking the scene to animate the press. This device makes two sounds, the music and Onkith, so nothing here fires a scene event and the key press is animated from this side instead: the same tilt the scene's rocker uses, but momentary and silent.

Dragging a control moves the part it belongs to while the track is playing, not only when it is stopped. The platter's angle carries where the track is as well as the turning it has done under its own power, so a scrub moves it either way, and the volume knob follows the hand rather than waiting for the level to come back from the core.

**A part is turned through the viewer's handle for it, never by writing to the scene node.** Every object in the scene is loaded with its matrix update switched off, because the runtime composes matrices itself from its own copy of each transform. A rotation set straight on a node is therefore stored and never composed: the platter's angle climbed past fifty radians while the object stood perfectly still on screen, and reading the property back agreed with itself the whole way, which is exactly how it survived two rounds of checking. The handle is asked for by the node's id rather than by its name, so a name that appears twice cannot turn one part while the page measures another. Scene nodes are still what get measured — a handle carries a transform but no geometry to project a control against.

Waiting is not the only way that goes wrong. Every message from the core carries its idea of the level, and a message already in flight carries the level from before the drag started; applying it mid-drag snapped the knob back to where the hand began, over and over, which reads as a control that does not move at all. The core's level is ignored while its own control is being dragged, which is what `previewingSeek` already did for the timeline.

The object is held back until it has been turned square on, sized to the stage and measured. The scene opens by flying its own camera into place at the angle it was composed at, so shown immediately it arrives tilted and cropped and then corrects itself; a beat of black is better than a wrong object fixing itself in front of the room.

The scene is published with a fixed zoom that frames it for the window it was authored in, so any other shape of box crops it rather than resizing it. The object is measured on screen and the camera scaled until the whole device sits inside the box the page gave it.

## Records

Every music result is a record. The one that is playing is the record that is out of its sleeve: a bare disc in the rail wearing the track's artwork as its label. The four that are not playing are still in their sleeves, disc drawn out to the right, the way a record is held before it goes on.

Both are the same black vinyl, cut to the same 0.34 label ratio, and differ only in size. Being unsleeved and twice as large is what marks the one that is loaded, so nothing needs a frame around it or a second material to stand out; a silver disc beside four black ones reads as a different object rather than the same one out of its cover.

A sleeve is the jacket plus the disc standing out of it, at the proportions of a real LP photograph: the jacket square, the disc 0.91 of it, and the disc's centre just past the opening so its label stays readable rather than hiding behind the artwork. The disc reaches the end of its card and no further, so a record never laps the sleeve of the result beside it. It is a sibling of the jacket rather than a child, because the jacket clips its own artwork and the disc has to stand outside that. Its label is printed paper: the artwork is on the jacket now, at four times the size it had as a label, so printing it on the disc as well would only be the same picture twice.

Labels are cut from the centre of the artwork. YouTube serves both true 16:9 files and letterboxed 4:3 ones, and a square or circular crop of a 4:3 file lands its top and bottom squarely in the black bars, so a 16:9 variant is requested and the crop is still derived from whatever actually arrives.\n\nThe platter on the object does not carry the track's artwork. The viewer exposes an object's transform, visibility and colour, but not its texture maps, so a per-track label cannot be printed onto it from here. The artwork is on the rail instead, where it is larger anyway.

Suggestion cards preserve normal click behavior. Mouse pointer capture begins only after a rail drag crosses its movement threshold, so clicking either side of the active track selects that result directly.

No JavaScript package download is required when the player starts.

## Terminal UI

The included `onkith_ui.py` is the original zero-dependency ANSI interface. It uses the Onkith palette, animated state indicator, audio level, masked utterance rendering, and pinned three-row status area. When output is redirected or the process is run without a TTY, it falls back to plain text with no animation thread or escape codes.

The terminal normally shows only masked output. Raw transcripts are retained in the local SQLite session database. These files are ignored by Git:

```text
session_transcript.db
session_transcript.db-wal
session_transcript.db-shm
```

## Useful checks

Run the static self-test:

```sh
./onkith --self-test
```

Test masking without using a microphone:

```sh
./onkith --text-test "Call Turki at 555 123 4567"
```

Run the complete test suite:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Raspberry Pi setup remains available through `install_pi.sh`, `check_pi.sh`, and `run_live.sh`.

## Runtime pipeline

1. The microphone records one bounded utterance.
2. Moonshine converts the audio to text locally.
3. Onkith wake commands are separated from ordinary speech.
4. Music commands are sent to the isolated music worker.
5. Ordinary speech is masked by the local classifier and structured recognizers.
6. The terminal displays the masked result and SQLite stores the local record.

Masking failures fail closed and display `[BLOCKED: PII MASKING FAILED]`.
