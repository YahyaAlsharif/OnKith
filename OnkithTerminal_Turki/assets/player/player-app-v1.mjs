import {
  createRecorderMachine,
  sixteenByNineThumbnail,
  VOLUME_SWEEP_RADIANS,
} from "./player-spline-v1.mjs";

// This surface owns no audio. The core process plays music in a headless
// mpv child and publishes state here; every control sends a command back
// over the same socket. Closing or killing this page stops nothing.

const DEFAULT_VOLUME = 82;
const TIMELINE_INTERVAL_MS = 250;
const DRAG_THRESHOLD = 7;
const RECONNECT_MINIMUM_MS = 400;
const RECONNECT_MAXIMUM_MS = 5000;
const POINTER_IDLE_MS = 3000;
// Named as the scene names them, so a press animates the part that was
// pressed. Only the three keys and the two knobs carry an authored press;
// the rest are quiet.
const MACHINE_NODE_BY_ACTION = Object.freeze({
  toggle: "play",
  stop: "stop",
  mute: "record",
  previous: "-",
  next: "+",
});

let tracks = [];
let currentIndex = 0;
let playback = {
  available: false,
  ducked: false,
  duration: 0,
  index: 0,
  muted: false,
  playing: false,
  position: 0,
  state: "idle",
  volume: DEFAULT_VOLUME,
};
let outputLevel = { peak: 0, rms: 0 };
let assistantState = "idle";
let hearing = false;
let panelAskedFor = null;
let devices = { mic: {}, audio: {} };
let socket = null;
let connected = false;
let reconnectDelay = RECONNECT_MINIMUM_MS;
let reconnectTimer = null;
let positionBase = 0;
let positionMeasuredAt = 0;
let timelineTimer = null;
let machine = null;
let machineReady = false;
let machineFailed = false;
let previewingSeek = false;
let railSlots = [];
let activeRailCard = null;
let dragSession = null;
let suppressClickUntil = 0;
let hotspotDrag = null;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function elementById(id) {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`Player element ${id} was not found`);
  }
  return element;
}

const elements = {
  trackFlow: elementById("track-flow"),
  youtubeStage: elementById("youtube-stage"),
  title: elementById("title"),
  channel: elementById("channel"),
  trackIndex: elementById("track-index"),
  status: elementById("status"),
  playerAlert: elementById("player-alert"),
  connection: elementById("connection"),
  connectionCopy: elementById("connection-copy"),
  elapsed: elementById("elapsed"),
  duration: elementById("duration"),
  progress: elementById("progress"),
  machineViewer: elementById("machine-viewer"),
  machineStage: elementById("machine-stage"),
  machineHotspots: elementById("machine-hotspots"),
  fallbackControls: elementById("fallback-controls"),
  fallbackSeek: elementById("fallback-seek"),
  fallbackVolume: elementById("fallback-volume"),
  currentArtwork: elementById("current-artwork"),
  shell: elementById("app-shell"),
  before: elementById("before"),
  heard: elementById("heard"),
  said: elementById("said"),
  gateChip: elementById("gate-chip"),
  petState: elementById("pet-state"),
  deviceMic: elementById("device-mic"),
  micState: elementById("mic-state"),
  micName: elementById("mic-name"),
  micDetail: elementById("mic-detail"),
  micMeter: elementById("mic-meter"),
  deviceAudio: elementById("device-audio"),
  audioState: elementById("audio-state"),
  audioName: elementById("audio-name"),
  audioDetail: elementById("audio-detail"),
  deviceLight: elementById("device-light"),
  lightState: elementById("light-state"),
  lightLevel: elementById("light-level"),
  lightDetail: elementById("light-detail"),
  lamp: elementById("lamp"),
  speaker: document.querySelector(".obj-speaker"),
};

function selectedVideo() {
  return tracks[currentIndex] || null;
}

function thumbnailFor(video) {
  if (video.thumbnail && video.thumbnail.length > 0) return video.thumbnail;
  return `https://i.ytimg.com/vi/${encodeURIComponent(video.video_id)}/hqdefault.jpg`;
}

function artworkFor(video) {
  // Artwork is cropped square here, as a sleeve, and cut to a circle on the
  // disc and on the record on the platter. YouTube's 4:3 variants carry
  // baked-in black bars, which a square or circular crop lands straight in,
  // so ask for the 16:9 file and crop that.
  return sixteenByNineThumbnail(thumbnailFor(video));
}

function formatTime(seconds) {
  if (Number.isFinite(seconds) === false || seconds < 0) return "00:00";
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normaliseRatio(value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) === false) return 0;
  return clamp(numeric, 0, 1);
}

function setStatus(message) {
  elements.status.textContent = message;
}

function showAlert(message) {
  elements.playerAlert.textContent = message;
  elements.playerAlert.hidden = false;
}

function clearAlert() {
  elements.playerAlert.textContent = "";
  elements.playerAlert.hidden = true;
}

function setConnection(state, message) {
  elements.connection.classList.remove("is-ready", "is-error");
  if (state.length > 0) elements.connection.classList.add(state);
  elements.connectionCopy.textContent = message;
}

function refreshConnection() {
  if (connected === false) {
    setConnection("is-error", "Reconnecting to Onkith");
    return;
  }
  if (playback.available === false) {
    setConnection("is-error", "Playback unavailable");
    return;
  }
  if (machineFailed) {
    setConnection("is-ready", "Onkith connected. Recorder fallback active");
    return;
  }
  if (machineReady) {
    setConnection("is-ready", "Onkith and recorder connected");
    return;
  }
  setConnection("is-ready", "Onkith connected. Preparing recorder");
}

// ------------------------------------------------------------------ socket

function send(command, payload = {}) {
  if (socket === null || socket.readyState !== WebSocket.OPEN) return;
  try {
    socket.send(JSON.stringify({ command, payload }));
  } catch (error) {
    // A send that races a closing socket is recovered by the reconnect.
  }
}

function scheduleReconnect() {
  if (reconnectTimer !== null) return;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, reconnectDelay);
  reconnectDelay = Math.min(RECONNECT_MAXIMUM_MS, reconnectDelay * 2);
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  try {
    socket = new WebSocket(`${protocol}//${window.location.host}/events`);
  } catch (error) {
    scheduleReconnect();
    return;
  }

  socket.addEventListener("open", () => {
    connected = true;
    reconnectDelay = RECONNECT_MINIMUM_MS;
    clearAlert();
    refreshConnection();
  });

  socket.addEventListener("message", (event) => {
    let message = null;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      return;
    }
    if (message === null || typeof message !== "object") return;
    applyMessage(message);
  });

  socket.addEventListener("close", () => {
    connected = false;
    refreshConnection();
    showAlert("Lost the connection to Onkith. Reconnecting.");
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    if (socket !== null && socket.readyState !== WebSocket.OPEN) {
      connected = false;
      refreshConnection();
    }
  });
}

function applyMessage(message) {
  const type = message.type;
  if (type === "snapshot") {
    if (message.tracks) applyTracks(message.tracks);
    if (message.playback) applyPlayback(message.playback);
    if (message.level) applyLevel(message.level);
    if (message.assistant) applyAssistant(message.assistant);
    if (message.listening) applyListening(message.listening);
    if (message.gate) applyGate(message.gate);
    if (message.panel) applyPanel(message.panel);
    if (message.devices) applyDevices(message.devices);
    if (message.light) applyLight(message.light);
    if (message.input) applyInput(message.input);
    if (message.conversation) applyConversation(message.conversation);
    if (message.status) applyStatus(message.status);
    refreshConnection();
    return;
  }
  if (type === "tracks") applyTracks(message);
  else if (type === "playback") applyPlayback(message);
  else if (type === "level") applyLevel(message);
  else if (type === "assistant") applyAssistant(message);
  else if (type === "listening") applyListening(message);
  else if (type === "gate") applyGate(message);
  else if (type === "panel") applyPanel(message);
  else if (type === "devices") applyDevices(message);
  else if (type === "light") applyLight(message);
  else if (type === "input") applyInput(message);
  else if (type === "conversation") applyConversation(message);
  else if (type === "status") applyStatus(message);
}

// The creature carries one state at a time, and they do not all come from
// the same place, so the order is what decides. Hearing somebody wins:
// while a person is mid sentence, nothing the device is doing about the
// last one matters more than showing that this one is landing.
function refreshPet() {
  const state = hearing
    ? "listening"
    : assistantState === "speaking" || assistantState === "thinking"
      ? assistantState
      : "resting";
  elements.shell.dataset.pet = state;
  elements.petState.textContent = state.charAt(0).toUpperCase() + state.slice(1);
}

function applyListening(section) {
  hearing = section.hearing === true;
  // A new utterance clears the last one off the screen rather than leaving
  // it there to be read as a reply to what is being said now.
  if (hearing) elements.heard.textContent = "";
  refreshPet();
  refreshMicCard();
}

// ------------------------------------------------------ attached devices

function applyDevices(section) {
  if (section.open !== undefined) {
    elements.shell.dataset.devices = section.open === true ? "open" : "closed";
    // The middle column has just changed width and the recorder in the
    // other panel measures itself against the box it is given.
    window.dispatchEvent(new Event("resize"));
  }
  if (section.mic) devices.mic = section.mic;
  if (section.audio) devices.audio = section.audio;
  refreshMicCard();
  refreshAudioCard();
}

function setCard(card, status, state, name, detail) {
  card.dataset.status = status;
  card.querySelector(".device-state").textContent = state;
  card.querySelector(".device-name").textContent = name;
  card.querySelector(".device-detail").textContent = detail;
}

function refreshMicCard() {
  const mic = devices.mic || {};
  if (!mic.name) {
    // Before capture opens, and on a surface-only run, which never opens
    // one at all. Saying "not opened" beats an empty card that reads as a
    // panel that failed to load.
    setCard(elements.deviceMic, "missing", "None", "Not opened", "");
    return;
  }
  const detail = [
    mic.id === null || mic.id === undefined ? null : `id ${mic.id}`,
    mic.sample_rate ? `${(mic.sample_rate / 1000).toFixed(1)} kHz` : null,
  ]
    .filter(Boolean)
    .join("  ·  ");
  setCard(
    elements.deviceMic,
    hearing ? "live" : "ready",
    hearing ? "Hearing" : "Idle",
    mic.name,
    detail,
  );
}

function refreshAudioCard() {
  const audio = devices.audio || {};
  // Playback is the honest headline. A sink can be named while mpv is
  // missing, and on this board the usual failure is the other way round:
  // everything installed and nothing to play it through.
  const sink = audio.sink || "No output device";
  const detail = [
    audio.server ? `${audio.server}` : null,
    audio.speech ? `voice ${audio.speech}` : null,
    audio.playback === true ? "mpv ready" : "mpv unavailable",
  ]
    .filter(Boolean)
    .join("  ·  ");
  const missing = !audio.sink;
  setCard(
    elements.deviceAudio,
    missing ? "missing" : "ready",
    missing ? "Silent" : "Ready",
    sink,
    detail,
  );
}

function applyLight(section) {
  const brightness = Number.isFinite(section.brightness)
    ? Math.max(0, Math.min(100, section.brightness))
    : 0;
  const on = section.on === true;
  // Off is dark, whatever the remembered level says. The controller keeps
  // the brightness through an off so that on returns to it, and drawing
  // that remembered number as light would contradict the lamp on the desk.
  const lit = on ? brightness / 100 : 0;
  elements.lamp.style.setProperty("--lamp", String(lit));
  // The card warms with the lamp. Light is the only thing on this panel
  // that emits rather than reports, so it is the only one allowed out of
  // its own box.
  elements.deviceLight.style.setProperty("--card-glow", String(lit));

  const wired = section.backend && section.backend !== "none";
  elements.lightLevel.textContent = `${on ? brightness : 0}%`;
  setCardStatus(
    elements.deviceLight,
    wired ? (on ? "live" : "ready") : "missing",
  );
  elements.lightState.textContent = wired ? (on ? "On" : "Off") : "Not wired";
  elements.lightDetail.textContent = wired
    ? `${section.backend}  ·  set to ${brightness}%`
    : "No PWM chip on this machine";
}

function setCardStatus(card, status) {
  card.dataset.status = status;
}

// The microphone's basket, filling from the bottom with the voice in the
// room. Scaled the same way the speaker is, so the two objects on the
// panel are reading in the same units as each other.
function applyInput(section) {
  elements.micMeter.style.setProperty(
    "--level",
    loudness(section.rms).toFixed(3),
  );
}

const GATE_LABEL = { held: "Held", sent: "Sent", blocked: "Blocked" };

function applyGate(section) {
  const state = GATE_LABEL[section.state] ? section.state : "held";
  elements.shell.dataset.gate = state;
  elements.gateChip.dataset.gate = state;
  elements.gateChip.textContent = GATE_LABEL[state];
}

// The newest exchange is what the room is looking at, so it is shown whole
// and large. Everything before it recedes above the creature: enough to see
// that a conversation happened, not enough to read from across the room.
const RETAINED_QUIET_TURNS = 4;

function applyConversation(section) {
  const turns = Array.isArray(section.turns) ? section.turns : [];
  const latestReply = lastIndexOfRole(turns, "onkith");
  const latestHeard = lastIndexOfRole(turns, "user");

  // The exchange on show is the newest question and the reply that came
  // after it. A question still waiting for one leaves the answer blank
  // rather than standing an old answer underneath a new question, and the
  // old answer recedes into the quiet stack where it belongs.
  const replyAnswersIt = latestReply > latestHeard;
  if (replyAnswersIt) drawTurn(elements.said, turns[latestReply]);
  else elements.said.replaceChildren();
  if (latestHeard === -1) elements.heard.replaceChildren();
  else drawTurn(elements.heard, turns[latestHeard]);

  // Everything before that exchange, so the newest question is never shown
  // twice, once live and once behind the creature.
  const exchangeStart = latestHeard === -1 ? Math.max(latestReply, 0) : latestHeard;
  const shown = turns.slice(0, exchangeStart).slice(-RETAINED_QUIET_TURNS);
  elements.before.replaceChildren();
  shown.forEach((turn, index) => {
    const line = document.createElement("p");
    line.dataset.role = turn.role === "onkith" ? "onkith" : "room";
    line.dataset.age = String(shown.length - index - 1);
    drawTurn(line, turn);
    elements.before.append(line);
  });
}

// Draw a turn with the words the masker held back marked out. The core
// sends offsets rather than markup, and every piece goes in as text, so a
// sentence somebody said out loud can never become markup on the screen.
function drawTurn(target, turn) {
  const text = String(turn.text || "");
  const revealed = Array.isArray(turn.revealed) ? turn.revealed : [];
  target.replaceChildren();
  if (revealed.length === 0) {
    target.textContent = text;
    return;
  }
  let cursor = 0;
  revealed
    .slice()
    .sort((left, right) => left.start - right.start)
    .forEach((span) => {
      const start = Number(span.start);
      const end = Number(span.end);
      if (!Number.isInteger(start) || !Number.isInteger(end)) return;
      if (start < cursor || end > text.length || end <= start) return;
      if (start > cursor) target.append(text.slice(cursor, start));
      const mark = document.createElement("span");
      mark.className = "revealed";
      // Named for a screen reader, which cannot see the colour.
      mark.title = `Held back from the model: ${String(span.kind || "value")}`;
      mark.textContent = text.slice(start, end);
      target.append(mark);
      cursor = end;
    });
  if (cursor < text.length) target.append(text.slice(cursor));
}

function lastIndexOfRole(turns, role) {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (turns[index].role === role) return index;
  }
  return -1;
}

function applyTracks(section) {
  const items = Array.isArray(section.items) ? section.items : tracks;
  const nextIndex = Number.isInteger(section.index) ? section.index : currentIndex;
  const listChanged = items !== tracks;
  // A new search hands the decision back to the screen. Having asked for
  // the panel to go away once should not mean it stays away for good.
  if (listChanged) panelAskedFor = null;
  tracks = items;
  currentIndex = clamp(nextIndex, 0, Math.max(0, tracks.length - 1));
  if (listChanged) {
    buildTrackFlow();
  } else {
    updateTrackFlow("core", 0);
  }
  setCurrentMetadata();
  updateControlAvailability();
  refreshPlayerPanel();
}

// Somebody asking for the player out loud outranks the screen's own
// judgement, but only until the next search. Asking for it to go away
// should not mean it stays away the next time music starts.
function applyPanel(section) {
  panelAskedFor = section.open === null || section.open === undefined
    ? null
    : section.open === true;
  refreshPlayerPanel();
}

// Otherwise the player arrives beside the conversation when there is
// something to play and leaves when there is not. Having results is what
// decides, not whether audio happens to be paused: a paused track is still
// the thing on the deck, and closing the panel underneath somebody who just
// pressed pause would be the screen arguing with them.
function refreshPlayerPanel() {
  const open = panelAskedFor === null ? tracks.length > 0 : panelAskedFor;
  if ((elements.shell.dataset.player === "open") === open) return;
  elements.shell.dataset.player = open ? "open" : "closed";
  // The recorder measures itself against the box it is given, and that box
  // has just changed shape.
  window.dispatchEvent(new Event("resize"));
}

function applyPlayback(section) {
  const previousState = playback.state;
  // Every playback message carries the core's idea of the level, and a
  // message already in flight carries the one from before the drag. Taking
  // it mid-drag snaps the knob back to where the hand started, which reads
  // as a control that does not move. The timeline is protected from the
  // same race by previewingSeek.
  const draggingVolume = hotspotDrag !== null && hotspotDrag.kind === "volume";
  const incoming = { ...section };
  if (draggingVolume) delete incoming.volume;
  playback = { ...playback, ...incoming };
  if (Number.isFinite(section.position)) {
    positionBase = section.position;
    positionMeasuredAt = performance.now();
  }
  if (Number.isInteger(section.index) && section.index !== currentIndex) {
    currentIndex = clamp(section.index, 0, Math.max(0, tracks.length - 1));
    updateTrackFlow("core", 0);
    setCurrentMetadata();
  }
  callMachine("setPlaybackState", playback.playing ? "playing" : playback.state);
  callMachine("setMuted", playback.muted);
  callMachine("setVolume", playback.volume);
  if (playback.state !== previousState) announcePlaybackState();
  syncVolumeInterface();
  updateControlAvailability();
  updateTimeline();
  refreshConnection();
}

function applyLevel(section) {
  outputLevel = {
    peak: Number(section.peak) || 0,
    rms: Number(section.rms) || 0,
  };
  // The speaker's dust cap, pushed by what is actually coming out. This
  // number was measured and stored and never shown until the devices
  // panel had somewhere to put it.
  elements.speaker.style.setProperty("--out", String(loudness(outputLevel.rms)));
}

// Sound sits low in a linear scale, so a raw RMS barely moves anything at
// a normal speaking or listening volume, which is the opposite of what
// these two meters are for. The cube root opens out the quiet end, where
// the difference between working and nearly deaf actually lives.
function loudness(rms) {
  return Math.cbrt(Math.max(0, Math.min(1, Number(rms) || 0)));
}

function applyAssistant(section) {
  assistantState = String(section.state || "idle");
  document.documentElement.dataset.assistant = assistantState;
  refreshPet();
}

function applyStatus(section) {
  const message = String(section.message || "");
  if (message.length === 0) return;
  if (section.level === "error") showAlert(message);
  else setStatus(message);
}

function announcePlaybackState() {
  if (playback.state === "playing") {
    clearAlert();
    setStatus("Playing through your speakers.");
  } else if (playback.state === "buffering") {
    setStatus("Buffering the selected track.");
  } else if (playback.state === "paused") {
    setStatus("Paused.");
  } else if (playback.state === "stopped") {
    setStatus("Stopped.");
  } else {
    setStatus("Ready to play.");
  }
}

// -------------------------------------------------------------------- rail

function createTextElement(className) {
  const element = document.createElement("span");
  element.className = className;
  return element;
}

function createSuggestionCard(slotIndex, distance) {
  const card = document.createElement("article");
  card.className = "flow-card";
  card.dataset.distance = String(distance);
  card.dataset.slot = String(slotIndex);
  card.setAttribute("role", "listitem");

  const button = document.createElement("button");
  button.type = "button";

  const media = document.createElement("span");
  media.className = "flow-card-media";

  // A record that is not playing is in its sleeve, disc drawn out to the
  // right. The disc is a sibling rather than a child of the jacket because
  // it has to stand outside it, and the jacket clips its own artwork.
  const disc = document.createElement("span");
  disc.className = "flow-card-disc";

  const sleeve = document.createElement("span");
  sleeve.className = "flow-card-sleeve";

  const image = document.createElement("img");
  image.decoding = "async";
  image.draggable = false;

  const copy = document.createElement("span");
  copy.className = "flow-card-copy";
  const position = createTextElement("flow-position");
  const title = createTextElement("flow-card-title");
  const channel = createTextElement("flow-card-channel");
  copy.append(position, title, channel);
  sleeve.append(image);
  media.append(disc, sleeve);
  // The caption sits outside the sleeve: the jacket clips anything inside
  // it, and the caption belongs under the record rather than printed on it.
  button.append(media, copy);
  card.append(button);

  button.addEventListener("click", () => {
    if (performance.now() < suppressClickUntil) return;
    const index = Number(button.dataset.videoIndex);
    selectVideo(index, "card");
  });

  return { card, button, image, position, title, channel, media, active: false };
}

function createActiveCard(slotIndex) {
  const card = document.createElement("article");
  card.className = "flow-card is-active";
  card.dataset.distance = "0";
  card.dataset.slot = String(slotIndex);
  card.setAttribute("role", "listitem");
  card.setAttribute("aria-current", "true");
  elements.youtubeStage.classList.add("is-mounted");
  card.append(elements.youtubeStage);
  return { card, active: true };
}

function buildTrackFlow() {
  const count = tracks.length;
  railSlots = [];
  if (count === 0) {
    elements.trackFlow.replaceChildren();
    activeRailCard = null;
    return;
  }

  const activeSlotIndex = Math.floor(count / 2);
  const fragment = document.createDocumentFragment();
  for (let slotIndex = 0; slotIndex < count; slotIndex += 1) {
    const distance = Math.abs(slotIndex - activeSlotIndex);
    const slot = slotIndex === activeSlotIndex
      ? createActiveCard(slotIndex)
      : createSuggestionCard(slotIndex, distance);
    railSlots.push(slot);
    fragment.append(slot.card);
  }

  elements.trackFlow.replaceChildren(fragment);
  activeRailCard = railSlots[activeSlotIndex].card;
  updateTrackFlow("initial", 0);
  installRailDragging();
}

function railVideoIndex(slotIndex) {
  const activeSlotIndex = Math.floor(tracks.length / 2);
  const offset = slotIndex - activeSlotIndex;
  return (currentIndex + offset + tracks.length) % tracks.length;
}

function updateTrackFlow(reason, direction) {
  const selected = selectedVideo();
  if (selected === null) return;

  railSlots.forEach((slot, slotIndex) => {
    const videoIndex = railVideoIndex(slotIndex);
    const video = tracks[videoIndex];
    slot.card.dataset.videoIndex = String(videoIndex);

    if (slot.active) {
      slot.card.setAttribute("aria-label", `Now playing ${video.title} by ${video.channel}`);
      return;
    }

    slot.button.dataset.videoIndex = String(videoIndex);
    slot.button.setAttribute("aria-label", `Play ${video.title} by ${video.channel}`);
    slot.image.src = artworkFor(video);
    slot.image.alt = `Artwork for ${video.title} by ${video.channel}`;
    slot.image.loading = Number(slot.card.dataset.distance) > 1 ? "lazy" : "eager";
    slot.position.textContent = `${String(videoIndex + 1).padStart(2, "0")} / ${String(tracks.length).padStart(2, "0")}`;
    slot.title.textContent = video.title;
    slot.channel.textContent = video.channel;
  });

  if ((reason === "initial") === false && reducedMotion.matches === false) {
    const startOffset = direction >= 0 ? 18 : -18;
    railSlots.forEach((slot) => {
      if (slot.active) return;
      if (typeof slot.media.animate === "function") {
        slot.media.animate(
          [
            { opacity: 0.55, transform: `translateX(${startOffset}px)` },
            { opacity: 1, transform: "translateX(0)" },
          ],
          { duration: 280, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
        );
      }
    });
  }

  window.requestAnimationFrame(() => {
    centerActiveCard(reason === "initial" ? "auto" : "smooth");
  });
}

function centerActiveCard(behavior) {
  if (activeRailCard === null) return;
  const flow = elements.trackFlow;
  if ((typeof flow.scrollTo === "function") === false) return;
  // Measured against the rail, not against offsetLeft. offsetLeft is
  // relative to the offsetParent, and the rail is not positioned, so
  // that is the body: in the side panel it carried the entire width of
  // the conversation sitting to the left of it. The target came out
  // past the end of the rail, the browser clamped it to the end, and
  // the playing record sat off the left edge with half of it cut away.
  // It looked right only while the player was the whole page.
  const flowBox = flow.getBoundingClientRect();
  const cardBox = activeRailCard.getBoundingClientRect();
  const withinRail = (cardBox.left - flowBox.left) + flow.scrollLeft;
  const targetLeft = withinRail - ((flow.clientWidth - cardBox.width) / 2);
  flow.scrollTo({
    left: Math.max(0, targetLeft),
    behavior: reducedMotion.matches ? "auto" : behavior,
  });
}

function installRailDragging() {
  elements.trackFlow.addEventListener("pointerdown", (event) => {
    if ((event.pointerType === "mouse") === false || (event.button === 0) === false) return;
    if (elements.youtubeStage.contains(event.target)) return;
    dragSession = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startScrollLeft: elements.trackFlow.scrollLeft,
      dragged: false,
    };
  });

  elements.trackFlow.addEventListener("pointermove", (event) => {
    if (dragSession === null || (dragSession.pointerId === event.pointerId) === false) return;
    const delta = event.clientX - dragSession.startX;
    if (dragSession.dragged === false && Math.abs(delta) >= DRAG_THRESHOLD) {
      dragSession.dragged = true;
      elements.trackFlow.classList.add("is-dragging");
      elements.trackFlow.setPointerCapture(event.pointerId);
    }
    if (dragSession.dragged) {
      event.preventDefault();
      elements.trackFlow.scrollLeft = dragSession.startScrollLeft - delta;
    }
  });

  const finishDrag = (event) => {
    if (dragSession === null || (dragSession.pointerId === event.pointerId) === false) return;
    if (dragSession.dragged) suppressClickUntil = performance.now() + 300;
    elements.trackFlow.classList.remove("is-dragging");
    if (elements.trackFlow.hasPointerCapture(event.pointerId)) {
      elements.trackFlow.releasePointerCapture(event.pointerId);
    }
    dragSession = null;
  };

  window.addEventListener("pointerup", finishDrag);
  window.addEventListener("pointercancel", finishDrag);
  elements.trackFlow.addEventListener("click", (event) => {
    if (performance.now() >= suppressClickUntil) return;
    event.preventDefault();
    event.stopPropagation();
  }, true);
}

function setCurrentMetadata() {
  const video = selectedVideo();
  if (video === null) {
    elements.trackIndex.textContent = "No track";
    // No track, no record on the platter.
    callMachine("setArtwork", "");
    return;
  }
  elements.title.textContent = video.title;
  elements.channel.textContent = video.channel;
  elements.currentArtwork.src = artworkFor(video);
  elements.currentArtwork.alt = `Artwork for ${video.title} by ${video.channel}`;
  elements.trackIndex.textContent = `Track ${String(currentIndex + 1).padStart(2, "0")} / ${String(tracks.length).padStart(2, "0")}`;
  document.title = `${video.title} \\ Player \\ Onkith`;
  // The record on the platter wears the track's own artwork.
  callMachine("setArtwork", artworkFor(video));
}

// ---------------------------------------------------------------- commands

function selectionMessage(reason) {
  if (reason === "card") return `Loading selected result ${currentIndex + 1} of ${tracks.length}.`;
  if (reason === "previous") return `Loading previous result ${currentIndex + 1} of ${tracks.length}.`;
  if (reason === "next") return `Loading next result ${currentIndex + 1} of ${tracks.length}.`;
  return `Loading result ${currentIndex + 1} of ${tracks.length}.`;
}

function selectVideo(index, reason = "card") {
  const numericIndex = Number(index);
  if (Number.isInteger(numericIndex) === false || numericIndex < 0 || numericIndex >= tracks.length) {
    showAlert("That music result is not available.");
    return false;
  }
  const previousIndex = currentIndex;
  currentIndex = numericIndex;
  resetTimeline();
  setCurrentMetadata();
  updateTrackFlow(reason, Math.sign(currentIndex - previousIndex));
  updateControlAvailability();
  setStatus(selectionMessage(reason));
  send("select", { index: numericIndex });
  return true;
}

function selectAdjacent(delta, reason) {
  const nextIndex = currentIndex + delta;
  if (nextIndex < 0) {
    setStatus("This is the first available result.");
    return;
  }
  if (nextIndex >= tracks.length) {
    setStatus("This is the final available result.");
    return;
  }
  selectVideo(nextIndex, reason);
}

function togglePlayback() {
  if (playback.available === false) {
    setStatus("Playback is not available.");
    return;
  }
  send("toggle");
}

function stopPlayback() {
  send("stop");
  resetTimeline();
  setStatus("Stopped.");
}

function toggleMute() {
  send("mute");
}

function setVolume(value) {
  const numericValue = Number(value);
  if (Number.isFinite(numericValue) === false) return;
  playback.volume = Math.round(clamp(numericValue, 0, 100));
  syncVolumeInterface();
  send("volume", { value: playback.volume });
}

function seekToRatio(value) {
  const ratio = normaliseRatio(value);
  previewingSeek = false;
  const duration = Number(playback.duration) || 0;
  if (duration > 0) {
    positionBase = duration * ratio;
    positionMeasuredAt = performance.now();
  }
  send("seek", { ratio });
  updateTimeline();
}

function previewSeek(value) {
  const ratio = normaliseRatio(value);
  const duration = Number(playback.duration) || 0;
  previewingSeek = true;
  const elapsed = duration > 0 ? duration * ratio : 0;
  elements.elapsed.textContent = formatTime(elapsed);
  elements.progress.value = Math.round(ratio * 1000);
  elements.progress.textContent = `${Math.round(ratio * 100)}%`;
  elements.fallbackSeek.value = String(Math.round(ratio * 1000));
  updateSeekAria(ratio, elapsed, duration);
  callMachine("setProgress", ratio);
}

function actionPayloadNumber(payload, fallbackValue) {
  if (typeof payload === "number") return payload;
  if (payload && typeof payload.value === "number") return payload.value;
  if (payload && typeof payload.delta === "number") return fallbackValue + payload.delta;
  return fallbackValue;
}

function handleMachineAction(action, payload = null, source = "external") {
  const shouldPulse = (source === "machine") === false;
  if (action === "toggle") {
    if (shouldPulse) callMachine("pulseControl", MACHINE_NODE_BY_ACTION.toggle);
    togglePlayback();
    return;
  }
  if (action === "stop") {
    if (shouldPulse) callMachine("pulseControl", MACHINE_NODE_BY_ACTION.stop);
    stopPlayback();
    return;
  }
  if (action === "mute") {
    if (shouldPulse) callMachine("pulseControl", MACHINE_NODE_BY_ACTION.mute);
    toggleMute();
    return;
  }
  if (action === "previous") {
    if (shouldPulse) callMachine("pulseControl", MACHINE_NODE_BY_ACTION.previous);
    selectAdjacent(-1, "previous");
    return;
  }
  if (action === "next") {
    if (shouldPulse) callMachine("pulseControl", MACHINE_NODE_BY_ACTION.next);
    selectAdjacent(1, "next");
    return;
  }
  if (action === "volume-step") {
    setVolume(actionPayloadNumber(payload, playback.volume));
    return;
  }
  if (action === "volume") {
    const value = actionPayloadNumber(payload, playback.volume);
    setVolume(value >= 0 && value <= 1 ? value * 100 : value);
    return;
  }
  if (action === "seek") {
    const value = actionPayloadNumber(payload, 0);
    seekToRatio(value > 1 ? value / 1000 : value);
  }
}

// -------------------------------------------------------------- timeline

function resetTimeline() {
  positionBase = 0;
  positionMeasuredAt = performance.now();
  elements.elapsed.textContent = "00:00";
  elements.progress.value = 0;
  elements.progress.textContent = "0%";
  elements.fallbackSeek.value = "0";
  updateSeekAria(0, 0, Number(playback.duration) || 0);
  callMachine("setProgress", 0);
}

function displayedPosition() {
  if (playback.playing === false) return positionBase;
  const elapsed = (performance.now() - positionMeasuredAt) / 1000;
  const duration = Number(playback.duration) || 0;
  const projected = positionBase + elapsed;
  return duration > 0 ? Math.min(duration, projected) : projected;
}

function updateSeekAria(ratio, elapsed, duration) {
  const value = String(Math.round(normaliseRatio(ratio) * 1000));
  const valueText = `${formatTime(elapsed)} of ${formatTime(duration)}`;
  const seekHotspot = elements.machineHotspots.querySelector('[data-machine-action="seek"]');
  if (seekHotspot) {
    seekHotspot.setAttribute("aria-valuenow", value);
    seekHotspot.setAttribute("aria-valuetext", valueText);
  }
  elements.fallbackSeek.setAttribute("aria-valuetext", valueText);
  elements.progress.setAttribute("aria-valuetext", valueText);
}

function updateTimeline() {
  if (previewingSeek) return;
  const duration = Number(playback.duration) || 0;
  const elapsed = displayedPosition();
  const ratio = duration > 0 ? clamp(elapsed / duration, 0, 1) : 0;

  elements.elapsed.textContent = formatTime(elapsed);
  elements.duration.textContent = formatTime(duration);
  elements.progress.value = Math.round(ratio * 1000);
  elements.progress.textContent = `${Math.round(ratio * 100)}%`;
  if ((document.activeElement === elements.fallbackSeek) === false) {
    elements.fallbackSeek.value = String(Math.round(ratio * 1000));
  }
  updateSeekAria(ratio, elapsed, duration);
  callMachine("setProgress", ratio);
}

function startTimeline() {
  if ((timelineTimer === null) === false) window.clearInterval(timelineTimer);
  timelineTimer = window.setInterval(updateTimeline, TIMELINE_INTERVAL_MS);
}

function stopTimeline() {
  if (timelineTimer === null) return;
  window.clearInterval(timelineTimer);
  timelineTimer = null;
}

// --------------------------------------------------------------- machine

function callMachine(method, ...args) {
  if (machine === null || machine === undefined) return;
  if ((typeof machine[method] === "function") === false) return;
  try {
    machine[method](...args);
  } catch (error) {
    revealFallback("The recorder controls stopped responding.");
  }
}

function revealFallback(message) {
  if (machineFailed) return;
  machineFailed = true;
  machineReady = false;
  elements.fallbackControls.hidden = false;
  elements.machineHotspots.hidden = true;
  elements.machineViewer.setAttribute("aria-hidden", "true");
  // Whatever the object managed to draw is better than a black stage, and
  // the machine may have failed after it was already being shown.
  elements.machineViewer.classList.add("is-ready");
  elements.machineStage.classList.add("has-fallback");
  refreshConnection();
  updateControlAvailability();
  showAlert(`${message} Use the controls shown on the recorder.`);
}

async function initialiseMachine() {
  try {
    machine = await createRecorderMachine({
      viewer: elements.machineViewer,
      stage: elements.machineStage,
      hotspotRoot: elements.machineHotspots,
      onStatus(message, level) {
        if (level === "error") {
          revealFallback(message);
          return;
        }
        if (connected === false && typeof message === "string" && message.length > 0) {
          setStatus(message);
        }
      },
    });
    machineReady = true;
    machineFailed = false;
    elements.fallbackControls.hidden = true;
    elements.machineHotspots.hidden = false;
    elements.machineViewer.removeAttribute("aria-hidden");
    callMachine("setMuted", playback.muted);
    callMachine("setVolume", playback.volume);
    callMachine("setPlaybackState", playback.playing ? "playing" : playback.state);
    // The machine may finish loading after the first track arrived, so the
    // record is dressed here too rather than only on a track change.
    const current = selectedVideo();
    callMachine("setArtwork", current === null ? "" : artworkFor(current));
    updateTimeline();
    updateControlAvailability();
    refreshConnection();
  } catch (error) {
    // The reason matters on a device with no console attached: a missing
    // asset, a lost WebGL context and a slow load all end up here, and
    // they are not the same problem.
    const reason = error && error.message ? ` ${error.message}` : "";
    revealFallback(`The 3D recorder could not load.${reason}`);
  }
}

// ---------------------------------------------------------------- controls

function hotspotFor(action) {
  return elements.machineHotspots.querySelector(`[data-machine-action="${action}"]`);
}

function setHotspotDisabled(action, disabled) {
  const hotspot = hotspotFor(action);
  if (hotspot === null) return;
  hotspot.setAttribute("aria-disabled", disabled ? "true" : "false");
  if (hotspot instanceof HTMLButtonElement) hotspot.disabled = disabled;
  if (hotspot.getAttribute("role") === "slider") hotspot.tabIndex = disabled ? -1 : 0;
}

function syncVolumeInterface() {
  // The knob follows the hand rather than the round trip. Waiting for the
  // core to echo the new level back leaves it still while it is dragged.
  callMachine("setVolume", playback.volume);
  elements.fallbackVolume.value = String(playback.volume);
  elements.fallbackVolume.setAttribute("aria-valuetext", `${playback.volume} percent`);
  const volumeHotspot = hotspotFor("volume");
  if (volumeHotspot) {
    volumeHotspot.setAttribute("aria-valuenow", String(playback.volume));
    volumeHotspot.setAttribute("aria-valuetext", `${playback.volume} percent`);
  }
  const muteHotspot = hotspotFor("mute");
  if (muteHotspot) {
    muteHotspot.setAttribute("aria-pressed", playback.muted ? "true" : "false");
    muteHotspot.setAttribute("aria-label", playback.muted ? "Unmute" : "Mute");
    muteHotspot.dataset.label = playback.muted ? "Unmute" : "Mute";
  }
  const fallbackMute = elements.fallbackControls.querySelector('[data-fallback-action="mute"]');
  if (fallbackMute) {
    fallbackMute.textContent = playback.muted ? "Unmute" : "Mute";
    fallbackMute.setAttribute("aria-pressed", playback.muted ? "true" : "false");
  }
  callMachine("setMuted", playback.muted);
  callMachine("setVolume", playback.volume);
}

function updateControlAvailability() {
  const atFirst = currentIndex === 0;
  const atLast = tracks.length === 0 || currentIndex === tracks.length - 1;
  const unavailable = connected === false || playback.available === false
    || tracks.length === 0;
  setHotspotDisabled("previous", unavailable || atFirst);
  setHotspotDisabled("next", unavailable || atLast);
  setHotspotDisabled("toggle", unavailable);
  setHotspotDisabled("stop", unavailable);
  setHotspotDisabled("mute", unavailable);
  setHotspotDisabled("seek", unavailable);
  setHotspotDisabled("volume", connected === false || playback.available === false);

  const playing = playback.playing === true || playback.state === "buffering";
  const toggleHotspot = hotspotFor("toggle");
  if (toggleHotspot) {
    toggleHotspot.setAttribute("aria-pressed", playing ? "true" : "false");
    toggleHotspot.setAttribute("aria-label", playing ? "Pause" : "Play");
    toggleHotspot.dataset.label = playing ? "Pause" : "Play";
  }

  elements.fallbackControls.querySelectorAll("[data-fallback-action]").forEach((button) => {
    const action = button.dataset.fallbackAction;
    button.disabled = unavailable
      || (action === "previous" && atFirst)
      || (action === "next" && atLast);
    if (action === "toggle") {
      button.textContent = playing ? "Pause" : "Play";
      button.setAttribute("aria-pressed", playing ? "true" : "false");
    }
  });
  elements.fallbackSeek.disabled = unavailable;
  elements.fallbackVolume.disabled = connected === false
    || playback.available === false;
}

function installFallbackControls() {
  elements.fallbackControls.querySelectorAll("[data-fallback-action]").forEach((button) => {
    button.addEventListener("click", () => {
      handleMachineAction(button.dataset.fallbackAction);
    });
  });

  elements.fallbackSeek.addEventListener("input", () => {
    previewSeek(Number(elements.fallbackSeek.value) / 1000);
  });
  elements.fallbackSeek.addEventListener("change", () => {
    seekToRatio(Number(elements.fallbackSeek.value) / 1000);
  });
  elements.fallbackSeek.addEventListener("blur", () => {
    previewingSeek = false;
    updateTimeline();
  });
  elements.fallbackVolume.addEventListener("input", () => {
    setVolume(elements.fallbackVolume.value);
  });
}

function sliderStepFromKey(event, normalStep, largeStep) {
  if (event.key === "ArrowRight" || event.key === "ArrowUp") return normalStep;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") return -normalStep;
  if (event.key === "PageUp") return largeStep;
  if (event.key === "PageDown") return -largeStep;
  return null;
}

function installSeekHotspot(seekHotspot) {
  seekHotspot.addEventListener("keydown", (event) => {
    if (seekHotspot.getAttribute("aria-disabled") === "true") return;
    let ratio = Number(elements.progress.value) / 1000;
    const step = sliderStepFromKey(event, 0.01, 0.1);
    if (event.key === "Home") ratio = 0;
    else if (event.key === "End") ratio = 1;
    else if (step === null) return;
    else ratio = normaliseRatio(ratio + step);
    event.preventDefault();
    previewSeek(ratio);
    seekToRatio(ratio);
  });

  seekHotspot.addEventListener("pointerdown", (event) => {
    if (seekHotspot.getAttribute("aria-disabled") === "true") return;
    hotspotDrag = {
      kind: "seek",
      pointerId: event.pointerId,
      startX: event.clientX,
      startValue: Number(elements.progress.value) / 1000,
    };
    seekHotspot.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  seekHotspot.addEventListener("pointermove", (event) => {
    if (hotspotDrag === null || (hotspotDrag.kind === "seek") === false) return;
    if ((hotspotDrag.pointerId === event.pointerId) === false) return;
    const travel = Math.max(120, elements.machineStage.clientWidth * 0.42);
    previewSeek(hotspotDrag.startValue + ((event.clientX - hotspotDrag.startX) / travel));
  });

  const finish = (event) => {
    if (hotspotDrag === null || (hotspotDrag.kind === "seek") === false) return;
    if ((hotspotDrag.pointerId === event.pointerId) === false) return;
    const ratio = Number(elements.progress.value) / 1000;
    if (seekHotspot.hasPointerCapture(event.pointerId)) seekHotspot.releasePointerCapture(event.pointerId);
    hotspotDrag = null;
    seekToRatio(ratio);
  };
  seekHotspot.addEventListener("pointerup", finish);
  seekHotspot.addEventListener("pointercancel", finish);
}

// Close to the middle of a knob the angle is mostly noise: a pixel of
// movement swings it wildly, and crossing the centre flips it by half a
// turn. Inside this radius the pointer is ignored and the last angle kept.
const KNOB_DEAD_ZONE_PX = 10;

// Where the pointer sits around the knob. Screen y grows downward, so a
// growing angle is a clockwise turn, which is the direction that raises
// this knob's level, the same as a real pot.
function knobAngle(hotspot, event) {
  const box = hotspot.getBoundingClientRect();
  const acrossX = event.clientX - (box.left + box.width / 2);
  const acrossY = event.clientY - (box.top + box.height / 2);
  if (Math.hypot(acrossX, acrossY) < KNOB_DEAD_ZONE_PX) return null;
  return Math.atan2(acrossY, acrossX);
}

// The knob is turned rather than pushed, so the level follows the angle the
// hand sweeps around it instead of how far up the screen it travels. The
// sweep matches the part: 270 degrees of hand is 270 degrees of knob is the
// whole range. Because the drag is captured, the hand can move well clear
// of a control this small and turn on a wider arc, which is finer than
// anything the knob's own 44 pixels could offer.
function installVolumeHotspot(volumeHotspot) {
  volumeHotspot.addEventListener("keydown", (event) => {
    if (volumeHotspot.getAttribute("aria-disabled") === "true") return;
    let value = playback.volume;
    const step = sliderStepFromKey(event, 4, 10);
    if (event.key === "Home") value = 0;
    else if (event.key === "End") value = 100;
    else if (step === null) return;
    else value += step;
    event.preventDefault();
    setVolume(value);
  });

  volumeHotspot.addEventListener("pointerdown", (event) => {
    if (volumeHotspot.getAttribute("aria-disabled") === "true") return;
    hotspotDrag = {
      kind: "volume",
      pointerId: event.pointerId,
      lastAngle: knobAngle(volumeHotspot, event),
      turned: 0,
      startValue: playback.volume,
    };
    volumeHotspot.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  volumeHotspot.addEventListener("pointermove", (event) => {
    if (hotspotDrag === null || (hotspotDrag.kind === "volume") === false) return;
    if ((hotspotDrag.pointerId === event.pointerId) === false) return;
    const angle = knobAngle(volumeHotspot, event);
    if (angle === null) return;
    if (hotspotDrag.lastAngle === null) {
      hotspotDrag.lastAngle = angle;
      return;
    }
    let step = angle - hotspotDrag.lastAngle;
    // The angle wraps at half a turn. Without this, crossing that point
    // reads as a sweep the whole way back around the other way.
    if (step > Math.PI) step -= Math.PI * 2;
    else if (step < -Math.PI) step += Math.PI * 2;
    hotspotDrag.lastAngle = angle;
    // Clamped to the travel still left in the knob, so turning past an end
    // stops there rather than banking rotation that has to be unwound
    // before the level moves back.
    const lowest = -(hotspotDrag.startValue / 100) * VOLUME_SWEEP_RADIANS;
    const highest = ((100 - hotspotDrag.startValue) / 100) * VOLUME_SWEEP_RADIANS;
    hotspotDrag.turned = clamp(hotspotDrag.turned + step, lowest, highest);
    setVolume(
      hotspotDrag.startValue + (hotspotDrag.turned / VOLUME_SWEEP_RADIANS) * 100,
    );
  });

  const finish = (event) => {
    if (hotspotDrag === null || (hotspotDrag.kind === "volume") === false) return;
    if ((hotspotDrag.pointerId === event.pointerId) === false) return;
    if (volumeHotspot.hasPointerCapture(event.pointerId)) volumeHotspot.releasePointerCapture(event.pointerId);
    hotspotDrag = null;
  };
  volumeHotspot.addEventListener("pointerup", finish);
  volumeHotspot.addEventListener("pointercancel", finish);
}

function installHotspotAccessibility() {
  const seekHotspot = hotspotFor("seek");
  const volumeHotspot = hotspotFor("volume");
  if (seekHotspot) installSeekHotspot(seekHotspot);
  if (volumeHotspot) installVolumeHotspot(volumeHotspot);

  // The transport controls are real buttons laid over the parts of the
  // object they operate, so one handler serves a pointer, Enter, Space and
  // a screen reader alike. The object plays its own press afterwards.
  elements.machineHotspots
    .querySelectorAll("button[data-machine-action]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        if (button.getAttribute("aria-disabled") === "true") return;
        handleMachineAction(button.dataset.machineAction);
      });
    });
}

function isInteractiveTarget(target) {
  if (target instanceof Element === false) return false;
  return target.closest(
    "button, input, select, textarea, a, iframe, [contenteditable], [role=button], [role=slider]",
  ) === null ? false : true;
}

function installKeyboardShortcut() {
  document.addEventListener("keydown", (event) => {
    if ((event.code === "Space") === false || event.repeat || isInteractiveTarget(event.target)) return;
    event.preventDefault();
    handleMachineAction("toggle");
  });
}

function installDebugInterface() {
  const debug = {
    get currentIndex() {
      return currentIndex;
    },
    get currentSelection() {
      const video = selectedVideo();
      return video === null
        ? null
        : Object.freeze({ index: currentIndex, video: Object.freeze({ ...video }) });
    },
    get connected() {
      return connected;
    },
    get playbackState() {
      return Object.freeze({ ...playback });
    },
    get outputLevel() {
      return Object.freeze({ ...outputLevel });
    },
    get assistantState() {
      return assistantState;
    },
    get recorderReady() {
      return machineReady;
    },
    get fallbackVisible() {
      return elements.fallbackControls.hidden === false;
    },
    apply(message) {
      applyMessage(message);
    },
    select(index) {
      return selectVideo(index, "debug");
    },
    toggle() {
      handleMachineAction("toggle");
    },
    stop() {
      handleMachineAction("stop");
    },
    mute() {
      handleMachineAction("mute");
    },
    previous() {
      handleMachineAction("previous");
    },
    next() {
      handleMachineAction("next");
    },
    seek(ratio) {
      seekToRatio(ratio);
    },
    setVolume: (value) => setVolume(value),
  };
  Object.defineProperty(window, "__onkithPlayerDebug", {
    value: Object.freeze(debug),
    configurable: false,
    enumerable: false,
    writable: false,
  });
}

function destroy() {
  stopTimeline();
  if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
  if (machine && typeof machine.destroy === "function") machine.destroy();
  if (socket !== null) {
    try {
      socket.close();
    } catch (error) {
      // The page is going away; a failed close needs no recovery.
    }
  }
}

function installIdlePointer() {
  // An appliance screen should not show a pointer nobody is using, but
  // hiding it outright makes the window impossible to review. Fade it out
  // when it stops moving and bring it straight back when it does.
  const root = document.documentElement;
  let idleTimer = null;
  const markActive = () => {
    root.classList.remove("pointer-idle");
    if (idleTimer !== null) window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(() => {
      root.classList.add("pointer-idle");
    }, POINTER_IDLE_MS);
  };
  ["pointermove", "pointerdown", "wheel", "keydown"].forEach((type) => {
    window.addEventListener(type, markActive, { passive: true });
  });
  markActive();
}

function bootstrap() {
  if (new URLSearchParams(window.location.search).get("kiosk") === "1") {
    document.documentElement.classList.add("is-kiosk");
    installIdlePointer();
  }

  buildTrackFlow();
  setCurrentMetadata();
  resetTimeline();
  syncVolumeInterface();
  updateControlAvailability();
  installFallbackControls();
  installHotspotAccessibility();
  installKeyboardShortcut();
  installDebugInterface();
  refreshConnection();
  setStatus("Connecting to Onkith.");

  window.addEventListener("resize", () => centerActiveCard("auto"));
  window.addEventListener("beforeunload", destroy, { once: true });

  startTimeline();
  connect();
  void initialiseMachine();
}

bootstrap();
