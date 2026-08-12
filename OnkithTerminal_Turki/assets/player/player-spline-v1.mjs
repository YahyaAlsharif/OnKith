// The recorder is a Spline scene, served from this device rather than from
// Spline's CDN. Nothing here builds, shades, or lights the object: it
// arrives finished, with its own textures, contact shadows and printed
// symbols. What this module does is place the page's controls over it and
// reflect playback back onto it.

const SCENE_URL = "assets/spline/scene.splinecode";
const VIEWER_TAG = "spline-viewer";
const VIEWER_READY_TIMEOUT_MS = 25000;

// A pot's usable travel, and how far a finger drags to cross it.
// Exported because the hand and the knob have to agree on it: the page
// turns the level by the angle the pointer sweeps around the knob, and a
// full turn of the hand should be a full turn of the part.
export const VOLUME_SWEEP_RADIANS = Math.PI * 1.5;
// A record turns at 33 1/3 rpm, which is a revolution every 1.8 seconds:
// slow enough to read the label, fast enough to be unmistakably turning.
// It ran at a fraction of that while it was not turning at all on screen,
// which only made the standstill harder to see.
const PLATTER_RADIANS_PER_SECOND = (100 / 3 / 60) * Math.PI * 2;
// Dragging the platter to seek moves the whole track; the turn that goes
// with it is the same two revolutions the old reel made.
const PLATTER_SEEK_RADIANS = Math.PI * 4;

// The tilt the scene's own rocker uses, and how long a press takes to go
// down and come back up.
const KEY_PRESS_RADIANS = 0.113;
const KEY_PRESS_MS = 190;

// Named in the scene itself. The page's data-model-node attributes carry
// the same names, so the markup stays the one place a control is bound to
// a part of the object.
const PLATTER_NODE = "disk";
const VOLUME_KNOB_NODE = "bottom_knob_stick";

const MINIMUM_TARGET_PX = 44;

// The scene is published with its own fixed zoom, which frames it for the
// editor's window rather than for whatever box this page gives it. Left
// alone the object is simply cropped by the stage. This is how much of the
// stage it is allowed to fill once it has been sized to fit.
const FIT_MARGIN = 0.92;
// Below this the object is already the right size and setting the zoom
// again would only cost a reprojection.
const FIT_TOLERANCE = 0.005;

// How long to let the scene's own camera move finish before measuring
// against it, and how many quiet samples count as finished.
const CAMERA_SETTLE_TIMEOUT_MS = 6000;
// A camera that has stopped still dithers in the last bits of a double:
// the z it sits at is a number in the hundreds, so one unit in its last
// place is around a ten-trillionth, and the value walks a few of those
// every frame forever. Compared exactly, the camera never stops and the
// whole settle rides out its timeout. Real movement here is hundreds of
// units, so a thousandth of a unit sits ten orders of magnitude clear of
// the noise and nowhere near anything the eye could see.
const CAMERA_STILL_TOLERANCE = 0.001;
const CAMERA_SETTLE_INTERVAL_MS = 80;
const CAMERA_SETTLE_SAMPLES = 3;

export function sixteenByNineThumbnail(url) {
  // Only rewrite the filename of a known YouTube thumbnail; anything else
  // is passed through untouched.
  if (typeof url !== "string" || url.length === 0) return url;
  return url.replace(
    /^(https?:\/\/i\.ytimg\.com\/vi\/[A-Za-z0-9_-]{11}\/)[a-z0-9]+\.jpg(\?.*)?$/,
    (match, prefix, query) => `${prefix}mqdefault.jpg${query ?? ""}`,
  );
}

function statusReporter(onStatus) {
  return (message, level = "info") => {
    if (typeof onStatus !== "function") return;
    try {
      onStatus(message, level);
    } catch (error) {
      // A surface that cannot show status must not stop the object.
    }
  };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normaliseRatio(value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) === false) return 0;
  return clamp(numeric, 0, 1);
}

function playbackIsActive(state) {
  return state === "playing" || state === true;
}

async function waitForViewer(viewer) {
  if (typeof customElements === "undefined") {
    throw new Error("This browser has no custom element registry.");
  }
  await withTimeout(
    customElements.whenDefined(VIEWER_TAG),
    "The Spline viewer script did not load.",
  );
  // The element builds its application object as soon as it upgrades, well
  // before the scene inside it exists, so having one is not the same as
  // being ready. It announces itself with load-complete rather than load,
  // and sets _loaded either way, so the flag is what gets watched: a
  // renamed event then costs a few hundred milliseconds instead of the
  // whole object.
  await waitFor(
    () => viewer._loaded === true,
    "The recorder scene did not finish loading.",
  );
  if (!viewer._spline) throw new Error("The recorder scene reported no application.");
  return viewer._spline;
}

// Polled rather than driven by requestAnimationFrame, which a browser stops
// entirely for a tab that is not visible: the kiosk window is often started
// behind something else and has to finish loading anyway.
async function waitFor(isReady, message) {
  const deadline = Date.now() + VIEWER_READY_TIMEOUT_MS;
  while (isReady() === false) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
  }
}

function withTimeout(promise, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), VIEWER_READY_TIMEOUT_MS);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

// The viewer's own hit testing only fires for objects the scene author gave
// an interaction to, which is three keys and two knobs. The page needs
// seven controls, so the parts are projected to the screen and the page's
// own buttons are laid over them, exactly as they were over the old export.
function sceneInternals(application) {
  const scene = application._scene;
  const camera = application._camera;
  const canvas = application.canvas;
  const usable =
    scene && camera && canvas
    && typeof scene.traverse === "function"
    && camera.projectionMatrix
    && camera.matrixWorldInverse;
  if (!usable) return null;
  return { scene, camera, canvas };
}

// The scene animates its camera into place on load, and the controls are
// projected against wherever it stops. Watching the matrix rather than
// waiting a fixed time means a slow device gets the time it needs and a
// fast one is not held up.
async function settleCamera(internals) {
  const snapshot = () => [...internals.camera.matrixWorld.elements, internals.camera.zoom];
  const deadline = Date.now() + CAMERA_SETTLE_TIMEOUT_MS;
  let previous = snapshot();
  let stillFor = 0;
  while (stillFor < CAMERA_SETTLE_SAMPLES && Date.now() < deadline) {
    await new Promise((resolve) => {
      setTimeout(resolve, CAMERA_SETTLE_INTERVAL_MS);
    });
    const current = snapshot();
    stillFor = hasMoved(previous, current) ? 0 : stillFor + 1;
    previous = current;
  }
}

function hasMoved(before, after) {
  return before.some((value, index) => Math.abs(value - after[index]) > CAMERA_STILL_TOLERANCE);
}

function transform(elements, point) {
  return [
    elements[0] * point[0] + elements[4] * point[1] + elements[8] * point[2] + elements[12] * point[3],
    elements[1] * point[0] + elements[5] * point[1] + elements[9] * point[2] + elements[13] * point[3],
    elements[2] * point[0] + elements[6] * point[1] + elements[10] * point[2] + elements[14] * point[3],
    elements[3] * point[0] + elements[7] * point[1] + elements[11] * point[2] + elements[15] * point[3],
  ];
}

function findNode(scene, name) {
  let found = null;
  scene.traverse((node) => {
    if (found === null && node.name === name) found = node;
  });
  return found;
}

// Every object in the scene is loaded with its matrix update switched off:
// the runtime composes matrices itself, from its own copy of each
// transform. Writing a rotation straight onto a node therefore stores a
// number that nothing reads and nothing draws, which is why the platter and
// the knob could be measured climbing while the object stood still. The
// viewer's own handle for a part is what actually turns it.
//
// The handle is looked up by the node's id rather than by its name, so a
// name that appears twice cannot drive one part while the page measures
// another. It is a stable object, cached by the runtime, so holding onto it
// is the same as asking for it again.
function driveable(application, scene, name) {
  const node = findNode(scene, name);
  if (node === null) return null;
  const handle = application.findObjectById(node.uuid) || application.findObjectByName(name);
  return handle && handle.rotation ? handle : null;
}

// The camera is square on and orthographic, so a control's near and far
// faces project to the same rectangle and its whole box is its footprint.
// Nothing here needs to reason about depth, which is what makes this a
// straight measurement rather than a guess: at three quarters it took only
// the corners nearest the camera, and on a key that is the printed symbol
// rather than the cap it is printed on.
function projectedBounds(internals, node) {
  const { camera, canvas } = internals;
  const box = canvas.getBoundingClientRect();
  if (box.width === 0 || box.height === 0) return null;

  const points = [];
  node.updateWorldMatrix(true, true);
  node.traverse((mesh) => {
    const geometry = mesh.geometry;
    if (!geometry || mesh.visible === false) return;
    if (!geometry.boundingBox && typeof geometry.computeBoundingBox === "function") {
      geometry.computeBoundingBox();
    }
    const bounds = geometry.boundingBox;
    // The scene's text meshes carry an empty box, whose min is +Infinity
    // and max is -Infinity. One of those anywhere under a control turns the
    // whole measurement into NaN, and the control silently stops being
    // placed: it is why the platter and the two small keys had no box.
    if (!bounds || isFiniteBox(bounds) === false) return;
    const world = mesh.matrixWorld.elements;
    for (let corner = 0; corner < 8; corner += 1) {
      const local = [
        corner & 1 ? bounds.max.x : bounds.min.x,
        corner & 2 ? bounds.max.y : bounds.min.y,
        corner & 4 ? bounds.max.z : bounds.min.z,
        1,
      ];
      const view = transform(camera.matrixWorldInverse.elements, transform(world, local));
      const clip = transform(camera.projectionMatrix.elements, view);
      if (clip[3] === 0) continue;
      const x = box.left + ((clip[0] / clip[3]) * 0.5 + 0.5) * box.width;
      const y = box.top + ((-clip[1] / clip[3]) * 0.5 + 0.5) * box.height;
      if (Number.isFinite(x) === false || Number.isFinite(y) === false) continue;
      points.push({ x, y });
    }
  });
  if (points.length === 0) return null;

  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  points.forEach((point) => {
    minimumX = Math.min(minimumX, point.x);
    maximumX = Math.max(maximumX, point.x);
    minimumY = Math.min(minimumY, point.y);
    maximumY = Math.max(maximumY, point.y);
  });
  const measured = {
    centerX: (minimumX + maximumX) / 2,
    centerY: (minimumY + maximumY) / 2,
    width: maximumX - minimumX,
    height: maximumY - minimumY,
  };
  return Object.values(measured).every(Number.isFinite) ? measured : null;
}

function isFiniteBox(bounds) {
  return Number.isFinite(bounds.min.x) && Number.isFinite(bounds.min.y)
    && Number.isFinite(bounds.min.z) && Number.isFinite(bounds.max.x)
    && Number.isFinite(bounds.max.y) && Number.isFinite(bounds.max.z);
}

export async function createRecorderMachine({
  viewer,
  stage = viewer?.parentElement,
  hotspotRoot = stage,
  onStatus,
} = {}) {
  if (!(viewer instanceof HTMLElement) || viewer.tagName.toLowerCase() !== VIEWER_TAG) {
    throw new TypeError("A spline-viewer element is required for the recorder.");
  }
  if (!(stage instanceof HTMLElement)) {
    throw new TypeError("A stage element is required for the recorder.");
  }

  const reportStatus = statusReporter(onStatus);
  if (viewer.getAttribute("url") === null) viewer.setAttribute("url", SCENE_URL);

  const application = await waitForViewer(viewer);
  // The camera and the scene graph land during the same load, and on a slow
  // device the last of them can arrive a moment after the flag.
  await waitFor(
    () => sceneInternals(application) !== null,
    "The recorder scene did not expose a camera to place controls against.",
  );
  const internals = sceneInternals(application);

  hideAuthoringNotes(application);
  raiseLights(application);
  lockCamera(application);
  removeViewerChrome(viewer);

  const platter = driveable(application, internals.scene, PLATTER_NODE);
  const volumeKnob = driveable(application, internals.scene, VOLUME_KNOB_NODE);
  const platterRest = platter ? platter.rotation.z : 0;
  const volumeRest = volumeKnob ? volumeKnob.rotation.z : 0;

  const hotspots = Array.from(
    hotspotRoot instanceof HTMLElement
      ? hotspotRoot.querySelectorAll("[data-model-node]")
      : [],
  );

  let destroyed = false;
  let playing = false;
  let muted = false;
  let volume = 82;
  let progress = 0;
  let spinAngle = 0;
  let frame = null;
  let lastFrameTime = 0;
  const pressing = new Map();

  function applyVolume(value) {
    const next = clamp(Number(value), 0, 100);
    if (Number.isFinite(next) === false) return;
    volume = next;
    if (volumeKnob === null) return;
    // Counter-clockwise from rest as the level rises, which is the way this
    // knob reads on the object.
    volumeKnob.rotation.z = volumeRest - (volume / 100) * VOLUME_SWEEP_RADIANS;
  }

  // Where the platter sits is the turning it has done under its own power
  // plus where the track is. Keeping the second term means a scrub moves
  // it while it is playing too, rather than only when it is stopped. Both
  // are subtracted, which turns it clockwise: the way a record turns, and
  // the only way that reads as the track going forward.
  function applyPlatter() {
    if (platter === null) return;
    platter.rotation.z = platterRest - spinAngle - progress * PLATTER_SEEK_RADIANS;
  }

  // The scene animates its own key press, but the press it animates also
  // plays a click sample that is baked into the scene. This device makes
  // two sounds, the music and Onkith, so the press is done here instead:
  // the same tilt the scene's rocker uses, but momentary and silent.
  function pressKey(node) {
    if (node === null || node === undefined) return;
    const existing = pressing.get(node);
    if (existing) window.cancelAnimationFrame(existing.frame);
    const rest = existing ? existing.rest : node.rotation.x;
    const started = performance.now();
    const step = (now) => {
      if (destroyed) return;
      const through = Math.min(1, (now - started) / KEY_PRESS_MS);
      node.rotation.x = rest + Math.sin(through * Math.PI) * KEY_PRESS_RADIANS;
      if (through < 1) {
        pressing.set(node, { frame: window.requestAnimationFrame(step), rest });
        return;
      }
      node.rotation.x = rest;
      pressing.delete(node);
    };
    pressing.set(node, { frame: window.requestAnimationFrame(step), rest });
  }

  function step(now) {
    frame = null;
    if (destroyed) return;
    const elapsed = lastFrameTime === 0 ? 0 : Math.min(0.1, (now - lastFrameTime) / 1000);
    lastFrameTime = now;
    if (playing) {
      spinAngle += elapsed * PLATTER_RADIANS_PER_SECOND;
      applyPlatter();
      frame = window.requestAnimationFrame(step);
      return;
    }
    lastFrameTime = 0;
  }

  function startTurning() {
    if (destroyed || playing === false || frame !== null) return;
    lastFrameTime = 0;
    frame = window.requestAnimationFrame(step);
  }

  function stopTurning() {
    if (frame === null) return;
    window.cancelAnimationFrame(frame);
    frame = null;
    lastFrameTime = 0;
  }

  // The published scene carries a fixed zoom, so a stage that is a
  // different shape from the one it was framed in crops the object rather
  // than resizing it. Measure what is actually on screen and scale the
  // camera until the whole device is inside the box it was given.
  function fitToStage() {
    if (destroyed) return;
    const box = internals.canvas.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    const device = projectedBounds(internals, internals.scene);
    if (device === null || device.width <= 0 || device.height <= 0) return;
    const scale = Math.min(box.width / device.width, box.height / device.height) * FIT_MARGIN;
    if (Number.isFinite(scale) === false) return;
    if (Math.abs(scale - 1) < FIT_TOLERANCE) return;
    try {
      application.setZoom(internals.camera.zoom * scale);
    } catch (error) {
      // A viewer that will not take a zoom still draws; it is only cropped.
    }
  }

  function placeHotspots() {
    if (destroyed) return;
    const stageBox = stage.getBoundingClientRect();
    if (stageBox.width === 0 || stageBox.height === 0) return;

    const placements = [];
    hotspots.forEach((hotspot) => {
      const names = String(hotspot.dataset.modelNode || "").split(/\s+/).filter(Boolean);
      let placed = null;
      for (const name of names) {
        const node = findNode(internals.scene, name);
        if (node === null) continue;
        placed = projectedBounds(internals, node);
        if (placed !== null) break;
      }
      if (placed === null) {
        hotspot.hidden = true;
        return;
      }
      hotspot.hidden = false;
      const width = Math.max(MINIMUM_TARGET_PX, placed.width);
      const height = Math.max(MINIMUM_TARGET_PX, placed.height);
      hotspot.style.insetInlineStart = `${placed.centerX - stageBox.left}px`;
      hotspot.style.insetBlockStart = `${placed.centerY - stageBox.top}px`;
      hotspot.style.inlineSize = `${width}px`;
      hotspot.style.blockSize = `${height}px`;
      placements.push({ hotspot, area: width * height });
    });

    // The platter is most of the front of the device and the two small keys
    // beside it sit inside its box, so a control has to be stacked above
    // anything larger than itself or it cannot be clicked at all.
    placements.sort((left, right) => right.area - left.area);
    placements.forEach((placement, index) => {
      placement.hotspot.style.zIndex = String(index + 1);
    });
  }

  // A part is measured by projecting the corners of its bounding box, and
  // a disc's bounding box is a square, so the platter's corners sweep as it
  // turns: at 45 degrees they reach 28 per cent wider than the record
  // itself. Measured mid-turn, the seek ring is sized for an angle the
  // record has already left, and the camera fit behind it is dragged along
  // too, which is how a ring ended up wider than the whole device. The
  // turning parts are put back to rest for the measurement and returned
  // afterwards. Nothing can be drawn in between: no frame lands in the
  // middle of a synchronous pass.
  function measuredAtRest(measure) {
    const parked = [];
    if (platter !== null) parked.push([platter, platter.rotation.z, platterRest]);
    if (volumeKnob !== null) parked.push([volumeKnob, volumeKnob.rotation.z, volumeRest]);
    parked.forEach(([part, , rest]) => {
      part.rotation.z = rest;
    });
    try {
      return measure();
    } finally {
      parked.forEach(([part, turned]) => {
        part.rotation.z = turned;
      });
    }
  }

  // The viewer answers a resize on its own debounce, and recomputes its own
  // zoom when it does. Fitting on the same frame as the resize is simply
  // undone a moment later, so the fit waits for the camera to stop moving
  // and only then measures. The counter drops the results of a layout that
  // a newer one has already replaced.
  let layoutRun = 0;
  function layout() {
    if (destroyed) return;
    measuredAtRest(placeHotspots);
    const run = layoutRun + 1;
    layoutRun = run;
    settleCamera(internals).then(() => {
      if (destroyed || run !== layoutRun) return;
      // Square on first, then sized to the box, then measured. Turning the
      // camera changes what every control projects to, so it cannot come
      // after the placement.
      faceFront(internals, application);
      measuredAtRest(() => {
        fitToStage();
        placeHotspots();
      });
      // Square on, sized and measured: now it can be looked at.
      viewer.classList.add("is-ready");
    });
  }

  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => layout())
    : null;
  if (observer) observer.observe(stage);
  const onResize = () => layout();
  window.addEventListener("resize", onResize);

  // A device left running for a day can lose its WebGL context. Saying so
  // hands the page back to the controls that do not need one, instead of
  // leaving a black box with live hotspots over it.
  const onContextLoss = () => {
    reportStatus("The recorder lost its graphics context.", "error");
  };
  viewer.addEventListener("context-loss", onContextLoss);

  applyVolume(volume);
  applyPlatter();
  layout();
  reportStatus("The recorder is ready.");

  return {
    setPlaybackState(state) {
      const next = playbackIsActive(state);
      if (next === playing) return;
      playing = next;
      if (playing) startTurning();
      else stopTurning();
    },
    setMuted(value) {
      muted = Boolean(value);
    },
    setVolume(value) {
      applyVolume(value);
    },
    setProgress(value) {
      const next = normaliseRatio(value);
      if (Math.abs(next - progress) < 0.000001) return;
      progress = next;
      applyPlatter();
    },
    setArtwork() {
      // The record's own artwork is on the rail. The viewer exposes an
      // object's transform, visibility and colour, but not its texture
      // maps, so a per-track label cannot be printed onto this platter.
    },
    pulseControl(node) {
      if (typeof node !== "string" || node.length === 0) return;
      // A knob is turned rather than pressed, and its own control already
      // shows that; only the keys rock.
      const key = application.findObjectByName(node);
      if (key && typeof key.rotation === "object") pressKey(key);
    },
    controlReport() {
      return hotspots.map((hotspot) => ({
        action: hotspot.dataset.machineAction,
        node: hotspot.dataset.modelNode,
        hidden: hotspot.hidden === true,
        box: hotspot.hidden === true ? null : hotspot.getBoundingClientRect().toJSON(),
      }));
    },
    state() {
      return { playing, muted, volume, progress };
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      stopTurning();
      pressing.forEach((press) => window.cancelAnimationFrame(press.frame));
      pressing.clear();
      if (observer) observer.disconnect();
      window.removeEventListener("resize", onResize);
      viewer.removeEventListener("context-loss", onContextLoss);
      try {
        application.dispose();
      } catch (error) {
        // A viewer that is already gone needs no disposal.
      }
    },
  };
}

// The scene ships with an editing note reading "Click these buttons" and a
// leader line pointing at the transport keys. Both belong in the editor.
// They are the only objects at the root of the scene that are not the
// device, its camera or its lights, and the names inside the device repeat,
// so being at the root is what identifies them.
const AUTHORING_NOTE_NAMES = new Set(["Text", "Shape"]);

// The device is a matte near-black object on a near-black page, so its own
// light is the only thing separating it from the background: unlit, the
// silhouette, the deck and the keys all read as one black shape. The scene
// is composed for a bright editor canvas and carries three point lights,
// two at 1 and one at 4.
//
// Every light is raised by the same amount rather than only the dim two.
// The ratio between them is the modelling on the object, and lifting the
// fills to match the key would flatten exactly the surface this scene was
// chosen for. Four is where the body, the deck edge, the knob and the
// printed symbols all read while the device is still plainly black; past
// five or so the case starts to grey out and look like plastic.
const LIGHT_GAIN = 4;

function raiseLights(application) {
  if (typeof application.getAllObjects !== "function") return;
  application.getAllObjects().forEach((object) => {
    // Lights are what carry a brightness. Asking for the property rather
    // than for a list of light types means a scene that later swaps a point
    // light for a spot is still lit the same way.
    if (typeof object.intensity !== "number") return;
    object.intensity *= LIGHT_GAIN;
  });
}

// The viewer ships with orbit controls on, so a drag anywhere on the stage
// turns the object and a wheel over it zooms the object away instead of
// scrolling the page. On an appliance both are ways to lose the device and
// never find it again, and turning it would invalidate every control
// position, which are projected for a camera that does not move.
//
// The controller itself stays enabled: it is what runs the scene's own
// camera move on load and what the zoom goes through. Switching the whole
// thing off freezes the camera part way through that move and leaves the
// object at whatever angle it had reached.
function lockCamera(application) {
  const orbit = application.controls && application.controls.orbitControls;
  if (!orbit) return;
  orbit.enableRotate = false;
  orbit.enablePan = false;
  // Off, the wheel falls through to the page and the stage scrolls like
  // the rest of it. The fit still sets the zoom, which does not go through
  // this flag.
  orbit.enableZoom = false;
  orbit.autoRotate = false;
}

// The scene is composed at three quarters, which is a good photograph and a
// bad control panel: it foreshortens the keys, hides the right hand side of
// the deck, and skews every projected target away from the face it belongs
// to. The device is turned to face the room instead, square on, from the
// distance the scene already chose.
function faceFront(internals, application) {
  const orbit = application.controls && application.controls.orbitControls;
  const target = orbit && orbit.target;
  if (!target) return;
  const camera = internals.camera;
  const distance = Math.hypot(
    camera.position.x - target.x,
    camera.position.y - target.y,
    camera.position.z - target.z,
  );
  if (Number.isFinite(distance) === false || distance === 0) return;
  camera.position.set(target.x, target.y, target.z + distance);
  if (typeof camera.lookAt === "function") camera.lookAt(target.x, target.y, target.z);
  // Reading the camera back into the controller, so its own next update
  // does not put the object back where it was.
  if (typeof orbit.update === "function") orbit.update();
  if (typeof camera.updateMatrixWorld === "function") camera.updateMatrixWorld(true);
}

// The viewer stamps its own badge over the bottom right of the stage and
// offers to explain that the object can be dragged, which it no longer can.
function removeViewerChrome(viewer) {
  [viewer._logo, viewer._hintDrag].forEach((element) => {
    if (element && typeof element.remove === "function") element.remove();
  });
}

function hideAuthoringNotes(application) {
  let objects;
  try {
    objects = application.getAllObjects();
  } catch (error) {
    return;
  }
  if (Array.isArray(objects) === false) return;
  objects.forEach((object) => {
    if (AUTHORING_NOTE_NAMES.has(object.name) === false) return;
    const parent = object.parentUuid
      ? application.findObjectById(object.parentUuid)
      : null;
    if (parent) return;
    object.visible = false;
  });
}
