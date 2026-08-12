# Standalone player design QA

## Source visual truth

- Primary recorder target: `qa/references/supplied-recorder-target.png`, 1118 x 1306 px.
- User feedback captures: the four screenshots supplied on 2026-08-02 at 18:27 and 18:28. They show the gray recorder panel, right-facing perspective, hover-only labels, and oversized translucent control targets that this pass corrects.
- Latest layout target: the user's 2026-08-02 request to remove the physical-transport heading, description, visible machine status, and machine instruction, then place the active title and artist above the rail and the single timeline directly beneath the active video. `qa/v3/player-mobile-recorder-final.png` is the visual before-state for that request.
- Physical source asset: `assets/player/tape-recorder-v1.glb`, supplied by the user.
- The source image is a Spline editor view, while the implementation evidence is the recorder inside its playback stage. The comparison therefore judges object treatment, framing, material readability, direct controls, and product hierarchy rather than editor chrome.

## Implementation evidence

- Desktop player: `qa/v2/player-desktop.png`, 1280 x 720 px.
- Desktop recorder: `qa/v2/player-recorder.png`, 1280 x 720 px.
- Mobile player: `qa/v2/player-mobile.png`, 1280 x 848 px containing a 360 x 800 CSS px player frame.
- Mobile recorder: `qa/v2/player-mobile-recorder.png`, 1280 x 720 px containing a 360 px wide player frame.
- Side by side focused comparison: `qa/v2/reference-comparison.png`, 1680 x 980 px.
- Feedback-pass desktop section: `qa/v3/player-desktop-section-final.png`, 1280 x 720 px.
- Feedback-pass desktop recorder: `qa/v3/player-desktop-recorder-final.png`, 1280 x 720 px.
- Feedback-pass selected side card: `qa/v3/player-desktop-card-selected.png`, 1280 x 720 px.
- Feedback-pass mobile top: `qa/v3/player-mobile-top.png`, 360 x 800 px.
- Feedback-pass mobile recorder: `qa/v3/player-mobile-recorder-final.png`, 360 x 800 px.
- Feedback comparison: `qa/v3/user-feedback-comparison.png`, 2000 x 1420 px. The left column contains the user captures and the right column contains the corrected implementation at matching states.
- Tightened desktop player: `qa/v4/player-desktop-tight.png`, 1440 x 900 px at a 1440 x 900 CSS px viewport.
- Tightened desktop comparison frame: `qa/v4/player-desktop-1280x720.png`, 1280 x 720 px at a 1280 x 720 CSS px viewport.
- Tightened mobile full view: `qa/v4/player-mobile-tight.png`, 360 px wide at a 360 x 800 CSS px viewport.
- Tightened mobile machine state: `qa/v4/player-mobile-machine-state.png`, 360 x 800 px.
- Tightened mobile recovery guidance: `qa/v4/player-mobile-error.png`, 360 x 800 px.
- Latest focused comparison: `qa/v4/user-request-comparison.png`, 720 x 800 px. The left side is the earlier 360 x 800 machine state and the right side is the tightened state at the same viewport and density.
- Desktop CSS viewport: 1280 x 720 px.
- Mobile CSS viewport: 360 x 800 px. The player measured 360 px for both client width and scroll width. The recorder stage measured 328 px wide.
- Browser device pixel ratio: 2. Browser screenshots were normalized to CSS pixel dimensions by the browser surface.
- Latest layout pass device pixel ratio: 1. The v4 evidence files match their CSS pixel dimensions directly.
- State: dark theme, recorder ready, deterministic YouTube QA fixture playing, and side-card selection exercised. The production player path was also checked with the real YouTube API.

## Findings

- No actionable P0, P1, or P2 findings remain.
- Fonts and typography: local Archivo and IBM Plex Mono assets provide the intended restrained display hierarchy, clear metadata, tabular time values, and compact technical labels. The heading wraps cleanly at 360 px and track titles truncate without breaking the rail.
- Spacing and layout rhythm: the active video stays centered between four real suggestions. The recorder is geometrically centered, shown front-on, and sits directly on the page canvas rather than inside a contrasting panel. The 360 px layout keeps the recorder and all labels visible without horizontal clipping.
- Latest spacing and layout rhythm: the timeline matches the active card width at every breakpoint. Browser measurements show 24 px from the active media to the timeline and 8 px from the timeline to the recorder at both 1440 px and 360 px widths. The recorder begins at 747 px in a 900 px desktop viewport and at 598 px in an 800 px mobile viewport, so the physical object is already visible without passing through a second heading or status region.
- Colors and visual tokens: every interface color comes from the Onkith tokens. The machine stage and body both resolve to `rgb(8, 7, 6)`, and the stage border resolves to `0px`. Warm black, neutral ink, and one vermillion accent carry the complete interface. No gradient, glow, CSS shadow, or second accent hue was introduced.
- Image quality and asset fidelity: the recorder is the supplied GLB, not raster art or a generated image. Runtime PBR materials use local CC0 scans. Cards use the exact verified YouTube thumbnails, titles, and channels. The brushed reel and powder-coated enclosure remain readable at desktop and mobile sizes.
- Copy and content: the selected title and channel now occupy the primary heading and update on every card selection. The physical-transport marketing copy, visible machine status panel, machine instruction, duplicate track details, and output label are removed. Routine playback messages remain in a visually hidden live region. Actionable failures use a separate visible alert that remains present until playback successfully recovers.
- Icons and controls: the Onkith favicon is the real local brand asset. Physical controls come from the supplied model geometry. Persistent labels identify Play or Pause, Stop, Mute, Previous, Next, Seek, and Volume before hover, while the semantic hotspots retain 64 x 64 px targets and the reel keeps a 96 x 96 px target.
- Interaction states: model Play, pause, Stop, mute, volume, seek, previous, and next controls were exercised. Both sides of the music rail were clicked with real pointer input and changed the selected track. Drag capture begins only after the movement threshold, so a normal card press is no longer redirected to the rail. Automatic unplayable-result fallback, autoplay recovery, missing-model fallback, focus semantics, reduced motion code paths, and dynamic tab titles were checked. Buffering suspends the startup watchdog, and the WebGL renderer sleeps while the recorder is idle or hidden.
- Responsiveness and accessibility: the 360 px player has zero document overflow. Keyboard volume changed from 82 to 86, keyboard seeking advanced the live timecode, and projected controls expose labels, pressed states, disabled states, and slider values. The fallback transport remains readable without WebGL.

## Comparison history

### Iteration 1

- Earlier evidence: `qa/v2/player-recorder-before.png` and `qa/v2/player-mobile-recorder-before.png`.
- P2 framing: the supplied recorder read too small in the stage, especially at 360 px, and the dark finish lost some material definition.
- Fix: the fitted camera now uses a tighter responsive framing margin and a slightly higher neutral exposure. The result enlarges the real object without changing its geometry or adding visual effects.
- P2 collision: the desktop machine status surface touched the top edge of the object.
- Fix: its maximum width was reduced from 28 rem to 22 rem so the readout and recorder remain separate.
- Post-fix evidence: `qa/v2/player-recorder.png`, `qa/v2/player-mobile-recorder.png`, and `qa/v2/reference-comparison.png`.
- Post-fix result: the object has a stronger physical presence, the reel texture is readable, the status panel no longer overlaps it, and no new P0, P1, or P2 issue is visible.

### Iteration 2

- Earlier evidence: the four user screenshots supplied on 2026-08-02 at 18:27 and 18:28.
- P1 interaction: the rail captured every mouse pointer on pointerdown, so ordinary presses on side suggestions could finish on the rail instead of the card button.
- Fix: pointer capture now starts only after movement crosses the drag threshold, and window-level pointer completion clears presses released outside the rail. A deterministic browser fixture confirmed that selecting result 2 changed the title to `Never Gonna Give You Up` and the readout to `Track 02 / 05`. Left and right suggestion presses were both exercised.
- P2 surface separation: the machine stage used `--surface` plus a border, creating a gray card inside the black page.
- Fix: the stage now uses `--void`, no border, and no corner treatment. Computed stage and body backgrounds match exactly.
- P2 composition: the camera vector showed the recorder from the upper right and pointer movement added more yaw.
- Fix: the camera is now front-on at `(0, 0, 1)`, pointer parallax is disabled, and responsive framing keeps the object centered with room for its control labels.
- P2 affordance: control labels appeared only on hover and large translucent circles obscured the hardware.
- Fix: labels remain visible at rest, Previous, Next, and Volume labels sit beside their controls, disabled actions are subdued, and the circular wash appears only for hover or keyboard focus.
- Post-fix evidence: `qa/v3/player-desktop-section-final.png`, `qa/v3/player-desktop-recorder-final.png`, `qa/v3/player-mobile-recorder-final.png`, and `qa/v3/user-feedback-comparison.png`.
- Post-fix result: the card rail selects reliably, the recorder reads as a centered front-facing object on one continuous canvas, and each control is named before interaction. No actionable P0, P1, or P2 issue remains.

### Iteration 3

- Earlier evidence: `qa/v3/player-desktop-section-final.png` and `qa/v3/player-mobile-recorder-final.png`.
- P2 scroll distance: a second headline, explanatory paragraph, status card, instruction card, duplicate metadata block, and wide timeline header delayed the recorder after the active video.
- Fix: the active song and artist replace the static carousel headline, the one timeline is centered directly beneath the active video, the duplicated metadata and output row are removed, and runtime status is retained only as an accessible live region.
- P2 vertical rhythm: the carousel reserved more height than its active media needed, leaving unnecessary space before the timeline.
- Fix: the rail height is reduced responsively, its bottom padding is reduced, and the recorder transition uses the existing `--s2` spacing token. The media-to-timeline gap measures 24 px and the timeline-to-recorder gap measures 8 px at the tested desktop and mobile widths.
- P2 recovery guidance: routine recorder readiness could replace an autoplay or source failure before the user acted on it.
- Fix: routine status and visible recovery guidance now use separate live regions. Recorder readiness no longer clears the visible alert, while successful playback clears it explicitly.
- Post-fix evidence: `qa/v4/player-desktop-tight.png`, `qa/v4/player-mobile-tight.png`, `qa/v4/player-mobile-machine-state.png`, and `qa/v4/user-request-comparison.png`.
- Post-fix result: the active title, video, timeline, and recorder read as one continuous player. The focused 360 x 800 comparison shows the removed copy and status surfaces no longer occupy the path to the recorder. No actionable P0, P1, or P2 issue remains.

## Primary interactions tested

- Five verified cards rendered with real titles, channels, thumbnails, and a centered active result.
- Selecting suggestions on both sides updated the visible video, metadata, track index, and `Music name \ Player \ Onkith` tab title. A deterministic fixture verified the exact selected result before publisher embed policy could trigger fallback.
- The latest rendered pass opened with `Sirens`, `Ludwig Göransson - Topic`, and `Track 01 / 05` in the primary heading. A side suggestion was exercised, and the timeline remained singular and in place below the active media.
- A 52 px rail drag preserved `Track 02 / 05` instead of selecting the dragged card, while a normal click selected it. Releasing outside the rail is covered by window-level pointer cleanup.
- Pressing the physical Play key changed the status to `Playing through your speakers.`, set the pressed state, advanced elapsed time, and activated reel motion from the YouTube state.
- Pressing Play again paused playback and stopped the active reel state.
- Keyboard volume changed from 82 to 86.
- Keyboard seek advanced the live position to 00:13 and updated the slider to 10 of 1000.
- Mute changed the physical control to Unmute.
- Stop reset the elapsed time to 00:00.
- A missing GLB revealed the conventional fallback transport while the visible YouTube player remained available.
- The live Adele search returned five verified results in the original order. The in-app browser rejected those particular embeds with YouTube error 150, and the player correctly advanced through the remaining results. A separate known-embeddable playback fixture verified live playback and every transport action.
- Desktop console errors checked: none from the player. The 360 px iframe harness logs one Browser instrumentation observer error in its outer QA shell; it is not emitted by the player source and does not affect the embedded player.
- Latest desktop and 360 px console checks reported no errors from the player, and neither viewport had horizontal document overflow. At 360 px, an autoplay alert remained visible after the recorder reached its ready state, then cleared only after the physical Play control started playback.

## Iteration 4: the page stopped owning audio

Every capture above, including the `qa/v4` files, predates the move of
playback into the core process. They show a player that embedded YouTube and
played through the page, and a control overlay whose labels sat below or
beside their controls. They are kept as the before-state and no longer
describe the current player. New captures are needed on the device.

What changed:

- Audio plays in a headless `mpv` child owned by the core, so the page is a
  surface only. There is no embed, no autoplay watchdog, and no per-search
  page: track data, playback state, and output levels arrive over a websocket.
- Control labels are centred on the part they operate instead of being pushed
  below or beside it, which is what put some labels off the device body.
- The overlay now depth tests against the mesh by raycasting from the camera,
  so a covered control hides its label instead of drawing it in front of the
  object.
- Control targets are sized from the projected part rather than a fixed 64 px.
- The volume controls turn about the axis derived from their own geometry. The
  previous fixed Z axis stood the top wheel on edge through the top face and
  swung the front knob into the panel rather than across it.

Not re-verified visually. The 3D scene needs a real display, and the capture
surface used for the earlier iterations does not run the render loop here, so
the recorder could not be photographed. The geometry claims above are checked
against the supplied GLB in `tests/model_geometry_check.mjs` instead. Confirm
the appearance on the Pi.

## Residual test gaps

- The recorder has no post-change visual evidence; the geometry is verified
  numerically against the model, not photographically.
- Native YouTube availability varies by video, region, and browser embed policy. The app handles player errors and advances to the next verified result, but no client can force a publisher-restricted video to embed.
- The source is an editor screenshot rather than a complete page mock, so exact screen-level pixel parity is not applicable. The supplied geometry, material direction, one-reel motion, and direct-control intent were compared in the focused evidence instead.

## Follow-up polish

- P3: the source editor view includes a blue display treatment. The implementation keeps the display neutral because the Onkith system permits only one accent hue. This is an intentional brand constraint.

final result: passed
