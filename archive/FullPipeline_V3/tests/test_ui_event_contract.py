"""The UI must survive the events the segmenter actually emits.

The live UI previously indexed ``data["start_threshold"]`` on
``calibration_complete``, but the segmenter emits ``energy_gate``. The existing
UI tests fed a hand-written payload containing the wrong key, so they passed
while a real session aborted with ``KeyError: 'start_threshold'`` immediately
after calibration. These tests drive the UI with payloads captured from the
real segmenter instead of invented ones.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.canary_live_segmenter import CanaryLiveSegmenter
from app.terminal_ui import TerminalUI, _calibration_threshold
from integrated_pipeline.config import CALIBRATION_SAMPLES, PipelineConfig


class RecordingLiveUI:
    """Stand-in with the same surface the real LiveUI exposes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def ready(self, threshold=None):
        self.calls.append(("ready", threshold))

    def set_state(self, state, detail=None):
        self.calls.append(("set_state", state))

    def info(self, message):
        self.calls.append(("info", message))

    def warn(self, message):
        self.calls.append(("warn", message))

    def error(self, message):
        self.calls.append(("error", message))

    def notice(self, message):
        self.calls.append(("notice", message))


class SilentVAD:
    def probability(self, frame):
        return 0.0

    def reset(self):
        return None


class _SpeechThenSilenceVAD:
    """Speech whenever the frame is loud, so a real session is produced."""

    def probability(self, frame):
        return 0.9 if float(np.max(np.abs(frame))) > 0.5 else 0.1

    def reset(self):
        return None


def real_calibration_event() -> dict:
    """Capture the payload the segmenter really emits, not an invented one."""

    events: list[tuple[str, dict]] = []
    segmenter = CanaryLiveSegmenter(
        PipelineConfig(),
        "ui-contract",
        SilentVAD(),
        on_event=lambda name, payload: events.append((name, payload)),
        on_segment=lambda segment: None,
    )
    rng = np.random.default_rng(20260805)
    segmenter.process_samples(
        rng.normal(0.0, 0.01, CALIBRATION_SAMPLES + 512).astype(np.float32)
    )
    for name, payload in events:
        if name == "calibration_complete":
            return payload
    raise AssertionError("segmenter emitted no calibration_complete event")


class CalibrationEventContractTests(unittest.TestCase):
    def test_segmenter_payload_has_energy_gate_not_start_threshold(self) -> None:
        payload = real_calibration_event()
        self.assertIn("energy_gate", payload)
        self.assertNotIn("start_threshold", payload)

    def test_live_ui_survives_the_real_payload(self) -> None:
        payload = real_calibration_event()
        live = RecordingLiveUI()
        ui = TerminalUI(live_ui=live)
        # This used to raise KeyError and abort the session.
        ui.event("calibration_complete", payload)
        self.assertEqual(live.calls[-1][0], "ready")
        self.assertAlmostEqual(
            float(live.calls[-1][1]), float(payload["energy_gate"])
        )

    def test_plain_terminal_survives_the_real_payload(self) -> None:
        payload = real_calibration_event()
        ui = TerminalUI(quiet=True)
        ui.event("calibration_complete", payload)  # must not raise

    def test_every_real_segmenter_event_survives_both_ui_modes(self) -> None:
        """Replay a whole real session's events through the UI.

        This is the check that was missing: the UI used to index payload keys
        the segmenter never emits (``next_sequence_number``, ``batch_id`` and
        ``captured_audio_seconds`` on segment_closed, ``overlap_seconds``), so
        a live run crashed on the first utterance.
        """

        events: list[tuple[str, dict]] = []
        vad = _SpeechThenSilenceVAD()
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "ui-contract",
            vad,
            on_event=lambda name, payload: events.append((name, payload)),
            on_segment=lambda segment: None,
        )
        signal = np.concatenate(
            (
                np.zeros(CALIBRATION_SAMPLES, dtype=np.float32),
                np.ones(600 * 512, dtype=np.float32),
                np.zeros(120 * 512, dtype=np.float32),
            )
        )
        for offset in range(0, len(signal), 1_600):
            segmenter.process_samples(signal[offset : offset + 1_600])
        segmenter.shutdown(reason="end_of_stream")

        names = {name for name, _ in events}
        # The session must actually have produced the interesting events.
        for required in (
            "calibration_complete",
            "recording_started",
            "segment_closed",
            "forced_split_continuation",
        ):
            self.assertIn(required, names)

        for live_ui in (RecordingLiveUI(), None):
            ui = TerminalUI(live_ui=live_ui) if live_ui else TerminalUI(quiet=True)
            for name, payload in events:
                with self.subTest(event=name, live=live_ui is not None):
                    ui.event(name, payload)  # must never raise

    def test_threshold_reader_handles_both_keys_and_neither(self) -> None:
        self.assertAlmostEqual(_calibration_threshold({"energy_gate": 0.25}), 0.25)
        self.assertAlmostEqual(
            _calibration_threshold({"start_threshold": 0.017}), 0.017
        )
        # energy_gate wins when both are present.
        self.assertAlmostEqual(
            _calibration_threshold({"energy_gate": 0.3, "start_threshold": 0.9}), 0.3
        )
        self.assertEqual(_calibration_threshold({}), 0.0)
        self.assertEqual(_calibration_threshold({"energy_gate": None}), 0.0)


if __name__ == "__main__":
    unittest.main()
