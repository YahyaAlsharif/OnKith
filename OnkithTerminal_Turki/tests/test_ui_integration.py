from __future__ import annotations

import inspect
import io
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from app.live_audio import LiveAudioRuntime
from app.terminal_ui import TerminalUI
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import (
    BufferOutput,
    LiveOutput,
    PIIResult,
    TranscriptFragment,
)
from onkith_ui import LiveUI
from tests.helpers import recording, stt_result


class TTYStream(io.StringIO):
    def __bool__(self) -> bool:
        return True

    def isatty(self) -> bool:
        return True


class FakeLiveUI:
    def __init__(self) -> None:
        self.calls = []

    def set_state(self, state, detail="", sticky=None):
        self.calls.append(("state", state, detail, sticky))

    def ready(self, threshold=None):
        self.calls.append(("ready", threshold))

    def utterance(self, text, entities=None, latency=None, ts=None):
        self.calls.append(("utterance", text, entities, latency))

    def onkith(self, text):
        self.calls.append(("onkith", text))

    def info(self, text, detail=None):
        self.calls.append(("info", text, detail))

    def ok(self, text, detail=None):
        self.calls.append(("ok", text, detail))

    def warn(self, text, detail=None):
        self.calls.append(("warn", text, detail))

    def error(self, text, detail=None):
        self.calls.append(("error", text, detail))


class LiveUILifecycleTests(unittest.TestCase):
    def test_plain_mode_has_no_ansi_or_paint_thread(self) -> None:
        stream = io.StringIO()
        with patch.dict(os.environ, {"OMIT_UI": "plain"}, clear=False):
            ui = LiveUI(stream=stream, motion=True)
            with ui:
                ui.banner(system={"platform": "test"})
                with ui.task("model", done_label="model ready"):
                    pass
                ui.ready(threshold=0.01)
                ui.set_level(0.02)
                ui.set_state("speech")
                ui.utterance(
                    "hello [GIVENNAME_1]",
                    entities={"GIVENNAME": 1},
                    latency={"end_to_end": 0.5, "rtf": 0.2},
                )
        self.assertIsNone(ui._thread)
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertIn("hello [GIVENNAME_1]", stream.getvalue())
        self.assertIn("You", stream.getvalue())
        self.assertNotIn("rtf", stream.getvalue())
        self.assertNotIn("0.5s", stream.getvalue())

    def test_turns_are_clean_and_onkith_uses_exact_orange(self) -> None:
        stream = TTYStream()
        with patch.dict(
            os.environ,
            {"COLORTERM": "truecolor"},
            clear=True,
        ):
            ui = LiveUI(stream=stream, motion=False)
            ui.start()
            ui.utterance(
                "hello [GIVENNAME_1]",
                entities={"GIVENNAME": 1},
                latency={"end_to_end": 0.5, "rtf": 0.2},
            )
            ui.onkith("Opening Dancing with Monsters")
            ui.stop()
        rendered = stream.getvalue()
        self.assertIn("You", rendered)
        self.assertIn("hello ", rendered)
        self.assertIn("[GIVENNAME_1]", rendered)
        self.assertIn("Onkith", rendered)
        self.assertIn("Opening Dancing with Monsters", rendered)
        self.assertNotIn("rtf 0.20", rendered)
        self.assertNotRegex(rendered, r"\d{2}:\d{2}:\d{2}")
        orange = "\x1b[38;2;255;90;54m"
        self.assertIn(orange, rendered)
        onkith_line = next(
            line for line in rendered.splitlines() if "Onkith" in line
        )
        self.assertGreaterEqual(onkith_line.count(orange), 2)

    def test_one_paint_thread_stops_and_repeated_stop_is_safe(self) -> None:
        stream = TTYStream()
        with patch.dict(
            os.environ,
            {"OMIT_UI_ASCII": "1", "OMIT_UI_MOTION": "1"},
            clear=True,
        ):
            ui = LiveUI(stream=stream, motion=True)
            ui.start()
            first_thread = ui._thread
            ui.start()
            self.assertIs(ui._thread, first_thread)
            self.assertIsNotNone(first_thread)
            self.assertTrue(first_thread.is_alive())
            self.assertTrue(first_thread.daemon)
            ui.stop()
            ui.stop()
        self.assertFalse(first_thread.is_alive())
        self.assertNotIn(first_thread, threading.enumerate())

    def test_context_stops_after_exception(self) -> None:
        stream = TTYStream()
        with patch.dict(os.environ, {}, clear=True):
            ui = LiveUI(stream=stream, motion=True)
            with self.assertRaisesRegex(RuntimeError, "expected"):
                with ui:
                    raise RuntimeError("expected")
        self.assertFalse(ui._active)
        if ui._thread is not None:
            self.assertFalse(ui._thread.is_alive())
        self.assertIn("\x1b[?25h", stream.getvalue())


class UIEventMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.live = FakeLiveUI()
        self.terminal = TerminalUI(live_ui=self.live)

    def test_existing_events_map_to_required_states(self) -> None:
        self.terminal.event(
            "calibration_complete",
            {"noise_rms": 0.001, "start_threshold": 0.017},
        )
        self.terminal.event(
            "recording_started", {"next_sequence_number": 1}
        )
        self.terminal.event(
            "segment_closed",
            {
                "batch_id": "batch-1",
                "reason": "endpoint_silence",
                "captured_audio_seconds": 1.0,
                "submitted_audio_seconds": 0.8,
                "reliable": True,
            },
        )
        self.terminal.event("stage5_submitted", {"batch_id": "batch-1"})
        self.terminal.event("masking_started", {"batch_id": "batch-1"})

        self.assertIn(("ready", 0.017), self.live.calls)
        self.assertIn(("state", "speech", "", None), self.live.calls)
        self.assertIn(
            ("state", "thinking", "transcribing", None),
            self.live.calls,
        )
        self.assertIn(
            ("state", "buffer", "immediate Stage 5", None),
            self.live.calls,
        )
        self.assertIn(
            ("state", "masking", "privacy classification", None),
            self.live.calls,
        )

    def test_after_ready_only_final_onkith_status_becomes_a_turn(self) -> None:
        self.terminal.event(
            "calibration_complete",
            {"noise_rms": 0.001, "start_threshold": 0.017},
        )
        self.live.calls.clear()
        self.terminal.event(
            "recording_started", {"next_sequence_number": 1}
        )
        self.terminal.event(
            "segment_closed",
            {
                "batch_id": "batch-1",
                "reason": "endpoint_silence",
                "captured_audio_seconds": 1.0,
                "submitted_audio_seconds": 0.8,
                "reliable": True,
            },
        )
        self.terminal.event(
            "onkith_command_detected",
            {
                "wake_word": "onkith",
                "command_text": "play dancing with the monster",
                "intent": "PLAY",
            },
        )
        self.terminal.event(
            "onkith_command_queued", {"intent": "PLAY"}
        )
        self.terminal.event(
            "onkith_status",
            {"message": 'Searching YouTube for: "dancing monster song"'},
        )
        self.terminal.event(
            "onkith_status",
            {"message": "Opening: Dancing with Monsters"},
        )

        visible = [
            call for call in self.live.calls
            if call[0] in {"info", "ok", "warn", "onkith"}
        ]
        self.assertEqual(
            visible,
            [("onkith", "Opening Dancing with Monsters")],
        )

    def test_why_the_assistant_failed_reaches_the_screen(self) -> None:
        """The room is told nothing came back; the operator needs more.

        Every status message the conversation does not recognise is
        dropped, which is right for the chatty ones and was wrong for
        this one. A decommissioned model name produced a device that
        looked offline, and the reason existed only in a web dashboard.
        """
        self.terminal.event(
            "calibration_complete",
            {"noise_rms": 0.001, "start_threshold": 0.017},
        )
        self.live.calls.clear()
        self.terminal.event(
            "onkith_status",
            {
                "message": (
                    "Assistant reply failed: ModelGone: Groq does not have "
                    "a model called llama-3.1-8b-instant."
                )
            },
        )
        warnings = [call for call in self.live.calls if call[0] == "warn"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("llama-3.1-8b-instant", warnings[0][1])

    def test_final_ui_receives_exact_masked_text_entities_and_latency(self):
        item = recording(1)
        stt = stt_result(
            item,
            "Mohammed spoke",
            processing_seconds=0.2,
            finalization_seconds=0.1,
        )
        fragment = TranscriptFragment(
            fragment_id="fragment-1",
            text=stt.final_text,
            arrival_time_seconds=1.0,
            source_audio_id=item.batch_id,
            stt_result=stt,
            session_id=item.session_id,
            batch_id=item.batch_id,
            sequence_number=1,
        )
        buffer = BufferOutput(
            buffer_id=item.batch_id,
            text=stt.final_text,
            fragments=[fragment],
            started_at_seconds=1.0,
            flushed_at_seconds=1.0,
            flush_reason="finalized_recording",
            session_id=item.session_id,
            batch_id=item.batch_id,
            sequence_number=1,
        )
        output = LiveOutput(
            buffer=buffer,
            pii=PIIResult(
                original_text=stt.final_text,
                masked_text="[GIVENNAME_1] spoke",
                latency_seconds=0.04,
                entities=[
                    {
                        "entity_type": "GIVENNAME",
                        "placeholder": "[GIVENNAME_1]",
                    }
                ],
                safe_to_release=True,
                privacy_status="masked",
            ),
            total_seconds=0.8,
            mask_queue_wait_seconds=0.03,
        )

        terminal = TerminalUI(show_raw=True, live_ui=self.live)
        terminal.output(output)

        utterance = next(
            call for call in self.live.calls if call[0] == "utterance"
        )
        self.assertEqual(utterance[1], "[GIVENNAME_1] spoke")
        self.assertEqual(utterance[2], {"GIVENNAME": 1})
        self.assertEqual(utterance[3]["end_to_end"], 0.8)
        self.assertAlmostEqual(utterance[3]["rtf"], 0.3)
        self.assertEqual(utterance[3]["audio"], 1.0)
        self.assertEqual(utterance[3]["stt"], 0.2)
        self.assertEqual(utterance[3]["finalize"], 0.1)
        self.assertEqual(utterance[3]["buffer_wait"], 0.0)
        self.assertEqual(utterance[3]["pii_total"], 0.04)
        self.assertNotIn(
            stt.final_text,
            repr(
                [
                    call
                    for call in self.live.calls
                    if call[0] == "utterance"
                ]
            ),
        )
        self.assertFalse(
            any(
                call[0] == "info" and stt.final_text in call[1]
                for call in self.live.calls
            )
        )

    def test_json_output_bypasses_live_ui(self) -> None:
        terminal = TerminalUI(json_output=True, live_ui=self.live)
        with patch("builtins.print") as printed:
            terminal.event(
                "recording_started", {"next_sequence_number": 1}
            )
        self.assertEqual(self.live.calls, [])
        self.assertTrue(printed.called)


class AudioAndStartupHookTests(unittest.TestCase):
    def test_existing_rms_value_is_forwarded_without_terminal_output(self):
        levels = []
        runtime = LiveAudioRuntime(
            PipelineConfig(),
            "session-ui",
            device=1,
            once=False,
            on_event=lambda *args: None,
            on_level=levels.append,
            clock=lambda: 10.0,
        )
        runtime.sample_rate = 100
        samples = np.full((10, 1), 0.02, dtype=np.float32)
        runtime._audio_callback(samples, 10, None, None)
        queued = runtime.audio_chunk_queue.get_nowait()
        runtime.audio_chunk_queue.task_done()
        self.assertEqual(levels, [queued.rms])

        callback_source = inspect.getsource(LiveAudioRuntime._audio_callback)
        self.assertNotIn("print(", callback_source)
        self.assertNotIn("on_event(", callback_source)

    def test_quiet_boot_precedes_project_imports_and_one_ui_is_created(self):
        source = inspect.getsource(
            __import__(
                "app.omit_ai_live",
                fromlist=["main"],
            )
        )
        self.assertLess(
            source.find("quiet_boot()"),
            source.find("from app.live_audio import"),
        )
        self.assertEqual(source.count("LiveUI("), 1)
        self.assertIn("with muted_stderr():", source)


if __name__ == "__main__":
    unittest.main()
