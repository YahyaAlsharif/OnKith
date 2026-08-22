"""Release gates from the specification that no other module already covers."""

from __future__ import annotations

import queue
import sys
import threading
import types
import unittest
import zipfile
from pathlib import Path

import numpy as np

from app.canary_live_segmenter import CanaryLiveSegmenter
from app.live_runtime import LivePrivacyRuntime
from integrated_pipeline.canary_adapter import CanaryAdapter
from integrated_pipeline.config import (
    CALIBRATION_SAMPLES,
    OPERATIONAL_AUDIO_START_SAMPLE,
    PipelineConfig,
)
from scripts.build_release_zip import (
    ARCHIVE_PREFIX,
    InvalidMemberName,
    collect,
    is_excluded,
    validate_member_name,
)
from tests.helpers import canary_segment
from tests.test_canary_adapter import TransformerAEDBeamInfer
from tests.test_forced_split_parity import ScriptedVAD, build, online_run, speech_script

ROOT = Path(__file__).resolve().parents[1]


class SileroLifecycleTests(unittest.TestCase):
    def test_silero_resets_only_when_the_acoustic_utterance_ends(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (60 * 512, True),
            (110 * 512, False),
            (60 * 512, True),
            (110 * 512, False),
        )
        script = speech_script(signal, speaking)
        outputs, vad, _ = online_run(
            signal, script, [1_600] * 600, shutdown_reason=None
        )
        finals = [item for item in outputs if item.is_final_segment_of_utterance]
        self.assertEqual(len(finals), 2)
        # Exactly one reset per completed acoustic utterance, never mid-utterance.
        self.assertEqual(vad.reset_count, 2)

    def test_shutdown_resets_silero_even_with_no_active_utterance(self) -> None:
        vad = ScriptedVAD([0.0] * 200)
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "session",
            vad,
            on_event=lambda event, payload: None,
            on_segment=lambda segment: None,
        )
        segmenter.process_samples(np.zeros(CALIBRATION_SAMPLES + 4 * 512, dtype=np.float32))
        self.assertEqual(vad.reset_count, 0)
        segmenter.shutdown()
        self.assertEqual(vad.reset_count, 1)


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_canary_inference_is_refused(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingModel:
            def __init__(self) -> None:
                self.cfg = types.SimpleNamespace(
                    decoding={
                        "strategy": "greedy",
                        "temperature": 0.5,
                        "beam": {
                            "beam_size": 1,
                            "return_best_hypothesis": False,
                            "max_generation_delta": 1,
                        },
                    }
                )
                self.calls = 0

            def change_decoding_strategy(self, decoding: object) -> None:
                self.cfg.decoding = decoding
                self.decoding = types.SimpleNamespace(
                    cfg=decoding,
                    decoding=TransformerAEDBeamInfer(),
                )

            def transcribe(self, *, audio, override_config):
                self.calls += 1
                if self.calls > 1:  # not the warmup
                    entered.set()
                    release.wait(timeout=3)
                return [types.SimpleNamespace(text="blocking")]

        model = BlockingModel()
        adapter = CanaryAdapter(
            PipelineConfig(),
            model_loader=lambda path: model,
            transcription_config_factory=lambda **kwargs: types.SimpleNamespace(
                **kwargs
            ),
            verify_artifacts=False,
            verify_versions=False,
        )
        adapter.load()
        errors: list[str] = []

        def first() -> None:
            adapter.transcribe_segment(canary_segment(0, final=True))

        def second() -> None:
            try:
                adapter.transcribe_segment(canary_segment(0, final=True, utterance=2))
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=first)
        thread.start()
        self.assertTrue(entered.wait(timeout=2))
        second()
        release.set()
        thread.join(timeout=3)
        self.assertEqual(len(errors), 1)
        self.assertIn("Concurrent Canary inference is forbidden", errors[0])
        self.assertEqual(adapter.counters.concurrent_inference_rejections, 1)
        adapter.close()


class Stage5Tests(unittest.TestCase):
    def test_live_stage5_must_be_immediate(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.stage5["mode"], "immediate")

        class Rejecting(dict):
            pass

        broken = PipelineConfig()
        broken.data = dict(broken.data)
        broken.data["stage5"] = {"mode": "buffered"}
        with self.assertRaisesRegex(ValueError, "immediate"):
            LivePrivacyRuntime(
                broken,
                masker=object(),
                storage=object(),
                on_event=lambda event, payload: None,
                on_output=lambda output: None,
            )

    def test_inactive_three_second_buffer_is_never_instantiated(self) -> None:
        import inspect

        source = inspect.getsource(LivePrivacyRuntime)
        self.assertNotIn("WallClockBufferWorker(", source)
        self.assertNotIn("InactivityTranscriptBuffer(", source)


class OverflowReliabilityTests(unittest.TestCase):
    def test_queue_overflow_marks_the_final_logical_utterance_unreliable(self) -> None:
        from app.canary_stt_runtime import LiveSTTRuntime
        from integrated_pipeline.schemas import SegmentTranscript

        class Echo:
            def transcribe_segment(self, segment):
                return SegmentTranscript(
                    segment=segment,
                    text=f"word{segment.segment_index}",
                    processing_seconds=0.0,
                    queue_wait_seconds=0.0,
                    stt_started_at_utc="2026-08-05T10:00:00Z",
                    stt_completed_at_utc="2026-08-05T10:00:01Z",
                )

        class Privacy:
            def __init__(self) -> None:
                self.results = []

            def submit_stt(self, result) -> None:
                self.results.append(result)

            def submit_stt_failure(self, *args) -> None:
                raise AssertionError(args)

        completed: queue.Queue = queue.Queue(maxsize=4)
        privacy = Privacy()
        worker = LiveSTTRuntime(
            completed,
            Echo(),
            privacy,
            on_event=lambda event, payload: None,
            utc_now=lambda: "2026-08-05T10:00:03Z",
        )
        worker.start()
        # A dropped chunk during the utterance marks that segment unreliable;
        # the merged logical utterance must inherit it.
        completed.put(canary_segment(0, final=False, start=0, end=141_824))
        completed.put(
            canary_segment(
                1, final=True, start=130_560, end=180_000, reliable=False, dropped=2
            )
        )
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)
        self.assertFalse(privacy.results[0].reliable)
        self.assertEqual(privacy.results[0].dropped_audio_chunks, 2)


class ArchivePathTests(unittest.TestCase):
    """The Linux release must contain only relative POSIX member paths."""

    def test_backslash_absolute_and_traversal_are_rejected(self) -> None:
        for bad in (
            "Omit_AI_RPi_Live\\app\\live_audio.py",
            "/etc/passwd",
            "C:/Windows/system32",
            "Omit_AI_RPi_Live/../escape.py",
        ):
            with self.assertRaises(InvalidMemberName):
                validate_member_name(bad)

    def test_valid_posix_member_is_accepted(self) -> None:
        name = "Omit_AI_RPi_Live/app/canary_live_segmenter.py"
        self.assertEqual(validate_member_name(name), name)

    def test_caches_databases_and_secrets_are_excluded(self) -> None:
        for path in (
            Path("app/__pycache__/live_audio.cpython-311.pyc"),
            Path("session_transcript.db"),
            Path(".env"),
            Path("models/canary/canary-180m-flash.nemo"),
            Path("run.log"),
        ):
            self.assertTrue(is_excluded(path), f"{path} should be excluded")
        self.assertFalse(is_excluded(Path("app/live_audio.py")))
        self.assertFalse(is_excluded(Path("models/silero/silero_vad.onnx")))

    def test_every_collected_source_file_yields_a_posix_member(self) -> None:
        files = collect(ROOT)
        self.assertTrue(files)
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            name = validate_member_name(f"{ARCHIVE_PREFIX}/{relative}")
            self.assertNotIn("\\", name)

    def test_built_release_archive_uses_posix_paths_when_present(self) -> None:
        candidates = sorted(ROOT.parent.glob("*_FIXED.zip"))
        if not candidates:
            self.skipTest("release archive not built yet")
        with zipfile.ZipFile(candidates[-1]) as archive:
            self.assertIsNone(archive.testzip())
            for info in archive.infolist():
                self.assertNotIn("\\", info.filename)
                validate_member_name(info.filename)


class OperationalBoundaryTests(unittest.TestCase):
    def test_operational_start_equals_calibration_samples(self) -> None:
        self.assertEqual(OPERATIONAL_AUDIO_START_SAMPLE, 12_000)
        self.assertEqual(OPERATIONAL_AUDIO_START_SAMPLE, CALIBRATION_SAMPLES)


if __name__ == "__main__":
    unittest.main()
