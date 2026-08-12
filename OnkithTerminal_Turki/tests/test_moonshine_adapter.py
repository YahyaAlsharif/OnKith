from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.moonshine_adapter import MoonshineAdapter

from tests.helpers import recording


class FakeLine:
    def __init__(self, text: str):
        self.id = 1
        self.text = text


class FakeEvent:
    def __init__(self, text: str):
        self.line = FakeLine(text)


class FakeStream:
    def __init__(self, text: str):
        self.text = text
        self.listeners = []
        self.add_calls = 0
        self.started = False

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def start(self):
        self.started = True

    def add_audio(self, samples, sample_rate):
        self.add_calls += 1
        self.listeners[0].on_line_completed(FakeEvent(self.text))

    def stop(self):
        self.started = False

    def close(self):
        pass


class FakeTranscriber:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.streams = []
        self.__class__.instances.append(self)

    def create_stream(self, **kwargs):
        stream = FakeStream(f"text-{len(self.streams) + 1}")
        self.streams.append(stream)
        return stream

    def close(self):
        pass


class FakeListener:
    pass


class MoonshineAdapterTests(unittest.TestCase):
    def test_resource_loads_once_and_stream_is_fresh_per_recording(self):
        medium = object()
        requested_arches = []

        class ModelArch:
            MEDIUM_STREAMING = medium

        def get_model(language, arch):
            requested_arches.append((language, arch))
            return "/cached/english-medium-streaming", medium

        module = types.SimpleNamespace(
            ModelArch=ModelArch,
            Transcriber=FakeTranscriber,
            TranscriptEventListener=FakeListener,
            get_model_for_language=get_model,
        )
        FakeTranscriber.instances.clear()
        adapter = MoonshineAdapter(PipelineConfig())
        with patch.dict(sys.modules, {"moonshine_voice": module}):
            with patch(
                "integrated_pipeline.moonshine_adapter.verify_package_version",
                return_value="0.0.69",
            ):
                adapter.load()
                adapter.load()
                first = adapter.transcribe_recording(recording(1))
                second = adapter.transcribe_recording(recording(2))
        self.assertEqual(requested_arches, [("en", medium)])
        self.assertEqual(adapter.counters.model_resource_load_count, 1)
        self.assertEqual(adapter.counters.utterance_session_count, 2)
        resource = FakeTranscriber.instances[0]
        self.assertEqual(len(resource.streams), 2)
        self.assertIsNot(resource.streams[0], resource.streams[1])
        self.assertEqual(resource.streams[0].add_calls, 1)
        self.assertEqual(resource.streams[1].add_calls, 1)
        self.assertEqual(first.final_text, "text-1")
        self.assertEqual(second.final_text, "text-2")
        self.assertGreaterEqual(first.queue_wait_seconds, 0.0)
        self.assertGreater(first.total_stt_seconds, 0.0)
        self.assertAlmostEqual(
            first.rtf,
            first.total_stt_seconds / first.submitted_audio_seconds,
        )


if __name__ == "__main__":
    unittest.main()
