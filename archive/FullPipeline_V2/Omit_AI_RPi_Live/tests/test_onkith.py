from __future__ import annotations

import queue
import threading
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.live_runtime import LiveSTTRuntime
from app.onkith_music import (
    OnkithMusicController,
    OnkithMusicWorker,
    append_song_to_search_query,
    close_player_process,
    create_player_page,
    intercept_onkith_command,
    search_youtube,
    verify_results,
)

from tests.helpers import recording, stt_result


class InterceptorTests(unittest.TestCase):
    def test_required_wake_aliases_and_intents(self) -> None:
        cases = {
            "Onkith imagine dragon bones": ("PLAY", "imagine dragon bones"),
            "ONKITH play Adele easy on me": (
                "PLAY",
                "play Adele easy on me",
            ),
            "Onkuth close it please": ("CLOSE", "close it please"),
            "Onkoth stop the song": ("CLOSE", "stop the song"),
            "On kit turn it off": ("CLOSE", "turn it off"),
            "Ankith play blinding lights": (
                "PLAY",
                "play blinding lights",
            ),
            "Unkith play believer": ("PLAY", "play believer"),
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                result = intercept_onkith_command(transcript)
                self.assertTrue(result.command_detected)
                self.assertEqual((result.intent, result.command_text), expected)
                self.assertEqual(result.remaining_text, "")

    def test_no_wake_forwards_the_exact_original_text(self) -> None:
        cases = [
            "play imagine dragons bones",
            "please stop the music",
            "we were talking about a song",
            "someone named Ankit called me",
            "  punctuation, and spacing stay exactly!  ",
        ]
        for transcript in cases:
            with self.subTest(transcript=transcript):
                result = intercept_onkith_command(transcript)
                self.assertFalse(result.command_detected)
                self.assertEqual(result.remaining_text, transcript)

    def test_case_punctuation_and_conservative_fuzzy_match(self) -> None:
        exact = intercept_onkith_command("Hello, ONKITH: play Bones!")
        fuzzy = intercept_onkith_command("Onkithh play Bones")
        false_name = intercept_onkith_command("Ankit, play Bones")
        self.assertTrue(exact.command_detected)
        self.assertEqual(exact.remaining_text, "Hello,")
        self.assertTrue(fuzzy.command_detected)
        self.assertFalse(false_name.command_detected)

    def test_only_five_words_are_consumed(self) -> None:
        result = intercept_onkith_command(
            "Onkith one two three four five six seven"
        )
        self.assertEqual(result.command_text, "one two three four five")
        self.assertEqual(result.remaining_text, "six seven")
        self.assertEqual(result.command_span_start, 0)
        self.assertEqual(result.command_span_end, 6)

    def test_surrounding_speech_is_preserved(self) -> None:
        result = intercept_onkith_command(
            "hello there Onkith play imagine dragons bones continue now"
        )
        self.assertEqual(result.intent, "PLAY")
        self.assertEqual(result.command_text, "play imagine dragons bones")
        self.assertEqual(result.remaining_text, "hello there continue now")

        prompt_example = intercept_onkith_command(
            "Hello everyone Onkith play Imagine Dragons Bones please continue"
        )
        self.assertEqual(
            prompt_example.command_text, "play Imagine Dragons Bones"
        )
        self.assertEqual(
            prompt_example.remaining_text, "Hello everyone please continue"
        )

    def test_close_variations(self) -> None:
        for transcript in [
            "Onkith stop",
            "Onkith close it please",
            "Onkuth turn music off",
            "Onkoth skip this",
            "On kit shut it down",
            "Onkith cloze it",
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(
                    intercept_onkith_command(transcript).intent,
                    "CLOSE",
                )


class FakeProcess:
    def __init__(self, pid: int = 321) -> None:
        self.pid = pid
        self.closed = False

    def poll(self):
        return 0 if self.closed else None


class FakeServer:
    def __init__(self) -> None:
        self.stopped = False
        self.closed = False

    def shutdown(self) -> None:
        self.stopped = True

    def server_close(self) -> None:
        self.closed = True


class FakeServerThread:
    def __init__(self) -> None:
        self.joined = False

    def join(self) -> None:
        self.joined = True


class ControllerTests(unittest.TestCase):
    def make_controller(self):
        searches = []
        closed = []
        opened = []
        server = FakeServer()
        server_thread = FakeServerThread()

        def search(query, api_key, region):
            searches.append((query, api_key, region))
            return [
                {
                    "video_id": "video-1",
                    "title": f"Title for {query}",
                    "channel": "Channel",
                }
            ]

        def verify(results, api_key, region):
            return results

        def close(process):
            closed.append(process)
            if process is not None:
                process.closed = True

        def open_player(url, profile):
            process = FakeProcess(pid=100 + len(opened))
            opened.append((url, profile, process))
            return process

        controller = OnkithMusicController(
            "test-key",
            "sa",
            search_function=search,
            verify_function=verify,
            page_factory=lambda videos, directory, port: directory / "player.html",
            server_factory=lambda directory: (
                server,
                server_thread,
                8765,
            ),
            player_opener=open_player,
            player_closer=close,
        )
        return controller, searches, closed, opened, server, server_thread

    def test_play_replaces_only_owned_player_and_close_is_idempotent(self):
        (
            controller,
            searches,
            closed,
            opened,
            server,
            server_thread,
        ) = self.make_controller()
        controller.play("first song")
        first = controller.browser_process
        controller.play("second song")
        second = controller.browser_process
        self.assertIsNot(first, second)
        self.assertTrue(first.closed)
        self.assertEqual([item[0] for item in searches], [
            "first song",
            "second song",
        ])
        self.assertEqual(len(opened), 2)

        controller.close()
        controller.close()
        self.assertIsNone(controller.browser_process)
        self.assertTrue(second.closed)
        self.assertNotIn("chrome", repr(closed).casefold())

        controller.play("third song")
        third = controller.browser_process
        controller.shutdown()
        self.assertTrue(third.closed)
        self.assertTrue(server.stopped)
        self.assertTrue(server.closed)
        self.assertTrue(server_thread.joined)

    def test_close_never_searches_youtube(self) -> None:
        controller, searches, _, _, _, _ = self.make_controller()
        controller.close()
        self.assertEqual(searches, [])
        controller.shutdown()

    def test_play_appends_song_immediately_before_youtube_search(self) -> None:
        controller, searches, _, _, _, _ = self.make_controller()
        controller.play("play believer")
        self.assertEqual(searches[0][0], "believer song")
        controller.shutdown()

    def test_play_does_not_duplicate_existing_song_suffix(self) -> None:
        controller, searches, _, _, _, _ = self.make_controller()
        controller.play("play believer song")
        self.assertEqual(searches[0][0], "believer song")
        controller.shutdown()

    def test_missing_key_fails_before_search_or_browser_open(self) -> None:
        controller, searches, _, opened, _, _ = self.make_controller()
        controller._api_key = ""
        with self.assertRaisesRegex(RuntimeError, "YOUTUBE_API_KEY"):
            controller.play("song")
        self.assertEqual(searches, [])
        self.assertEqual(opened, [])
        controller.shutdown()

    def test_environment_is_the_only_api_configuration_source(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOUTUBE_API_KEY": "runtime-test-key", "YOUTUBE_REGION": "gb"},
            clear=True,
        ):
            controller = OnkithMusicController.from_environment()
        self.assertEqual(controller._api_key, "runtime-test-key")
        self.assertEqual(controller.region, "GB")
        controller.shutdown()

    def test_windows_close_targets_only_the_owned_pid_tree(self) -> None:
        process = FakeProcess(pid=4321)
        with patch("app.onkith_music.subprocess.run") as taskkill:
            close_player_process(process)
        arguments = taskkill.call_args.args[0]
        self.assertEqual(
            arguments,
            ["taskkill", "/PID", "4321", "/T", "/F"],
        )


class YouTubeFunctionTests(unittest.TestCase):
    def test_song_suffix_helper(self) -> None:
        self.assertEqual(
            append_song_to_search_query("Imagine Dragons Bones"),
            "Imagine Dragons Bones song",
        )
        self.assertEqual(
            append_song_to_search_query("Imagine Dragons song"),
            "Imagine Dragons song",
        )
        self.assertEqual(
            append_song_to_search_query("Imagine Dragons SONG!"),
            "Imagine Dragons SONG",
        )
        self.assertEqual(
            append_song_to_search_query("Imagine song Dragons live"),
            "Imagine song Dragons live",
        )

    def test_search_only_normalization_strips_control_and_limits_words(self):
        for control in [
            "play",
            "plays",
            "open",
            "opens",
            "do",
            "start",
            "put",
            "listen",
            "find",
            "search",
            "launch",
        ]:
            with self.subTest(control=control):
                self.assertEqual(
                    append_song_to_search_query(
                        f"{control} one two three four five"
                    ),
                    "one two three four song",
                )
        self.assertEqual(
            append_song_to_search_query("one two three four five"),
            "one two three four five song",
        )
        self.assertEqual(
            append_song_to_search_query("one two three four song extra"),
            "one two three four song",
        )
        self.assertEqual(
            append_song_to_search_query("one two three four five song"),
            "one two three four five song",
        )

    def test_search_requests_music_embeddable_syndicated_results(self) -> None:
        response = {
            "items": [
                {
                    "id": {"videoId": "abc"},
                    "snippet": {
                        "title": "Artist &amp; Song",
                        "channelTitle": "Channel",
                    },
                }
            ]
        }
        with patch(
            "app.onkith_music.youtube_get",
            return_value=response,
        ) as get:
            results = search_youtube("natural language query", "key", "SA")
        parameters = get.call_args.args[1]
        self.assertEqual(parameters["videoCategoryId"], "10")
        self.assertEqual(parameters["videoEmbeddable"], "true")
        self.assertEqual(parameters["videoSyndicated"], "true")
        self.assertEqual(parameters["regionCode"], "SA")
        self.assertEqual(results[0]["title"], "Artist & Song")

    def test_verification_filters_status_embedding_and_region(self) -> None:
        candidates = [
            {"video_id": "good", "title": "Good", "channel": "A"},
            {"video_id": "private", "title": "Private", "channel": "B"},
            {"video_id": "blocked", "title": "Blocked", "channel": "C"},
        ]
        response = {
            "items": [
                {
                    "id": "good",
                    "status": {
                        "privacyStatus": "public",
                        "uploadStatus": "processed",
                        "embeddable": True,
                    },
                    "contentDetails": {},
                },
                {
                    "id": "private",
                    "status": {
                        "privacyStatus": "private",
                        "uploadStatus": "processed",
                        "embeddable": True,
                    },
                    "contentDetails": {},
                },
                {
                    "id": "blocked",
                    "status": {
                        "privacyStatus": "public",
                        "uploadStatus": "processed",
                        "embeddable": True,
                    },
                    "contentDetails": {
                        "regionRestriction": {"blocked": ["SA"]}
                    },
                },
            ]
        }
        with patch("app.onkith_music.youtube_get", return_value=response):
            self.assertEqual(
                verify_results(candidates, "key", "SA"),
                [candidates[0]],
            )

    def test_player_page_has_autoplay_watchdog_and_no_api_key(self) -> None:
        videos = [
            {"video_id": "abc", "title": "</script>", "channel": "Channel"}
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = create_player_page(videos, Path(directory), 8765)
            page = path.read_text(encoding="utf-8")
        self.assertIn("autoplay=1", page)
        self.assertIn("startWatchdog", page)
        self.assertIn("tryNextVideo", page)
        self.assertNotIn("</script>\"", page)
        self.assertNotIn("YOUTUBE_API_KEY", page)


class WorkerTests(unittest.TestCase):
    def test_worker_survives_an_action_exception(self) -> None:
        class Controller:
            def __init__(self):
                self.played = []
                self.closed = 0
                self.shutdown_called = False

            def play(self, query, *, should_cancel):
                self.played.append(query)
                if query == "broken":
                    raise RuntimeError("expected failure")

            def close(self):
                self.closed += 1

            def shutdown(self):
                self.shutdown_called = True

        controller = Controller()
        worker = OnkithMusicWorker(controller)
        worker.start()
        self.assertFalse(worker.thread.daemon)
        worker.submit_play("broken")
        worker.queue.join()
        worker.submit_play("working")
        worker.queue.join()
        self.assertTrue(worker.thread.is_alive())
        self.assertEqual(controller.played, ["broken", "working"])
        self.assertEqual(len(worker.errors), 1)
        worker.shutdown()
        self.assertTrue(controller.shutdown_called)
        self.assertFalse(worker.thread.is_alive())

    def test_newer_play_supersedes_slow_play(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()

        class Controller:
            def __init__(self):
                self.opened = []

            def play(self, query, *, should_cancel):
                if query == "first":
                    first_started.set()
                    release_first.wait(timeout=2)
                if not should_cancel():
                    self.opened.append(query)

            def close(self):
                pass

            def shutdown(self):
                pass

        controller = Controller()
        worker = OnkithMusicWorker(controller)
        worker.start()
        worker.submit_play("first")
        self.assertTrue(first_started.wait(timeout=1))
        worker.submit_play("second")
        release_first.set()
        worker.queue.join()
        worker.shutdown()
        self.assertEqual(controller.opened, ["second"])

    def test_newer_close_invalidates_slow_play(self) -> None:
        play_started = threading.Event()
        release_play = threading.Event()

        class Controller:
            def __init__(self):
                self.opened = []
                self.close_count = 0

            def play(self, query, *, should_cancel):
                play_started.set()
                release_play.wait(timeout=2)
                if not should_cancel():
                    self.opened.append(query)

            def close(self):
                self.close_count += 1

            def shutdown(self):
                pass

        controller = Controller()
        worker = OnkithMusicWorker(controller)
        worker.start()
        worker.submit_play("slow")
        self.assertTrue(play_started.wait(timeout=1))
        worker.submit_close()
        release_play.set()
        worker.queue.join()
        worker.shutdown()
        self.assertEqual(controller.opened, [])
        self.assertEqual(controller.close_count, 1)


class STTIntegrationTests(unittest.TestCase):
    class Privacy:
        def __init__(self):
            self.results = []
            self.forwarded = threading.Event()

        def submit_stt(self, result):
            self.results.append(result)
            self.forwarded.set()

    class Moonshine:
        def __init__(self, text):
            self.text = text

        def transcribe_recording(self, item, partial_callback=None):
            return stt_result(item, self.text)

    def test_command_is_removed_immediately_while_music_is_slow(self) -> None:
        music_started = threading.Event()
        release_music = threading.Event()

        class SlowController:
            def play(self, query, *, should_cancel):
                music_started.set()
                release_music.wait(timeout=2)

            def close(self):
                pass

            def shutdown(self):
                pass

        music = OnkithMusicWorker(SlowController())
        privacy = self.Privacy()
        completed: queue.Queue = queue.Queue()
        stt = LiveSTTRuntime(
            completed,
            self.Moonshine(
                "normal speech Onkith play a slow song"
            ),
            privacy,
            music_worker=music,
            on_event=lambda *args: None,
        )
        music.start()
        stt.start()
        started = time.perf_counter()
        completed.put(recording(1))
        self.assertTrue(privacy.forwarded.wait(timeout=0.5))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.5)
        self.assertEqual(
            [result.final_text for result in privacy.results],
            ["normal speech"],
        )
        self.assertTrue(music_started.wait(timeout=0.5))
        stt.shutdown()
        release_music.set()
        music.shutdown()

    def test_command_only_transcript_never_reaches_stage5(self) -> None:
        class Music:
            def __init__(self):
                self.plays = []

            def submit_play(self, query):
                self.plays.append(query)

            def submit_close(self):
                raise AssertionError("unexpected close")

        privacy = self.Privacy()
        music = Music()
        completed: queue.Queue = queue.Queue()
        stt = LiveSTTRuntime(
            completed,
            self.Moonshine("Onkith play believer"),
            privacy,
            music_worker=music,
            on_event=lambda *args: None,
        )
        stt.start()
        completed.put(recording(1))
        stt.shutdown()
        self.assertEqual(music.plays, ["play believer"])
        self.assertEqual(privacy.results, [])

    def test_no_wake_reaches_stage5_unchanged(self) -> None:
        privacy = self.Privacy()
        completed: queue.Queue = queue.Queue()
        text = "ordinary speech with exact punctuation!"
        stt = LiveSTTRuntime(
            completed,
            self.Moonshine(text),
            privacy,
            on_event=lambda *args: None,
        )
        stt.start()
        completed.put(recording(1))
        stt.shutdown()
        self.assertEqual(privacy.results[0].final_text, text)


if __name__ == "__main__":
    unittest.main()
