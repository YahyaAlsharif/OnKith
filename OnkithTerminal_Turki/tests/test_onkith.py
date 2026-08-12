from __future__ import annotations

import json
import math
import queue
import re
import shutil
import subprocess
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
    WakeWords,
    append_song_to_search_query,
    configure_wake_words,
    create_player_page,
    intercept_onkith_command,
    normalize_videos,
    search_youtube,
    verify_results,
    wake_words,
    without_wake_word,
)
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.whisper_adapter import name_hint

from tests.helpers import recording, stt_result


class InterceptorTests(unittest.TestCase):
    def test_required_wake_aliases_and_intents(self) -> None:
        cases = {
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
            "Sada play believer": ("PLAY", "play believer"),
            "صدى play believer": ("PLAY", "play believer"),
            "Sitr open believer": ("PLAY", "open believer"),
            "Sitter stop the music": ("CLOSE", "stop the music"),
            "Setter play believer": ("PLAY", "play believer"),
            "ستر close it": ("CLOSE", "close it"),
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

    def test_only_five_words_are_consumed_by_a_music_command(self) -> None:
        result = intercept_onkith_command(
            "Onkith play one two three four five six seven"
        )
        self.assertEqual(result.intent, "PLAY")
        self.assertEqual(result.command_text, "play one two three four")
        self.assertEqual(result.remaining_text, "five six seven")

    def test_the_verb_picks_the_lane_and_everything_else_is_a_question(
        self,
    ) -> None:
        """This used to run the other way round.

        The wake word sent speech to the music worker and everything else
        went to the model, so "Onkith who was Alan Turing" searched YouTube
        for "who was Alan Turing song" while an overheard sentence about a
        meeting was uploaded.
        """

        asks = [
            "Onkith what is the weather tomorrow",
            "Onkith who was Alan Turing",
            "Onkith remind me to call the dentist",
            # A stop word later in a question must not reach the transport.
            "Onkith what time do the shops close",
            # No verb at all, so no lane to put it in but the assistant's.
            "Onkith imagine dragon bones",
            # The wake word alone is somebody getting its attention.
            "Onkith",
        ]
        for transcript in asks:
            with self.subTest(transcript=transcript):
                result = intercept_onkith_command(transcript)
                self.assertTrue(result.command_detected)
                self.assertEqual(result.intent, "ASK")

        for transcript in ("Onkith play believer", "Onkith stop the music"):
            with self.subTest(transcript=transcript):
                self.assertIn(
                    intercept_onkith_command(transcript).intent,
                    ("PLAY", "CLOSE"),
                )

    def test_the_player_can_be_asked_for_and_sent_away(self) -> None:
        """Showing the player is a screen instruction, not a music one.

        The verbs overlap and only the noun separates them: "close it" stops
        the track, "close the player" puts the panel away and leaves the
        track playing.
        """

        for transcript in (
            "Onkith show the player",
            "Onkith open the player",
            "Onkith bring up the player",
            "Onkith pop the player",
        ):
            with self.subTest(transcript=transcript):
                self.assertEqual(intercept_onkith_command(transcript).intent, "SHOW")

        for transcript in (
            "Onkith close the player",
            "Onkith hide the player",
            "Onkith dismiss the panel",
            "Onkith close the screen",
        ):
            with self.subTest(transcript=transcript):
                self.assertEqual(intercept_onkith_command(transcript).intent, "HIDE")

    def test_without_the_noun_it_is_still_about_the_music(self) -> None:
        # Otherwise "open believer" puts the panel up instead of playing
        # anything, and "close it" leaves the music running.
        self.assertEqual(intercept_onkith_command("Onkith open believer").intent, "PLAY")
        self.assertEqual(intercept_onkith_command("Onkith close it").intent, "CLOSE")
        self.assertEqual(
            intercept_onkith_command("Onkith stop the music").intent, "CLOSE"
        )
        # And a question that happens to mention the player is a question.
        self.assertEqual(
            intercept_onkith_command("Onkith what is on the player").intent, "ASK"
        )

    def test_a_question_is_not_five_words_long(self) -> None:
        """A song title has to end somewhere. A question does not.

        Cutting a question at five words was how "Onkith remind me to call
        the dentist" became a search for "remind me to call the song" with
        the word dentist left stranded in the transcript.
        """

        result = intercept_onkith_command(
            "Onkith remind me to call the dentist tomorrow morning"
        )
        self.assertEqual(result.intent, "ASK")
        self.assertEqual(
            result.command_text,
            "remind me to call the dentist tomorrow morning",
        )
        # Nothing is lifted out: the sentence is the reason they spoke, and
        # the transcript keeps it whole.
        self.assertEqual(
            result.remaining_text,
            "Onkith remind me to call the dentist tomorrow morning",
        )

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


class VolumeLaneTests(unittest.TestCase):
    """Louder and quieter by voice.

    set_volume and its route through the core were already there. What
    was missing was a lane, so no spoken phrase could reach any of it.
    """

    def lane(self, transcript: str):
        return intercept_onkith_command(transcript).intent

    def test_the_plain_ways_of_asking(self) -> None:
        for transcript, expected in [
            ("Onkith louder", "LOUDER"),
            ("Onkith quieter", "QUIETER"),
            ("Onkith softer", "QUIETER"),
            ("Onkith volume up", "LOUDER"),
            ("Onkith volume down", "QUIETER"),
            ("Onkith turn it up", "LOUDER"),
            ("Onkith turn it down", "QUIETER"),
            ("Onkith turn the volume up", "LOUDER"),
            ("Onkith turn the music down", "QUIETER"),
            ("Onkith increase the volume", "LOUDER"),
            ("Onkith lower the volume", "QUIETER"),
            ("Onkith raise the volume please", "LOUDER"),
            ("Onkith make it louder", "LOUDER"),
            ("Onkith make it quieter", "QUIETER"),
        ]:
            with self.subTest(transcript=transcript):
                self.assertEqual(self.lane(transcript), expected)

    def test_turning_it_off_still_stops_the_music(self) -> None:
        # "turn it down" and "turn it off" share a verb, and only the
        # last word says which was meant. Getting this backwards means
        # asking for it quieter kills the track.
        for transcript in ("Onkith turn it off", "Onkith turn the music off"):
            with self.subTest(transcript=transcript):
                self.assertEqual(self.lane(transcript), "CLOSE")

    def test_the_screen_still_wins_its_own_verbs(self) -> None:
        # "bring" and "put" are level verbs here and screen verbs there.
        # The noun decides, and the screen is read first.
        self.assertEqual(self.lane("Onkith bring up the player"), "SHOW")
        self.assertEqual(self.lane("Onkith bring the volume up"), "LOUDER")

    def test_a_question_about_volume_is_still_a_question(self) -> None:
        self.assertEqual(self.lane("Onkith why is the volume so low"), "ASK")
        self.assertEqual(self.lane("Onkith what is the volume"), "ASK")

    def test_being_told_to_be_quiet_is_not_the_level(self) -> None:
        # "Onkith quiet" is somebody telling it to stop talking, which is
        # the transport. Only the comparative means the level.
        self.assertNotIn(self.lane("Onkith quiet"), ("LOUDER", "QUIETER"))

    def test_the_hint_shows_the_recogniser_the_word_volume(self) -> None:
        # A word the hint does not contain is one the recogniser is not
        # biased toward, which is how "play bones" became "Playbons"
        # before the shapes were added.
        #
        # The word, not a particular wording. The hint carries one
        # example per lane rather than one per phrasing, because eight
        # examples cost a digit out of an id number, so asserting an
        # exact sentence here would fight the thing that keeps digits.
        hint = name_hint(["onkith"])
        self.assertIn("volume", hint)
        self.assertIn("Onkith, turn the volume up.", hint)


class WakeWordStrippingTests(unittest.TestCase):
    """The assistant should not think the person is called Onkith.

    Seen on screen: "My purpose is to assist and provide information to
    users like you, Onkith." The name is how the device was addressed,
    and the model has no way to tell that from the text it is handed.
    """

    def test_the_name_comes_off_the_front(self) -> None:
        self.assertEqual(
            without_wake_word("Onkith, what is your purpose?"),
            "what is your purpose?",
        )

    def test_the_question_keeps_its_punctuation(self) -> None:
        # command_text drops everything that is not a word, and a
        # question handed over without its question mark reads as a
        # statement to whatever answers it.
        self.assertTrue(without_wake_word("Onkith, who was Alan Turing?").endswith("?"))

    def test_a_name_later_in_the_sentence_survives(self) -> None:
        # It is part of what was asked, not how the device was called.
        self.assertIn("Onkith", without_wake_word("Onkith, what does Onkith mean?"))

    def test_speech_before_the_name_is_kept(self) -> None:
        self.assertEqual(
            without_wake_word("so I said Onkith what is the time"),
            "so I said what is the time",
        )

    def test_a_sentence_that_never_called_it_is_untouched(self) -> None:
        sentence = "the weather is fine today"
        self.assertEqual(without_wake_word(sentence), sentence)

    def test_the_bare_name_is_left_alone(self) -> None:
        # Nothing else was said, so stripping it would send an empty
        # message. Better the model sees the name than nothing at all.
        self.assertEqual(without_wake_word("Onkith"), "Onkith")

    def test_every_configured_name_is_stripped(self) -> None:
        # Against the shipped config rather than the code default, since
        # echo is a name this deployment added and the point of the test
        # is that renaming the device does not leave one name behind.
        previous = wake_words()
        configure_wake_words(WakeWords.from_config(PipelineConfig().wake))
        try:
            for name in ("Onkith", "Sada", "Sitr", "Echo"):
                with self.subTest(name=name):
                    self.assertEqual(
                        without_wake_word(f"{name}, what time is it?"),
                        "what time is it?",
                    )
        finally:
            configure_wake_words(previous)

    def test_a_placeholder_in_the_masked_text_is_untouched(self) -> None:
        # This runs on masked text, never raw, so placeholders are what
        # it actually sees most of the time.
        self.assertEqual(
            without_wake_word("Onkith, call [TELEPHONENUM_1] for me"),
            "call [TELEPHONENUM_1] for me",
        )


class FakePlayback:
    """Stand in for the core playback engine the controller now drives."""

    def __init__(self) -> None:
        self.loaded: list[list[dict]] = []
        self.stop_count = 0
        self.shutdown_count = 0

    def load(self, tracks, *, start_index: int = 0) -> None:
        self.loaded.append(list(tracks))

    def stop(self) -> None:
        self.stop_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


class ControllerTests(unittest.TestCase):
    def make_controller(self):
        searches = []
        playback = FakePlayback()

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

        controller = OnkithMusicController(
            "test-key",
            "sa",
            playback=playback,
            search_function=search,
            verify_function=verify,
        )
        return controller, searches, playback

    def test_play_replaces_the_queue_and_opens_no_browser(self) -> None:
        controller, searches, playback = self.make_controller()
        controller.play("first song")
        controller.play("second song")
        self.assertEqual(
            [item[0] for item in searches], ["first song", "second song"]
        )
        # Each command replaces the queue in the core playback process.
        self.assertEqual(len(playback.loaded), 2)
        self.assertEqual(playback.loaded[0][0]["video_id"], "video-1")

    def test_close_stops_playback_without_searching(self) -> None:
        controller, searches, playback = self.make_controller()
        controller.close()
        controller.close()
        self.assertEqual(searches, [])
        self.assertEqual(playback.stop_count, 2)

    def test_playback_outlives_the_controller(self) -> None:
        # The core runtime owns the audio process, so a controller
        # shutting down must not stop music.
        controller, _, playback = self.make_controller()
        controller.shutdown()
        self.assertEqual(playback.shutdown_count, 0)
        controller.close()
        self.assertEqual(playback.stop_count, 0)

    def test_play_appends_song_immediately_before_youtube_search(self) -> None:
        controller, searches, _ = self.make_controller()
        controller.play("play believer")
        self.assertEqual(searches[0][0], "believer song")

    def test_play_does_not_duplicate_existing_song_suffix(self) -> None:
        controller, searches, _ = self.make_controller()
        controller.play("play believer song")
        self.assertEqual(searches[0][0], "believer song")

    def test_missing_key_fails_before_search_or_playback(self) -> None:
        controller, searches, playback = self.make_controller()
        controller._api_key = ""
        with self.assertRaisesRegex(RuntimeError, "YOUTUBE_API_KEY"):
            controller.play("song")
        self.assertEqual(searches, [])
        self.assertEqual(playback.loaded, [])

    def test_no_playable_result_is_reported(self) -> None:
        playback = FakePlayback()
        controller = OnkithMusicController(
            "test-key",
            "sa",
            playback=playback,
            search_function=lambda query, key, region: [],
            verify_function=lambda results, key, region: [],
        )
        with self.assertRaisesRegex(RuntimeError, "No playable"):
            controller.play("nothing here")
        self.assertEqual(playback.loaded, [])

    def test_a_superseded_play_never_reaches_playback(self) -> None:
        controller, _, playback = self.make_controller()
        controller.play("song", should_cancel=lambda: True)
        self.assertEqual(playback.loaded, [])

    def test_environment_is_the_only_api_configuration_source(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOUTUBE_API_KEY": "runtime-test-key", "YOUTUBE_REGION": "gb"},
            clear=True,
        ):
            controller = OnkithMusicController.from_environment(
                playback=FakePlayback()
            )
        self.assertEqual(controller._api_key, "runtime-test-key")
        self.assertEqual(controller.region, "GB")


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
                        "thumbnails": {
                            "high": {
                                "url": "https://images.example/high.jpg"
                            },
                            "maxres": {
                                "url": "https://images.example/maxres.jpg"
                            },
                        },
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
        self.assertEqual(parameters["maxResults"], "25")
        self.assertEqual(results[0]["title"], "Artist & Song")
        self.assertEqual(
            results[0]["thumbnail"],
            "https://images.example/maxres.jpg",
        )

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
                    "snippet": {
                        "title": "Verified &amp; exact",
                        "channelTitle": "Verified channel",
                        "thumbnails": {
                            "standard": {
                                "url": "https://images.example/exact.jpg"
                            }
                        },
                    },
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
            verified = verify_results(candidates, "key", "SA")
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["video_id"], "good")
        self.assertEqual(verified[0]["title"], "Verified & exact")
        self.assertEqual(verified[0]["channel"], "Verified channel")
        self.assertEqual(
            verified[0]["thumbnail"],
            "https://images.example/exact.jpg",
        )

    def test_verification_preserves_order_and_caps_player_at_five(self) -> None:
        candidates = [
            {
                "video_id": f"video-{index}",
                "title": f"Search title {index}",
                "channel": f"Search channel {index}",
            }
            for index in range(7)
        ]
        response = {
            "items": [
                {
                    "id": candidate["video_id"],
                    "snippet": {
                        "title": f"Verified title {index}",
                        "channelTitle": f"Verified channel {index}",
                    },
                    "status": {
                        "privacyStatus": "public",
                        "uploadStatus": "processed",
                        "embeddable": True,
                    },
                    "contentDetails": {},
                }
                for index, candidate in reversed(list(enumerate(candidates)))
            ]
        }
        with patch("app.onkith_music.youtube_get", return_value=response):
            verified = verify_results(candidates, "key", "SA")
        self.assertEqual(
            [item["video_id"] for item in verified],
            [f"video-{index}" for index in range(5)],
        )
        self.assertEqual(
            [item["title"] for item in verified],
            [f"Verified title {index}" for index in range(5)],
        )

    def test_player_page_is_static_with_3d_controls_and_local_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            path = create_player_page(output_directory)
            page = path.read_text(encoding="utf-8")
            app_module = (
                output_directory / "assets" / "player-app-v1.mjs"
            ).read_text(encoding="utf-8")
            model_module = (
                output_directory / "assets" / "player-spline-v1.mjs"
            ).read_text(encoding="utf-8")
            copied_assets = [
                output_directory / "assets" / asset
                for asset in (
                    "onkith-favicon.svg",
                    # The recorder and everything that draws it. The viewer
                    # imports the other two chunks by relative path, so all
                    # four have to land in one directory.
                    "spline/spline-viewer.js",
                    "spline/opentype.js",
                    "spline/process.js",
                    "spline/scene.splinecode",
                )
            ]
            copied_assets_exist = all(
                asset.is_file() for asset in copied_assets
            )
            rejected_rasters_absent = all(
                not (output_directory / "assets" / filename).exists()
                for filename in (
                    "death-stranding-left-v1.png",
                    "death-stranding-right-v1.png",
                    "industrial-player-deck-v1.png",
                )
            )

        # Track data now arrives over the websocket, so the page carries no
        # embedded results and no YouTube embed to play them.
        self.assertNotIn('id="video-data"', page)
        self.assertNotIn('<div id="player"></div>', page)
        self.assertNotIn("youtube.com/iframe_api", page + app_module)
        self.assertNotIn("YT.Player", app_module)

        self.assertTrue(copied_assets_exist)
        self.assertIn('id="track-flow"', page)
        self.assertIn('id="machine-viewer"', page)
        self.assertIn("<spline-viewer", page)
        self.assertIn('data-machine-action="previous"', page)
        self.assertIn('data-machine-action="toggle"', page)
        self.assertIn('data-machine-action="stop"', page)
        self.assertIn('data-machine-action="mute"', page)
        self.assertIn('data-machine-action="next"', page)
        self.assertIn('data-machine-action="seek"', page)
        self.assertIn('data-machine-action="volume"', page)
        self.assertIn('role="slider"', page)
        self.assertIn('id="fallback-controls" hidden', page)
        self.assertIn('id="fallback-seek"', page)
        self.assertIn('id="fallback-volume"', page)
        self.assertIn('src="assets/player-app-v1.mjs"', page)
        self.assertIn('src="assets/spline/spline-viewer.js"', page)
        self.assertIn('url="assets/spline/scene.splinecode"', page)
        # Nothing about the recorder may be fetched from a network the
        # device might not have.
        self.assertNotIn("unpkg.com", page)
        self.assertNotIn("prod.spline.design", page)
        self.assertIn('<h1 id="title">', page)
        self.assertIn('<p class="track-channel" id="channel">', page)

        # The surface subscribes to the core rather than owning playback.
        self.assertIn('new WebSocket(`${protocol}//${window.location.host}/events`)', app_module)
        self.assertIn("selectVideo", app_module)
        self.assertIn("scheduleReconnect", app_module)
        self.assertIn("\\\\ Player \\\\ Onkith", app_module)
        self.assertIn("createRecorderMachine", app_module)
        for command in (
            'send("toggle")',
            'send("stop")',
            'send("mute")',
            'send("select", { index: numericIndex })',
            'send("seek", { ratio })',
            'send("volume", { value: playback.volume })',
        ):
            with self.subTest(command=command):
                self.assertIn(command, app_module)

        # The object is a finished Spline scene. Nothing here may rebuild
        # it, shade it, or reach for a modelling library to do so.
        self.assertIn("spline-viewer", model_module)
        self.assertIn("assets/spline/scene.splinecode", model_module)
        for rebuilt in ("THREE.", "GLTFLoader", "MeshStandardMaterial",
                        "getContext(\"2d\")", "createElement(\"canvas\")"):
            with self.subTest(rebuilt=rebuilt):
                self.assertNotIn(rebuilt, model_module)
        self.assertNotIn("three", page.lower().split("<style")[0])

        rejected_rasters = (
            "death-stranding-left-v1.png",
            "death-stranding-right-v1.png",
            "industrial-player-deck-v1.png",
        )
        rendered_player = page + app_module + model_module
        for filename in rejected_rasters:
            self.assertNotIn(filename, rendered_player)
        self.assertTrue(rejected_rasters_absent)
        self.assertNotIn("__VIDEOS_JSON__", page)
        self.assertNotIn("__INITIAL_TITLE__", page)
        self.assertNotIn("__INITIAL_CHANNEL__", page)
        self.assertNotIn("__ORIGIN__", page)
        self.assertNotIn("YOUTUBE_API_KEY", page)

    def test_kiosk_surface_hides_the_pointer(self) -> None:
        page = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "player_template.html"
        ).read_text(encoding="utf-8")
        app_module = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "player"
            / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")
        self.assertIn("html.pointer-idle", page)
        self.assertIn("cursor: none !important;", page)
        self.assertIn('get("kiosk") === "1"', app_module)
        # Hidden only once the pointer stops being used, and back on the
        # next movement; hiding it outright makes the window unreviewable.
        self.assertIn("function installIdlePointer", app_module)
        self.assertIn('root.classList.remove("pointer-idle")', app_module)
        self.assertIn("POINTER_IDLE_MS", app_module)

    def test_thumbnail_falls_back_to_the_exact_video(self) -> None:
        normalized = normalize_videos(
            [
                {
                    "video_id": "exact-video-id",
                    "title": "Exact result",
                    "channel": "Channel",
                }
            ]
        )
        self.assertEqual(
            normalized[0]["thumbnail"],
            "https://i.ytimg.com/vi/exact-video-id/hqdefault.jpg",
        )

    def test_supplied_thumbnail_is_preserved(self) -> None:
        normalized = normalize_videos(
            [
                {
                    "video_id": "exact-video-id",
                    "title": "Exact result",
                    "channel": "Channel",
                    "thumbnail": "https://images.example/real.jpg",
                }
            ]
        )
        self.assertEqual(
            normalized[0]["thumbnail"], "https://images.example/real.jpg"
        )

    def test_player_javascript_passes_node_syntax_check(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        player_assets = (
            Path(__file__).resolve().parents[1] / "assets" / "player"
        )
        scripts = sorted(player_assets.rglob("*.mjs"))
        scripts.extend(sorted(player_assets.rglob("*.js")))
        self.assertGreater(len(scripts), 2)
        for script in scripts:
            with self.subTest(script=script.relative_to(player_assets)):
                result = subprocess.run(
                    [node, "--check", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=result.stderr or result.stdout,
                )

    def test_the_recorder_asks_no_network_for_anything(self) -> None:
        # The device has to draw its own object on a Pi with nothing
        # plugged in. The viewer will happily fetch its runtime chunks and
        # its scene from unpkg and from Spline's API if either is missing,
        # and that failure only shows up on the day there is no network.
        player_root = Path(__file__).resolve().parents[1]
        spline = player_root / "assets" / "player" / "spline"
        required = {
            "spline-viewer.js": 1_000_000,
            "opentype.js": 50_000,
            "process.js": 50_000,
            "scene.splinecode": 1_000_000,
        }
        for filename, minimum in required.items():
            with self.subTest(filename=filename):
                asset = spline / filename
                self.assertTrue(asset.is_file(), msg=f"{asset} is missing")
                self.assertGreater(asset.stat().st_size, minimum)

        # opentype and process are the two chunks this scene pulls in, by
        # relative path, so they have to sit beside the viewer.
        viewer = (spline / "spline-viewer.js").read_text(
            encoding="utf-8", errors="ignore"
        )
        for chunk in ('import("./opentype.js")', 'import("./process.js")'):
            with self.subTest(chunk=chunk):
                self.assertIn(chunk, viewer)

        page = (player_root / "app" / "player_template.html").read_text(
            encoding="utf-8"
        )
        module = (
            player_root / "assets" / "player" / "player-spline-v1.mjs"
        ).read_text(encoding="utf-8")
        app_module = (
            player_root / "assets" / "player" / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")
        for host in (
            "unpkg.com",
            "prod.spline.design",
            "apis.spline.design",
            "gstatic.com",
        ):
            with self.subTest(host=host):
                self.assertNotIn(host, page + module + app_module)

        # And the page still works when the object cannot be drawn at all.
        self.assertIn('id="fallback-controls" hidden', page)
        self.assertIn("revealFallback", app_module)

    def test_control_labels_sit_on_their_control(self) -> None:
        player_root = Path(__file__).resolve().parents[1]
        page = (player_root / "app" / "player_template.html").read_text(
            encoding="utf-8"
        )
        model_module = (
            player_root / "assets" / "player" / "player-spline-v1.mjs"
        ).read_text(encoding="utf-8")
        app_module = (
            player_root / "assets" / "player" / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")

        label_rule = page.split("    .machine-hotspot::after {", maxsplit=1)[
            1
        ].split("    }", maxsplit=1)[0]
        # Centred on the control rather than pushed below or beside it,
        # which is what put labels off the device body.
        self.assertIn("transform: translate(-50%, -50%);", label_rule)
        self.assertNotIn("calc(100% + var(--s2))", label_rule)
        self.assertNotIn('data-machine-action="volume"]::after', page)

        # The viewer only reports clicks for the parts its author gave an
        # interaction to, which is fewer than the page needs, so each
        # control is projected out of the scene and the page's own button
        # laid over it.
        self.assertIn("function projectedBounds", model_module)
        self.assertIn("function placeHotspots", model_module)
        self.assertIn("MINIMUM_TARGET_PX", model_module)
        self.assertIn("dataset.modelNode", model_module)

        # The object is turned square on, which is what makes the targets a
        # measurement rather than a guess: near and far faces then project
        # to the same rectangle, so a control's whole box is its footprint.
        # At three quarters this had to take only the corners nearest the
        # camera, and on a key those belong to the printed symbol rather
        # than the cap it is printed on.
        self.assertIn("function faceFront", model_module)
        self.assertNotIn("FRONT_FACE_SHARE", model_module)
        self.assertNotIn("right.depth - left.depth", model_module)

        # The scene's text meshes carry an empty bounding box. One of those
        # under a control turns its measurement into NaN and the control is
        # silently never placed.
        self.assertIn("isFiniteBox", model_module)
        self.assertIn("Number.isFinite(x) === false", model_module)

        # The platter covers the two small keys beside it, so a control has
        # to stack above anything larger than itself to be clickable.
        self.assertIn("placements.sort((left, right) => right.area - left.area);", model_module)

        # Measured after the scene has finished moving its own camera, and
        # against a camera nobody can turn, pan or zoom. Zoom especially:
        # left on, a wheel over the stage shrinks the object away instead
        # of scrolling the page.
        self.assertIn("function settleCamera", model_module)
        self.assertIn("function lockCamera", model_module)
        # Still is compared with a tolerance, never exactly. A camera that
        # has stopped still walks the last bits of a double, so an exact
        # comparison never sees it stop and the settle rides out its whole
        # timeout: that alone held the object back for eight seconds.
        settle = model_module.split("async function settleCamera(", maxsplit=1)[
            1
        ].split("\n}", maxsplit=1)[0]
        self.assertIn("hasMoved(previous, current)", settle)
        self.assertNotIn("current === previous", settle)
        tolerance = re.search(
            r"const CAMERA_STILL_TOLERANCE = ([0-9.e-]+);", model_module
        )
        self.assertIsNotNone(tolerance)
        # Wide of the dither by orders of magnitude, and far under anything
        # a camera hundreds of units out could move visibly.
        self.assertGreater(float(tolerance.group(1)), 1e-9)
        self.assertLess(float(tolerance.group(1)), 0.1)
        for locked in ("orbit.enableRotate = false;", "orbit.enablePan = false;",
                       "orbit.enableZoom = false;"):
            with self.subTest(locked=locked):
                self.assertIn(locked, model_module)
        # The stage is not a scroll trap; only the dragged controls take
        # the gesture.
        viewer_rule = page.split("    #machine-viewer {", maxsplit=1)[1].split(
            "    }", maxsplit=1
        )[0]
        self.assertIn("touch-action: pan-y;", viewer_rule)
        slider_rule = page.split(
            '    .machine-hotspot[role="slider"] {', maxsplit=1
        )[1].split("    }", maxsplit=1)[0]
        self.assertIn("touch-action: none;", slider_rule)

        # The viewer's own badge is not part of this device.
        self.assertIn("function removeViewerChrome", model_module)
        self.assertIn("viewer._logo", model_module)

        # The scene carries a click sample of its author's choosing, and
        # firing its authored press plays it. This device makes two sounds,
        # the music and Onkith, so the press is animated here instead and
        # nothing emits a scene event.
        self.assertNotIn("emitEvent", model_module)
        self.assertIn("function pressKey", model_module)
        # And a click that misses a control must not reach the scene
        # either, which is the other way that sample gets played.
        viewer_rule = page.split("    #machine-viewer {", maxsplit=1)[1].split(
            "    }", maxsplit=1
        )[0]
        self.assertIn("pointer-events: none;", viewer_rule)

        # Dragging a control moves the part it belongs to while it is
        # playing, not only when it is stopped: the platter carries where
        # the track is as well as the turning it has done, and the knob
        # follows the hand rather than the round trip through the core.
        # Subtracted, not added: a record turns clockwise, and a platter
        # going the other way reads as the track running backwards.
        self.assertIn(
            "platter.rotation.z = platterRest - spinAngle "
            "- progress * PLATTER_SEEK_RADIANS;",
            model_module,
        )
        volume_sync = app_module.split(
            "function syncVolumeInterface() {", maxsplit=1
        )[1].split("\n}", maxsplit=1)[0]
        self.assertIn('callMachine("setVolume", playback.volume);', volume_sync)

        # Every message from the core carries its idea of the level, and one
        # already in flight carries the level from before the drag. Taking
        # it mid-drag snaps the knob back to where the hand started, which
        # is a control that appears not to move at all.
        apply_playback = app_module.split(
            "function applyPlayback(section) {", maxsplit=1
        )[1].split("\n}", maxsplit=1)[0]
        self.assertIn('hotspotDrag.kind === "volume"', apply_playback)
        self.assertIn("delete incoming.volume;", apply_playback)
        # The timeline is protected from the same race the other way.
        timeline = app_module.split(
            "function updateTimeline() {", maxsplit=1
        )[1].split("\n}", maxsplit=1)[0]
        self.assertIn("if (previewingSeek) return;", timeline)

        # The scene opens by flying its own camera in at the angle it was
        # composed at, so the object is held back until it has been turned
        # square on, sized and measured. Otherwise it arrives tilted and
        # cropped and corrects itself in front of the room.
        self.assertIn("viewer.classList.add(\"is-ready\");", model_module)
        self.assertIn("opacity: 0;", viewer_rule)
        ready_rule = page.split("    #machine-viewer.is-ready {", maxsplit=1)[
            1
        ].split("    }", maxsplit=1)[0]
        self.assertIn("opacity: 1;", ready_rule)
        # A recorder that failed still shows whatever it managed to draw.
        self.assertIn('elements.machineViewer.classList.add("is-ready");', app_module)

    def test_turning_a_part_reaches_the_object_and_not_only_a_number(self) -> None:
        """Every object arrives with its matrix update switched off.

        The runtime composes matrices itself, from its own copy of each
        transform, so a rotation written straight onto a scene node is a
        number nothing reads: the platter climbed past fifty radians while
        standing perfectly still on screen, and reading the property back
        agreed with itself the whole time. Only the viewer's own handle for
        a part actually turns it.
        """
        module = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "player"
            / "player-spline-v1.mjs"
        ).read_text(encoding="utf-8")

        self.assertIn("function driveable(", module)
        handle = module.split("function driveable(", maxsplit=1)[1].split(
            "\n}", maxsplit=1
        )[0]
        # Asked for by the node's id, so a name that appears twice in the
        # scene cannot drive one part while the page measures another.
        self.assertIn("application.findObjectById(node.uuid)", handle)

        # The two parts the page turns come from there. findNode returns a
        # raw scene node, which is the right thing to measure and the wrong
        # thing to write a rotation to.
        for part in ("platter", "volumeKnob"):
            with self.subTest(part=part):
                bound = re.search(rf"\n  const {part} = (.+);", module)
                self.assertIsNotNone(bound, msg=f"{part} is bound to nothing")
                self.assertIn("driveable(", bound.group(1))
                self.assertNotIn("findNode(", bound.group(1))

        # Measuring still reads the scene graph: a handle carries a
        # transform but no geometry to project a control against.
        self.assertIn("const node = findNode(internals.scene, name);", module)

        # A record turns at 33 1/3 rpm, a revolution every 1.8 seconds. It
        # ran at a fraction of that back when it was not turning at all,
        # which only made the standstill harder to see.
        speed = re.search(r"const PLATTER_RADIANS_PER_SECOND = (.+);", module)
        self.assertIsNotNone(speed)
        radians_per_second = eval(  # noqa: S307 - our own source, no input
            speed.group(1).replace("Math.PI", "math.pi"), {"math": math}
        )
        self.assertAlmostEqual(
            radians_per_second * 60 / (2 * math.pi), 100 / 3, places=2
        )

    def test_player_interaction_and_framing_regressions(self) -> None:
        player_root = Path(__file__).resolve().parents[1]
        page = (player_root / "app" / "player_template.html").read_text(
            encoding="utf-8"
        )
        app_module = (
            player_root / "assets" / "player" / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")
        model_module = (
            player_root / "assets" / "player" / "player-spline-v1.mjs"
        ).read_text(encoding="utf-8")

        rail_dragging = app_module.split(
            "function installRailDragging() {", maxsplit=1
        )[1].split("function setCurrentMetadata()", maxsplit=1)[0]
        pointerdown = rail_dragging.split(
            'elements.trackFlow.addEventListener("pointerdown"', maxsplit=1
        )[1].split(
            'elements.trackFlow.addEventListener("pointermove"', maxsplit=1
        )[0]
        drag_threshold = rail_dragging.split(
            "if (dragSession.dragged === false "
            "&& Math.abs(delta) >= DRAG_THRESHOLD) {",
            maxsplit=1,
        )[1].split("}", maxsplit=1)[0]
        capture = "elements.trackFlow.setPointerCapture(event.pointerId);"
        self.assertNotIn(capture, pointerdown)
        self.assertIn(capture, drag_threshold)
        self.assertEqual(rail_dragging.count(capture), 1)
        self.assertIn(
            'window.addEventListener("pointerup", finishDrag);',
            rail_dragging,
        )
        self.assertIn(
            'window.addEventListener("pointercancel", finishDrag);',
            rail_dragging,
        )

        # The scene is published with a fixed zoom that frames it for the
        # editor's window, so on any other shape of box it is simply
        # cropped. It is measured and scaled to the box it was given.
        self.assertIn("function fitToStage", model_module)
        self.assertIn("FIT_MARGIN", model_module)

        machine_stage = page.split("    .machine-stage {", maxsplit=1)[
            1
        ].split("    }", maxsplit=1)[0]
        self.assertRegex(
            machine_stage,
            r"background:\s*(?:var\(--void\)|transparent);",
        )
        self.assertRegex(machine_stage, r"border:\s*(?:0|none);")

        hotspot_label = page.split("    .machine-hotspot::after {", maxsplit=1)[
            1
        ].split("    }", maxsplit=1)[0]
        # The object carries its own printed symbols, so the written label
        # is not drawn at rest; it appears on hover and keyboard focus.
        self.assertIn("opacity: 0;", hotspot_label)
        self.assertIn("visibility: hidden;", hotspot_label)
        hover_label = page.split(
            "    .machine-hotspot:hover::after,", maxsplit=1
        )[1].split("    }", maxsplit=1)[0]
        self.assertIn("opacity: 1;", hover_label)
        self.assertIn("visibility: visible;", hover_label)
        self.assertIn(":focus-visible::after", hover_label)

    def test_player_keeps_track_details_and_one_timeline_together(self) -> None:
        player_root = Path(__file__).resolve().parents[1]
        page = (player_root / "app" / "player_template.html").read_text(
            encoding="utf-8"
        )
        app_module = (
            player_root / "assets" / "player" / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")
        body = page.split("<body>", maxsplit=1)[1]

        removed_copy = (
            "One track. Five ways forward.",
            "Physical transport",
            "The machine is the control.",
            "Press its keys, turn its controls, or drag the reel. "
            "The object responds to the track instead of sitting behind it.",
            "Machine status",
            "Click the machine. Drag the reel to seek.",
        )
        for copy in removed_copy:
            with self.subTest(copy=copy):
                self.assertNotIn(copy, body)

        section_heading = body.split(
            '<header class="section-heading">', maxsplit=1
        )[1].split("</header>", maxsplit=1)[0]
        self.assertIn('id="title"', section_heading)
        self.assertIn('id="channel"', section_heading)
        self.assertIn(".section-heading > div", page)
        self.assertGreaterEqual(page.count("overflow-wrap: anywhere;"), 2)

        status_tag = body.split('id="status"', maxsplit=1)[0].rsplit(
            "<", maxsplit=1
        )[1] + body.split('id="status"', maxsplit=1)[1].split(
            ">", maxsplit=1
        )[0]
        self.assertIn("visually-hidden", status_tag)
        self.assertIn('aria-live="polite"', status_tag)
        self.assertIn('aria-atomic="true"', status_tag)
        alert_tag = body.split('id="player-alert"', maxsplit=1)[0].rsplit(
            "<", maxsplit=1
        )[1] + body.split('id="player-alert"', maxsplit=1)[1].split(
            ">", maxsplit=1
        )[0]
        self.assertIn('role="alert"', alert_tag)
        self.assertIn('aria-atomic="true"', alert_tag)
        self.assertIn("hidden", alert_tag)

        flow_index = body.index('class="track-flow-wrap"')
        timeline_index = body.index('class="timeline-readout"')
        machine_index = body.index('class="machine-section"')
        self.assertLess(flow_index, timeline_index)
        self.assertLess(timeline_index, machine_index)
        self.assertEqual(body.count('class="timeline-readout"'), 1)
        self.assertEqual(body.count('id="progress"'), 1)
        self.assertIn(
            'id="progress" max="1000" value="0" '
            'aria-label="Playback position"',
            body,
        )

        self.assertIn(
            "function setStatus(message)",
            app_module,
        )
        self.assertIn(
            "function showAlert(message)",
            app_module,
        )
        self.assertIn(
            'showAlert("Lost the connection to Onkith. Reconnecting.");',
            app_module,
        )
        self.assertIn(
            'showAlert("That music result is not available.");',
            app_module,
        )
        playing_branch = app_module.split(
            'if (playback.state === "playing") {', maxsplit=1
        )[1].split("} else if", maxsplit=1)[0]
        self.assertIn("clearAlert();", playing_branch)

        main_rule = page.split("    main {", maxsplit=1)[1].split(
            "    }", maxsplit=1
        )[0]
        flow_wrap_rule = page.split(
            "    .track-flow-wrap {", maxsplit=1
        )[1].split("    }", maxsplit=1)[0]
        track_flow_rule = page.split(
            "    .track-flow {", maxsplit=1
        )[1].split("    }", maxsplit=1)[0]
        self.assertIn("gap: var(--s2);", main_rule)
        self.assertIn(
            "padding-block: var(--s5) var(--s2);", flow_wrap_rule
        )
        self.assertIn("min-block-size: 23rem;", track_flow_rule)

    def test_idle_results_are_records_in_their_sleeves(self) -> None:
        player_root = Path(__file__).resolve().parents[1]
        page = (player_root / "app" / "player_template.html").read_text(
            encoding="utf-8"
        )
        app_module = (
            player_root / "assets" / "player" / "player-app-v1.mjs"
        ).read_text(encoding="utf-8")

        def rule(selector: str) -> str:
            # The selector on its own, not as one line of a shared list:
            # #current-artwork and .youtube-stage each appear in both.
            found = re.search(
                r"[^,]\n    " + re.escape(selector) + r" \{\n(.*?)\n    \}",
                page,
                re.S,
            )
            self.assertIsNotNone(found, msg=f"{selector} has no rule of its own")
            return found.group(1)

        def percentage(text: str, name: str) -> float:
            found = re.search(rf"{name}:\s*([0-9.]+)%;", text)
            self.assertIsNotNone(found, msg=f"{name} is not a percentage")
            return float(found.group(1))

        media = rule(".flow-card:not(.is-active) .flow-card-media")
        sleeve = rule(".flow-card-sleeve")
        disc = rule(".flow-card-disc")

        sleeve_width = percentage(sleeve, "inline-size")
        disc_width = percentage(disc, "inline-size")
        disc_start = percentage(disc, "inset-inline-start")
        found_aspect = re.search(r"aspect-ratio:\s*([0-9.]+);", media)
        self.assertIsNotNone(found_aspect)
        aspect = float(found_aspect.group(1))

        # The disc stands out of the jacket rather than behind it, and far
        # enough that its label clears the opening instead of hiding.
        self.assertGreater(disc_start + disc_width, sleeve_width)
        self.assertGreater(disc_start + disc_width / 2, sleeve_width)
        # It reaches the end of the card and no further, so a record never
        # laps the sleeve of the result beside it.
        self.assertAlmostEqual(disc_start + disc_width, 100.0, places=1)
        # A record is a little smaller than the jacket it came out of.
        self.assertLess(disc_width, sleeve_width)
        self.assertGreater(disc_width / sleeve_width, 0.85)
        # The jacket is square and fills the height of the row: if the box
        # ratio and the jacket width drift apart it gains dead space above
        # and below the artwork.
        self.assertAlmostEqual(sleeve_width / 100.0, 1.0 / aspect, places=2)
        # Behind the jacket, or the disc covers the artwork.
        self.assertIn("z-index: 0;", disc)
        self.assertIn("z-index: 1;", sleeve)

        # The record that is playing is the same vinyl as the four still in
        # their sleeves and as the record on the platter, and is cut to the
        # same label and spindle ratios. Being unsleeved and twice the size
        # is what marks it; a second material there reads as a different
        # object rather than the same one out of its cover.
        playing = rule(".flow-card.is-active .youtube-stage")
        for colour in ("#100e0d", "#1c1a17", "#0c0b0a"):
            with self.subTest(colour=colour):
                self.assertIn(colour, disc)
                self.assertIn(colour, playing)
        self.assertNotIn("metal", playing.lower())

        playing_label = rule("#current-artwork")
        idle_label = rule(".flow-card-disc::before")
        label_inset = percentage(playing_label, "inset")
        self.assertEqual(label_inset, percentage(idle_label, "inset"))
        # A circle centred by its inset is only round if its size agrees.
        self.assertAlmostEqual(
            percentage(playing_label, "inline-size"),
            100.0 - 2 * label_inset,
            places=1,
        )
        self.assertEqual(
            percentage(rule(".flow-card.is-active .youtube-stage::after"),
                       "inline-size"),
            percentage(rule(".flow-card-disc::after"), "inline-size"),
        )

        # The artwork is the printed jacket, and the disc is a sibling of it
        # because it has to stand outside a jacket that clips its own art.
        self.assertIn('disc.className = "flow-card-disc";', app_module)
        self.assertIn('sleeve.className = "flow-card-sleeve";', app_module)
        self.assertIn("sleeve.append(image);", app_module)
        self.assertIn("media.append(disc, sleeve);", app_module)

        active_card = app_module.split(
            "function createActiveCard(slotIndex) {", maxsplit=1
        )[1].split("function buildTrackFlow()", maxsplit=1)[0]
        # The result that is playing is the bare disc, never a sleeve.
        self.assertNotIn("flow-card-sleeve", active_card)
        self.assertIn("elements.youtubeStage", active_card)


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
