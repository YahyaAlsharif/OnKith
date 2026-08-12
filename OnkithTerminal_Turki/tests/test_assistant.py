"""Groq client, assistant worker, and the masked-only privacy boundary."""

from __future__ import annotations

import io
import json
import sys
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assistant import UNREACHABLE_REPLY, AssistantWorker
from app.event_bus import EventBus
from app.llm import (
    DEFAULT_MODEL,
    GROQ_URL,
    SYSTEM_PROMPT,
    GroqClient,
    GroqSettings,
    LLMUnavailable,
    RateLimited,
    ambient_context,
    extract_reply,
    redact,
)
from app.omit_ai_live import forward_masked_output


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def groq_response(text: str) -> FakeResponse:
    return FakeResponse(
        json.dumps({"choices": [{"message": {"content": text}}]}).encode()
    )


def groq_rate_limit(model: str) -> urllib.error.HTTPError:
    """The refusal Groq actually sends, body and all."""

    body = json.dumps(
        {
            "error": {
                "message": (
                    f"Rate limit reached for model `{model}` in organization "
                    "`org_test` service tier `on_demand` on tokens per minute "
                    "(TPM): Limit 8000, Used 6711, Requested 3361."
                ),
                "type": "compound",
                "code": "rate_limit_exceeded",
            }
        }
    ).encode()
    return urllib.error.HTTPError(
        GROQ_URL, 429, "Too Many Requests", {}, io.BytesIO(body)
    )


class RateLimitedGroq:
    """Refuses one named model and answers for anything else."""

    def __init__(self, refuse: str, reply: str = "answered") -> None:
        self.refuse = refuse
        self.reply = reply
        self.asked: list[str] = []

    def __call__(self, request, timeout=None):
        model = json.loads(request.data)["model"]
        self.asked.append(model)
        if model == self.refuse:
            raise groq_rate_limit(model)
        return groq_response(self.reply)


class FallbackTest(unittest.TestCase):
    """A rate limit is not an answer.

    The compound model searches the web on the server, which costs about
    3400 tokens a question against a free tier of 8000 a minute. Measured,
    it refused roughly every other call. Somebody standing in front of the
    device asked a question, so it drops to the cheaper model and replies.
    """

    def settings(self, **overrides) -> GroqSettings:
        base = {
            "api_key": "test-key",
            "model": "groq/compound-mini",
            "fallback_model": "llama-3.3-70b-versatile",
        }
        base.update(overrides)
        return GroqSettings(**base)

    def test_a_refusal_is_answered_by_the_other_model(self) -> None:
        transport = RateLimitedGroq(refuse="groq/compound-mini")
        client = GroqClient(settings=self.settings())
        with patch("urllib.request.urlopen", transport):
            self.assertEqual(client.reply_to("what time is it"), "answered")
        self.assertEqual(
            transport.asked, ["groq/compound-mini", "llama-3.3-70b-versatile"]
        )

    def test_the_next_question_does_not_pay_for_the_discovery_again(self) -> None:
        # A round trip spent rediscovering a limit that is still there is a
        # round trip the person in front of the device waits through.
        transport = RateLimitedGroq(refuse="groq/compound-mini")
        client = GroqClient(settings=self.settings())
        with patch("urllib.request.urlopen", transport):
            client.reply_to("first")
            client.reply_to("second")
        self.assertEqual(
            transport.asked,
            [
                "groq/compound-mini",
                "llama-3.3-70b-versatile",
                "llama-3.3-70b-versatile",
            ],
        )

    def test_the_better_model_comes_back_once_the_window_passes(self) -> None:
        transport = RateLimitedGroq(refuse="groq/compound-mini")
        client = GroqClient(settings=self.settings())
        with patch("urllib.request.urlopen", transport):
            client.reply_to("first")
            client._rate_limited_until = time.monotonic() - 1
            transport.refuse = "nothing"
            client.reply_to("second")
        self.assertEqual(transport.asked[-1], "groq/compound-mini")

    def test_a_refusal_from_the_fallback_itself_is_raised(self) -> None:
        # There is nowhere left to go, and pretending otherwise would loop.
        transport = RateLimitedGroq(refuse="llama-3.3-70b-versatile")
        client = GroqClient(settings=self.settings(model="llama-3.3-70b-versatile"))
        with patch("urllib.request.urlopen", transport):
            with self.assertRaises(RuntimeError):
                client.reply_to("what time is it")
        self.assertEqual(transport.asked, ["llama-3.3-70b-versatile"])

    def test_falling_back_to_the_model_that_refused_is_not_a_fallback(self) -> None:
        same = self.settings(model="x", fallback_model="x")
        self.assertFalse(same.has_fallback)
        self.assertTrue(self.settings().has_fallback)

    def test_every_other_failure_is_still_a_plain_failure(self) -> None:
        """Only 429 means another model might work. A 401 means it will not."""

        def refuse(request, timeout=None):
            raise urllib.error.HTTPError(
                GROQ_URL, 401, "Unauthorized", {}, io.BytesIO(b"{}")
            )

        client = GroqClient(settings=self.settings())
        with patch("urllib.request.urlopen", refuse):
            with self.assertRaises(RuntimeError) as caught:
                client.reply_to("what time is it")
        self.assertNotIsInstance(caught.exception, RateLimited)

    def test_the_answer_that_came_back_is_the_one_remembered(self) -> None:
        # The history has to hold what was actually said, whichever model
        # said it, or the next turn refers to a reply that never happened.
        transport = RateLimitedGroq(refuse="groq/compound-mini", reply="Tokyo")
        client = GroqClient(settings=self.settings())
        with patch("urllib.request.urlopen", transport):
            client.reply_to("capital of Japan")
        self.assertEqual(
            client._history,
            [
                {"role": "user", "content": "capital of Japan"},
                {"role": "assistant", "content": "Tokyo"},
            ],
        )

    def test_the_fallback_is_configurable_without_touching_code(self) -> None:
        with patch.dict(
            "os.environ", {"GROQ_FALLBACK_MODEL": "llama-3.1-8b-instant"}, clear=False
        ):
            self.assertEqual(
                GroqSettings.from_environment().fallback_model,
                "llama-3.1-8b-instant",
            )


class RecordingSpeech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeLLM:
    configured = True

    def __init__(self, reply: str = "All good.", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.seen: list[str] = []

    def reply_to(self, masked_text: str) -> str:
        self.seen.append(masked_text)
        if self.error is not None:
            raise self.error
        return self.reply


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class GroqClientTest(unittest.TestCase):
    def client(self, **overrides) -> GroqClient:
        settings = GroqSettings(api_key="test-key", **overrides)
        return GroqClient(settings=settings)

    def test_without_a_key_the_assistant_reports_it(self) -> None:
        client = GroqClient(settings=GroqSettings(api_key=""))
        self.assertFalse(client.configured)
        with self.assertRaises(LLMUnavailable):
            client.reply_to("hello")

    def test_the_request_carries_the_model_and_masked_text(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode())
            return groq_response("Sure.")

        client = self.client()
        with patch("urllib.request.urlopen", fake_urlopen):
            reply = client.reply_to("Call [GIVENNAME_1] later")

        self.assertEqual(reply, "Sure.")
        self.assertIn("api.groq.com", captured["url"])
        self.assertEqual(captured["body"]["model"], DEFAULT_MODEL)
        messages = captured["body"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["content"], "Call [GIVENNAME_1] later")
        # The system prompt has to explain the placeholders, or the reply
        # tries to guess what was behind them.
        self.assertIn("[GIVENNAME_1]", messages[0]["content"])

    def test_the_model_is_told_what_moment_it_is(self) -> None:
        """A model has no clock, so without this it cannot answer the most
        ordinary question anybody asks a thing sitting in their kitchen."""

        from datetime import datetime, timedelta, timezone

        moment = datetime(2026, 8, 6, 11, 52, tzinfo=timezone(timedelta(hours=3)))
        context = ambient_context(moment)
        for expected in ("11:52", "Thursday", "August", "2026", "UTC+03:00"):
            with self.subTest(expected=expected):
                self.assertIn(expected, context)

    def test_the_moment_is_stamped_on_every_request(self) -> None:
        # Not once at start up: this device is left running for days.
        source = (ROOT / "app" / "llm.py").read_text(encoding="utf-8")
        messages = source.split("def _messages(", maxsplit=1)[1].split("\n    def")[0]
        self.assertIn("ambient_context()", messages)

    def test_a_placeholder_may_be_echoed_but_never_authored(self) -> None:
        """It used to be told to ask for any detail it could not see.

        Sitting next to three worked examples of placeholder syntax, that
        taught it to treat anything unknown as one: asked the time, it
        answered "[TIME_1]", which no map could fill because nobody had ever
        said it.
        """

        self.assertIn(
            "Never write a placeholder that was not already in the message",
            SYSTEM_PROMPT,
        )
        self.assertIn("say so in plain words", SYSTEM_PROMPT)
        self.assertNotIn("ask for just that detail", SYSTEM_PROMPT)

    def test_a_command_that_went_astray_is_answered_as_a_command(self) -> None:
        """Routing reads the first word after the name and nothing else.

        So "Onkith can you play bones" reaches the assistant, and so does
        every word the recogniser gets wrong. Told nothing about the music,
        it answers that it does not understand, which is the one reply that
        helps nobody. It cannot run the command, but it can say the wording
        that would have worked.
        """

        self.assertIn("plays music", SYSTEM_PROMPT)
        for wording in ("stop", "show the player", "close the player"):
            with self.subTest(wording=wording):
                self.assertIn(wording, SYSTEM_PROMPT)
        self.assertIn("Never say you have started or stopped anything", SYSTEM_PROMPT)

    def test_it_never_offers_a_command_the_voice_lane_does_not_have(self) -> None:
        """Every transport word by voice calls close(), which stops the music.

        So "skip" is recognised and then stops the track, and a device that
        tells somebody to say skip has told them something untrue. The
        buttons for skipping and pausing exist, on the recorder on screen.

        Volume used to be on that list and is not any more, because there
        is a lane for it now. This test is the reminder that the two have
        to move together: a wording offered here that no lane recognises
        is the same lie in the other direction.
        """

        self.assertIn("do not offer", SYSTEM_PROMPT)
        self.assertIn("stop the music instead", SYSTEM_PROMPT)
        self.assertNotIn("changing the volume", SYSTEM_PROMPT)

    def test_every_wording_it_offers_is_one_the_lane_recognises(self) -> None:
        """The prompt names the commands, and the lane decides them.

        Nothing keeps those two in step but this. A wording that reads
        well in the prompt and routes to conversation is a device that
        tells you how to talk to it and then does not listen.
        """

        from app.onkith_music import intercept_onkith_command

        offered = {
            "play Bones": "PLAY",
            "stop": "CLOSE",
            "show the player": "SHOW",
            "close the player": "HIDE",
            "louder": "LOUDER",
            "quieter": "QUIETER",
            "turn the light on": "LIGHT_ON",
            "turn the light off": "LIGHT_OFF",
            "brighter": "BRIGHTER",
            "dimmer": "DIMMER",
        }
        for wording, intent in offered.items():
            with self.subTest(wording=wording):
                self.assertIn(wording.split(" ")[0], SYSTEM_PROMPT)
                self.assertEqual(
                    intercept_onkith_command(f"Onkith {wording}").intent,
                    intent,
                )

    def test_the_list_of_commands_is_not_the_answer_to_everything(self) -> None:
        """Told "my ID is 1234, how are you doing", it recited the list.

        The first wording said to treat anything reading as an instruction
        as a misrouted command, and almost everything reads that way to a
        model looking for an excuse. A device that answers nearly any
        question then looks like one that answers five.
        """

        self.assertIn("ordinary conversation", SYSTEM_PROMPT)
        self.assertIn("not a command at all", SYSTEM_PROMPT)
        # The old wording, which was the cause.
        self.assertNotIn("treat it as one that went astray", SYSTEM_PROMPT)

    def test_the_key_is_sent_as_a_bearer_token(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["auth"] = request.get_header("Authorization")
            return groq_response("ok")

        with patch("urllib.request.urlopen", fake_urlopen):
            self.client().reply_to("hello")
        self.assertEqual(captured["auth"], "Bearer test-key")

    def test_history_is_kept_and_bounded(self) -> None:
        client = self.client()
        with patch("urllib.request.urlopen", lambda r, timeout=None: groq_response("ok")):
            for index in range(10):
                client.reply_to(f"turn {index}")

        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["messages"] = json.loads(request.data.decode())["messages"]
            return groq_response("ok")

        with patch("urllib.request.urlopen", fake_urlopen):
            client.reply_to("latest")

        roles = [item["role"] for item in captured["messages"]]
        self.assertEqual(roles[0], "system")
        self.assertEqual(captured["messages"][-1]["content"], "latest")
        # System prompt plus a bounded history plus the new turn.
        self.assertLessEqual(len(captured["messages"]), 14)
        self.assertIn("turn 9", json.dumps(captured["messages"]))

    def test_empty_text_makes_no_request(self) -> None:
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(request)
            return groq_response("ok")

        with patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(self.client().reply_to("   "), "")
        self.assertEqual(calls, [])

    def test_a_malformed_response_yields_no_reply(self) -> None:
        for body in ({}, {"choices": []}, {"choices": [{}]}, []):
            with self.subTest(body=body):
                self.assertEqual(extract_reply(body), "")

    def test_the_api_key_is_redacted_from_errors(self) -> None:
        self.assertNotIn("test-key", redact("failed for test-key", "test-key"))
        self.assertIn("[REDACTED]", redact("failed for test-key", "test-key"))

    def test_a_transport_failure_is_reported_not_swallowed(self) -> None:
        def fake_urlopen(request, timeout=None):
            raise OSError("connection reset")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(RuntimeError):
                self.client().reply_to("hello")


class AssistantWorkerTest(unittest.TestCase):
    def build(self, llm):
        bus = EventBus()
        speech = RecordingSpeech()
        worker = AssistantWorker(llm=llm, speech=speech, bus=bus)
        worker.start()
        self.addCleanup(worker.shutdown)
        return worker, bus, speech

    def test_a_masked_utterance_becomes_a_spoken_reply(self) -> None:
        llm = FakeLLM("The bins go out on Tuesday.")
        worker, bus, speech = self.build(llm)
        worker.submit("when do the bins go out")
        self.assertTrue(wait_for(lambda: speech.spoken))
        self.assertEqual(llm.seen, ["when do the bins go out"])
        self.assertEqual(speech.spoken, ["The bins go out on Tuesday."])

    def test_the_exchange_is_retained_for_a_surface(self) -> None:
        worker, bus, _ = self.build(FakeLLM("Yes."))
        worker.submit("are you there")
        self.assertTrue(
            wait_for(lambda: len(bus.snapshot()["conversation"]["turns"]) == 2)
        )
        turns = bus.snapshot()["conversation"]["turns"]
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[0]["text"], "are you there")
        self.assertEqual(turns[1]["role"], "onkith")
        self.assertEqual(turns[1]["text"], "Yes.")
        # Nothing was masked, so there is nothing for the screen to mark.
        self.assertEqual(turns[0]["revealed"], [])
        self.assertEqual(turns[1]["revealed"], [])

    def test_the_screen_shows_what_was_said_and_the_model_never_does(
        self,
    ) -> None:
        """Three audiences, two of them in the room.

        The model gets placeholders, which is the whole point. The screen is
        served from localhost to a browser in the same room as the mouth that
        said the words, so reading "[GIVENNAME_1]" back to somebody about
        their own name would be theatre rather than privacy.
        """

        llm = FakeLLM("Noted, [GIVENNAME_1]. I will call [TELEPHONENUM_1].")
        worker, bus, speech = self.build(llm)
        worker.submit(
            "Call [GIVENNAME_1] on [TELEPHONENUM_1]",
            [
                {"placeholder": "[GIVENNAME_1]", "text": "Turki"},
                {"placeholder": "[TELEPHONENUM_1]", "text": "0551234567"},
            ],
        )
        self.assertTrue(
            wait_for(lambda: len(bus.snapshot()["conversation"]["turns"]) == 2)
        )

        # What left the device is unchanged by any of this.
        self.assertEqual(llm.seen, ["Call [GIVENNAME_1] on [TELEPHONENUM_1]"])

        turns = bus.snapshot()["conversation"]["turns"]
        self.assertEqual(turns[0]["text"], "Call Turki on 0551234567")
        self.assertEqual(
            turns[1]["text"], "Noted, Turki. I will call 0551234567."
        )
        self.assertEqual(
            speech.spoken, ["Noted, Turki. I will call 0551234567."]
        )

    def test_a_placeholder_with_no_value_is_left_alone_on_the_screen(
        self,
    ) -> None:
        # Inventing a value would be worse than showing the token, and it
        # makes the gap visible rather than silent.
        worker, bus, _ = self.build(FakeLLM("About [GIVENNAME_9], yes."))
        worker.submit("a question", [])
        self.assertTrue(
            wait_for(lambda: len(bus.snapshot()["conversation"]["turns"]) == 2)
        )
        turns = bus.snapshot()["conversation"]["turns"]
        self.assertEqual(turns[1]["text"], "About [GIVENNAME_9], yes.")

    def test_thinking_is_published_while_the_model_runs(self) -> None:
        seen: list[str] = []
        llm = FakeLLM("done")
        worker, bus, _ = self.build(llm)
        bus.subscribe(
            lambda message: seen.append(message.get("state"))
            if message.get("type") == "assistant"
            else None
        )
        worker.submit("hello")
        self.assertTrue(wait_for(lambda: "thinking" in seen))

    def test_a_failure_speaks_once_and_not_on_every_retry(self) -> None:
        llm = FakeLLM(error=RuntimeError("Groq is unreachable"))
        worker, _, speech = self.build(llm)
        for _ in range(3):
            worker.submit("hello")
        self.assertTrue(wait_for(lambda: len(llm.seen) == 3))
        time.sleep(0.2)
        self.assertEqual(speech.spoken, [UNREACHABLE_REPLY])

    def test_a_missing_key_says_nothing_aloud(self) -> None:
        llm = FakeLLM(error=LLMUnavailable("no key"))
        worker, _, speech = self.build(llm)
        worker.submit("hello")
        self.assertTrue(wait_for(lambda: llm.seen))
        time.sleep(0.2)
        self.assertEqual(speech.spoken, [])

    def test_empty_text_is_never_sent(self) -> None:
        llm = FakeLLM()
        worker, _, _ = self.build(llm)
        worker.submit("  ")
        worker.submit("")
        time.sleep(0.2)
        self.assertEqual(llm.seen, [])


class PrivacyBoundaryTest(unittest.TestCase):
    """Nothing but cleared, masked text may reach a network model."""

    class Result:
        def __init__(
            self,
            masked_text: str,
            safe: bool,
            entities=None,
            original_text: str | None = None,
        ) -> None:
            self.pii = type(
                "PII",
                (),
                {
                    "masked_text": masked_text,
                    "safe_to_release": safe,
                    "entities": entities or [],
                    # Whether the device was spoken to is read from the
                    # unmasked text, which never leaves here. The masked copy
                    # is the only thing that travels.
                    "original_text": (
                        masked_text if original_text is None else original_text
                    ),
                },
            )()

    class Collector:
        def __init__(self) -> None:
            self.submitted: list[str] = []
            self.entities: list[list] = []

        def submit(self, text: str, entities=None) -> None:
            self.submitted.append(text)
            self.entities.append(list(entities or ()))

    def test_cleared_masked_text_is_forwarded(self) -> None:
        assistant = self.Collector()
        shown = []
        entities = [{"placeholder": "[GIVENNAME_1]", "text": "Turki"}]
        result = self.Result("Onkith who is [GIVENNAME_1]", True, entities)
        forward_masked_output(result, assistant, shown.append)
        # Masked, and without the name that summoned it. The placeholder
        # is the privacy boundary holding; the missing "Onkith" is why
        # the model stopped calling the person Onkith back.
        self.assertEqual(assistant.submitted, ["who is [GIVENNAME_1]"])
        # The span map travels with it so the reply can be rehydrated
        # locally before it is spoken.
        self.assertEqual(assistant.entities, [entities])
        self.assertEqual(shown, [result])

    def test_blocked_text_is_shown_but_never_forwarded(self) -> None:
        assistant = self.Collector()
        shown = []
        result = self.Result("Onkith call Turki on 555 123 4567", False)
        forward_masked_output(result, assistant, shown.append)
        self.assertEqual(assistant.submitted, [])
        # It still reaches the display, so the failure is visible.
        self.assertEqual(shown, [result])

    def test_no_assistant_still_displays(self) -> None:
        shown = []
        result = self.Result("Onkith anything", True)
        forward_masked_output(result, None, shown.append)
        self.assertEqual(shown, [result])

    def test_speech_the_device_was_not_addressed_with_is_never_forwarded(
        self,
    ) -> None:
        """The device sits in a room and hears all of it.

        Answering an overheard sentence is the wrong behaviour, and sending
        one to a network model is a conversation nobody agreed to release.
        """

        assistant = self.Collector()
        shown = []
        overheard = [
            "I was telling Sarah about the meeting",
            "we should stop by the shop later",
            "play it again from the top",
            "someone named Ankit called me",
        ]
        for text in overheard:
            with self.subTest(text=text):
                result = self.Result(text, True)
                forward_masked_output(result, assistant, shown.append)
                self.assertEqual(assistant.submitted, [])
        # Every one of them still reaches the transcript and the display.
        self.assertEqual(len(shown), len(overheard))

    def test_a_music_command_is_not_a_question(self) -> None:
        # It was already handled by the music worker, and the assistant has
        # nothing to say about it.
        assistant = self.Collector()
        for text in ("Onkith play imagine dragons bones", "Onkith stop"):
            with self.subTest(text=text):
                forward_masked_output(
                    self.Result(text, True), assistant, lambda _: None
                )
                self.assertEqual(assistant.submitted, [])

    class Gate:
        """Stands in for the core, which is the only thing that has a bus."""

        def __init__(self) -> None:
            self.states: list[str] = []

        def set_gate(self, state: str) -> None:
            self.states.append(state)

    def test_the_screen_is_told_what_became_of_every_utterance(self) -> None:
        """Held, sent or blocked, in the same three words everywhere.

        This is the one readout that says whether anything left, so it has
        to be driven by the boundary that actually decides rather than
        inferred from something that happens to correlate.
        """

        gate = self.Gate()
        assistant = self.Collector()
        forward_masked_output(
            self.Result("Onkith what is the time", True),
            assistant,
            lambda _: None,
            gate,
        )
        forward_masked_output(
            self.Result("we should leave at six", True),
            assistant,
            lambda _: None,
            gate,
        )
        forward_masked_output(
            self.Result("Call Turki on 555 123 4567", False),
            assistant,
            lambda _: None,
            gate,
        )
        self.assertEqual(gate.states, ["sent", "held", "blocked"])

    def test_a_surface_that_fails_cannot_stop_the_pipeline(self) -> None:
        class Broken:
            def set_gate(self, state: str) -> None:
                raise RuntimeError("the screen went away")

        assistant = self.Collector()
        forward_masked_output(
            self.Result("Onkith what is the time", True),
            assistant,
            lambda _: None,
            Broken(),
        )
        self.assertEqual(assistant.submitted, ["what is the time"])

    def test_addressing_is_read_from_the_unmasked_text(self) -> None:
        """The wake word can be inside a span the masker replaced.

        Reading the masked copy would then miss it and the device would sit
        there while somebody was plainly talking to it.
        """

        assistant = self.Collector()
        result = self.Result(
            "[PERSON_1] what time is it",
            True,
            original_text="Onkith what time is it",
        )
        forward_masked_output(result, assistant, lambda _: None)
        self.assertEqual(assistant.submitted, ["[PERSON_1] what time is it"])


if __name__ == "__main__":
    unittest.main()


class FailureIsVisibleTest(unittest.TestCase):
    """A question with nothing under it looks exactly like being ignored.

    Observed live: every wake word was heard, the gate said sent, and Groq
    refused both the model and its fallback because the whole account was
    over its limit. The screen showed a column of questions and not one
    reply, which is indistinguishable from a device that never woke up.
    Those two failures are nothing alike.
    """

    def worker(self, llm):
        return AssistantWorker(
            llm=llm, speech=RecordingSpeech(), bus=EventBus(), log=lambda m: None
        )

    def drain(self, worker, questions=1):
        worker.start()
        for _ in range(questions):
            worker.submit("what time is it", [])
        worker.queue.join()
        worker.shutdown()
        return [turn["role"] for turn in worker._turns]

    def test_a_refusal_reaches_the_screen_every_time(self) -> None:
        worker = self.worker(FakeLLM(error=RuntimeError("HTTP 429")))
        roles = self.drain(worker, questions=3)
        # Three questions, three answers. Written three times, spoken once:
        # repeating the same sentence aloud after every utterance is
        # unbearable, and a blank screen is worse than unbearable.
        self.assertEqual(roles.count("onkith"), 3)
        self.assertEqual(worker.speech.spoken.count(UNREACHABLE_REPLY), 1)

    def test_an_empty_reply_is_not_dropped_in_silence(self) -> None:
        # A compound model that answers inside its tool results returns
        # exactly this, and it used to leave the question with nothing
        # under it.
        worker = self.worker(FakeLLM(reply=""))
        self.assertIn("onkith", self.drain(worker))
