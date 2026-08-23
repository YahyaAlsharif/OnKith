"""The assistant layer V6 adds on top of V4's local pipeline.

Everything here is offline. The transport is replaced with a spy or a stub, so
these run on any machine, in CI, and without spending API quota. The real Groq
path is exercised once, deliberately and in small numbers, by
``Full_Pipeline_V6_EndToEnd_Benchmark.py``.

Three things are being defended:

* the privacy boundary - what the serialized request body actually contains,
  read off the bytes handed to the transport rather than off the arguments a
  function was called with;
* rehydration - and in particular that one turn's placeholder can never be
  filled with another turn's value, which is the defect V5 shipped with;
* graceful degradation - no key, no network, a refusal, a mute speaker. None of
  them may take the local pipeline down with them.
"""

from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import llm as llm_module
from app.assistant import (
    AssistantWorker,
    MISCONFIGURED_REPLY,
    UNCONFIGURED_REPLY,
    UNREACHABLE_REPLY,
)
from app.llm import (
    GroqClient,
    GroqSettings,
    LLMUnavailable,
    ModelGone,
    RateLimited,
    parse_command,
    redact,
)
from app.post_mask_assistant import strip_masked_wake_word
from app.rehydrate import (
    ConversationPlaceholders,
    map_from_entities,
    rehydrate,
    rehydrate_with_spans,
    unresolved_placeholders,
)


def entity(placeholder: str, text: str, start: int = 0, end: int = 0) -> dict:
    return {"placeholder": placeholder, "text": text, "start": start, "end": end}


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------


class RehydrationTests(unittest.TestCase):
    def test_one_placeholder_is_restored(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        self.assertEqual(
            rehydrate("I'll remind you to call [GIVENNAME_1] tomorrow.", mapping),
            "I'll remind you to call Fahad tomorrow.",
        )

    def test_multiple_types_are_restored(self):
        mapping = map_from_entities([
            entity("[GIVENNAME_1]", "Fahad"),
            entity("[TELEPHONENUM_1]", "555-0147"),
            entity("[EMAIL_1]", "fahad@example.org"),
        ])
        restored = rehydrate(
            "Sending [EMAIL_1] to [GIVENNAME_1] on [TELEPHONENUM_1].", mapping
        )
        self.assertEqual(restored, "Sending fahad@example.org to Fahad on 555-0147.")

    def test_a_repeated_placeholder_is_restored_every_time(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        self.assertEqual(
            rehydrate("[GIVENNAME_1], yes, [GIVENNAME_1].", mapping),
            "Fahad, yes, Fahad.",
        )

    def test_two_entities_of_one_type_stay_distinct(self):
        mapping = map_from_entities([
            entity("[GIVENNAME_1]", "Fahad"),
            entity("[GIVENNAME_2]", "Ali"),
        ])
        self.assertEqual(
            rehydrate("[GIVENNAME_2] then [GIVENNAME_1].", mapping),
            "Ali then Fahad.",
        )

    def test_an_unknown_placeholder_is_left_alone_and_reported(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        text = "Ask [GIVENNAME_1] about [ACCOUNTNUM_9]."
        # Inventing a value would be worse than saying the token out loud.
        self.assertEqual(rehydrate(text, mapping), "Ask Fahad about [ACCOUNTNUM_9].")
        self.assertEqual(unresolved_placeholders(text, mapping), [("ACCOUNTNUM", "9")])

    def test_a_malformed_placeholder_is_not_touched(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        for broken in ("[Name]", "[givenname_1]", "[GIVENNAME-1]", "GIVENNAME_1"):
            with self.subTest(broken=broken):
                self.assertEqual(rehydrate(broken, mapping), broken)

    def test_text_with_no_placeholder_is_unchanged(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        self.assertEqual(rehydrate("It is sunny today.", mapping), "It is sunny today.")

    def test_spans_point_at_the_restored_values(self):
        mapping = map_from_entities([entity("[GIVENNAME_1]", "Fahad")])
        text, spans = rehydrate_with_spans("Call [GIVENNAME_1] now.", mapping)
        self.assertEqual(text, "Call Fahad now.")
        self.assertEqual(len(spans), 1)
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], "Fahad")
        self.assertEqual(spans[0]["kind"], "GIVENNAME")

    def test_an_unresolved_placeholder_is_not_marked_as_revealed(self):
        _, spans = rehydrate_with_spans("Ask [ACCOUNTNUM_9].", {})
        self.assertEqual(spans, [])


# ---------------------------------------------------------------------------
# The V5 defect this version exists to fix
# ---------------------------------------------------------------------------


class CrossTurnCollisionTests(unittest.TestCase):
    """Two turns, one placeholder name, two different people."""

    def test_turn_two_does_not_overwrite_turn_one(self):
        placeholders = ConversationPlaceholders()

        sent_one, map_one = placeholders.adopt(
            "Remind me to call [GIVENNAME_1] tomorrow",
            [entity("[GIVENNAME_1]", "Fahad")],
        )
        sent_two, map_two = placeholders.adopt(
            "Remind me to call [GIVENNAME_1] tomorrow",
            [entity("[GIVENNAME_1]", "Ali")],
        )

        # The second turn is renumbered on the way out, so the model never sees
        # one token standing for two people.
        self.assertEqual(sent_one, "Remind me to call [GIVENNAME_1] tomorrow")
        self.assertEqual(sent_two, "Remind me to call [GIVENNAME_2] tomorrow")
        self.assertEqual(map_one, {"[GIVENNAME_1]": "Fahad"})
        self.assertEqual(map_two, {"[GIVENNAME_2]": "Ali"})

        # And a reply about either turn restores the right person. Under V5's
        # rolling map the first of these returned "Ali".
        self.assertEqual(
            placeholders.apply("I'll call [GIVENNAME_1] tomorrow."),
            "I'll call Fahad tomorrow.",
        )
        self.assertEqual(
            placeholders.apply("I'll call [GIVENNAME_2] tomorrow."),
            "I'll call Ali tomorrow.",
        )

    def test_the_wrong_name_can_never_be_restored(self):
        placeholders = ConversationPlaceholders()
        placeholders.adopt("call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Fahad")])
        placeholders.adopt("call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Ali")])
        self.assertEqual(placeholders.value_of("[GIVENNAME_1]"), "Fahad")
        self.assertEqual(placeholders.value_of("[GIVENNAME_2]"), "Ali")
        self.assertNotIn("Ali", placeholders.apply("[GIVENNAME_1]"))
        self.assertNotIn("Fahad", placeholders.apply("[GIVENNAME_2]"))

    def test_the_same_person_keeps_one_number_across_turns(self):
        placeholders = ConversationPlaceholders()
        first, _ = placeholders.adopt(
            "call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Fahad")]
        )
        second, _ = placeholders.adopt(
            "text [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Fahad")]
        )
        self.assertEqual(first, "call [GIVENNAME_1]")
        self.assertEqual(second, "text [GIVENNAME_1]")
        self.assertEqual(len(placeholders), 1)

    def test_renumbering_is_simultaneous_and_does_not_chain(self):
        placeholders = ConversationPlaceholders()
        placeholders.adopt("hi [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Zed")])
        sent, _ = placeholders.adopt(
            "[GIVENNAME_1] and [GIVENNAME_2]",
            [entity("[GIVENNAME_1]", "Ann"), entity("[GIVENNAME_2]", "Bob")],
        )
        # Sequential replacement would have turned Ann's new [GIVENNAME_2] back
        # into [GIVENNAME_3] on the next rule.
        self.assertEqual(sent, "[GIVENNAME_2] and [GIVENNAME_3]")
        self.assertEqual(placeholders.apply(sent), "Ann and Bob")

    def test_types_are_numbered_independently(self):
        placeholders = ConversationPlaceholders()
        sent, _ = placeholders.adopt(
            "[GIVENNAME_1] on [TELEPHONENUM_1]",
            [entity("[GIVENNAME_1]", "Fahad"), entity("[TELEPHONENUM_1]", "555-0147")],
        )
        self.assertEqual(sent, "[GIVENNAME_1] on [TELEPHONENUM_1]")

    def test_an_utterance_with_no_entities_passes_through(self):
        placeholders = ConversationPlaceholders()
        sent, contributed = placeholders.adopt("what time is it", [])
        self.assertEqual(sent, "what time is it")
        self.assertEqual(contributed, {})

    def test_eviction_never_reuses_a_number(self):
        placeholders = ConversationPlaceholders(limit=2)
        for index in range(4):
            placeholders.adopt(
                "[GIVENNAME_1]", [entity("[GIVENNAME_1]", f"Person{index}")]
            )
        # The map is bounded, but the counter is not rewound: the fourth person
        # is [GIVENNAME_4], not a recycled [GIVENNAME_1] meaning someone else.
        self.assertLessEqual(len(placeholders), 2)
        self.assertEqual(placeholders.value_of("[GIVENNAME_4]"), "Person3")
        # An evicted placeholder resolves to nothing rather than to the wrong
        # person, and is spoken as the token.
        self.assertEqual(placeholders.apply("[GIVENNAME_1]"), "[GIVENNAME_1]")


# ---------------------------------------------------------------------------
# The wake word must not become the speaker's name
# ---------------------------------------------------------------------------


class WakeWordTests(unittest.TestCase):
    def test_a_placeholder_inside_the_wake_word_is_dropped(self):
        raw = "Onkith, remind me to call Fahad tomorrow"
        masked = "[GIVENNAME_1], remind me to call [GIVENNAME_2] tomorrow"
        entities = [
            entity("[GIVENNAME_1]", "Onkith", 0, 6),
            entity("[GIVENNAME_2]", "Fahad", 26, 31),
        ]
        question, kept = strip_masked_wake_word(raw, masked, entities)
        self.assertEqual(question, "remind me to call [GIVENNAME_2] tomorrow")
        self.assertEqual([item["placeholder"] for item in kept], ["[GIVENNAME_2]"])
        # The device must not end up calling the person by its own name.
        self.assertNotIn("Onkith", rehydrate(question, map_from_entities(kept)))

    def test_the_speakers_own_name_survives(self):
        raw = "Onkith, remind me to call Fahad tomorrow"
        masked = "[GIVENNAME_1], remind me to call [GIVENNAME_2] tomorrow"
        entities = [
            entity("[GIVENNAME_1]", "Onkith", 0, 6),
            entity("[GIVENNAME_2]", "Fahad", 26, 31),
        ]
        question, kept = strip_masked_wake_word(raw, masked, entities)
        self.assertEqual(
            rehydrate(question, map_from_entities(kept)),
            "remind me to call Fahad tomorrow",
        )

    def test_a_span_reaching_past_the_wake_word_is_kept(self):
        # Containment is the whole guard: a span covering real content too
        # would take the speaker's words with it.
        raw = "Onkith Fahad here"
        masked = "[GIVENNAME_1] here"
        entities = [entity("[GIVENNAME_1]", "Onkith Fahad", 0, 12)]
        question, kept = strip_masked_wake_word(raw, masked, entities)
        self.assertEqual(question, masked)
        self.assertEqual(kept, entities)

    def test_names_early_in_a_sentence_are_not_suppressed_generally(self):
        raw = "Fahad called me this morning"
        masked = "[GIVENNAME_1] called me this morning"
        entities = [entity("[GIVENNAME_1]", "Fahad", 0, 5)]
        question, kept = strip_masked_wake_word(raw, masked, entities)
        self.assertEqual(question, masked)
        self.assertEqual(kept, entities)

    def test_an_utterance_that_is_only_the_wake_word_keeps_its_text(self):
        raw = "Onkith"
        masked = "[GIVENNAME_1]"
        entities = [entity("[GIVENNAME_1]", "Onkith", 0, 6)]
        question, kept = strip_masked_wake_word(raw, masked, entities)
        self.assertEqual(question, masked)
        self.assertEqual(kept, entities)


# ---------------------------------------------------------------------------
# What actually goes on the wire
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self, *_args):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class OutboundPayloadTests(unittest.TestCase):
    """Read the serialized body, not the arguments."""

    def setUp(self):
        self.captured = []
        self._real = llm_module.urllib.request.urlopen

        def spy(request, *args, **kwargs):
            self.captured.append(request)
            return _Response(
                {"choices": [{"message": {"content": "Sure, calling [GIVENNAME_1]."}}]}
            )

        llm_module.urllib.request.urlopen = spy
        self.addCleanup(setattr, llm_module.urllib.request, "urlopen", self._real)

    def body(self):
        return json.loads(self.captured[-1].data.decode("utf-8"))

    def strings(self):
        def walk(node):
            if isinstance(node, str):
                yield node
            elif isinstance(node, dict):
                for key, value in node.items():
                    yield str(key)
                    yield from walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    yield from walk(item)

        return " ".join(walk(self.body()))

    def client(self):
        return GroqClient(settings=GroqSettings(api_key="test-key-not-real"))

    def test_only_masked_text_is_sent(self):
        self.client().reply_to("Remind me to call [GIVENNAME_1] tomorrow")
        payload = self.strings()
        self.assertIn("[GIVENNAME_1]", payload)
        self.assertNotIn("Fahad", payload)

    def test_the_placeholder_map_is_never_in_the_payload(self):
        placeholders = ConversationPlaceholders()
        masked, _ = placeholders.adopt(
            "Remind me to call [GIVENNAME_1] tomorrow",
            [entity("[GIVENNAME_1]", "Fahad")],
        )
        self.client().reply_to(masked)
        payload = self.strings()
        for forbidden in ("Fahad", "mapping", "entities", "original_text"):
            self.assertNotIn(forbidden, payload)

    def test_the_raw_transcript_is_not_hidden_anywhere_in_the_body(self):
        self.client().reply_to("Call [GIVENNAME_1] on [TELEPHONENUM_1]")
        body = self.body()
        # The body has exactly the fields the client builds and no smuggled
        # extras carrying a raw copy.
        self.assertEqual(
            set(body) - {"reasoning_effort"},
            {"model", "messages", "max_tokens", "temperature", "stream"},
        )
        for message in body["messages"]:
            self.assertEqual(set(message), {"role", "content"})

    def test_the_api_key_travels_as_a_header_and_not_in_the_body(self):
        self.client().reply_to("What time is it")
        self.assertNotIn("test-key-not-real", json.dumps(self.body()))
        header_names = {name.lower() for name in self.captured[-1].headers}
        self.assertIn("authorization", header_names)

    def test_history_kept_for_the_model_is_masked_only(self):
        client = self.client()
        client.reply_to("Call [GIVENNAME_1]")
        client.reply_to("And [GIVENNAME_2] as well")
        payload = self.strings()
        # Turn one is in turn two's request, still masked.
        self.assertIn("Call [GIVENNAME_1]", payload)
        self.assertNotIn("Fahad", payload)

    def test_reset_clears_the_history(self):
        client = self.client()
        client.reply_to("Call [GIVENNAME_1]")
        client.reset()
        client.reply_to("What time is it")
        self.assertNotIn("Call [GIVENNAME_1]", self.strings())

    def test_an_api_key_is_redacted_from_any_message(self):
        self.assertEqual(
            redact("failed for key sk-abc123", "sk-abc123"),
            "failed for key [REDACTED]",
        )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def http_error(code: str | int, detail: str = "{}"):
    import io

    return urllib.error.HTTPError(
        "https://api.groq.com/openai/v1/chat/completions",
        int(code),
        "error",
        {},
        io.BytesIO(detail.encode("utf-8")),
    )


class FailureHandlingTests(unittest.TestCase):
    def test_no_key_raises_the_distinct_unavailable_error(self):
        client = GroqClient(settings=GroqSettings(api_key=""))
        self.assertFalse(client.configured)
        with self.assertRaises(LLMUnavailable):
            client.reply_to("What time is it")

    def test_a_network_error_is_reported_as_a_runtime_failure(self):
        real = llm_module.urllib.request.urlopen

        def boom(*_a, **_k):
            raise urllib.error.URLError("no route to host")

        llm_module.urllib.request.urlopen = boom
        self.addCleanup(setattr, llm_module.urllib.request, "urlopen", real)
        client = GroqClient(settings=GroqSettings(api_key="k"))
        with self.assertRaises(RuntimeError) as caught:
            client.reply_to("hello")
        self.assertIn("unreachable", str(caught.exception))

    def test_a_429_falls_back_to_the_other_model(self):
        real = llm_module.urllib.request.urlopen
        seen = []

        def refuse_primary(request, *_a, **_k):
            model = json.loads(request.data.decode("utf-8"))["model"]
            seen.append(model)
            if model == "openai/gpt-oss-20b":
                raise http_error(429, '{"error":"rate limit"}')
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        llm_module.urllib.request.urlopen = refuse_primary
        self.addCleanup(setattr, llm_module.urllib.request, "urlopen", real)
        client = GroqClient(settings=GroqSettings(api_key="k"))
        self.assertEqual(client.reply_to("hello"), "ok")
        self.assertEqual(seen, ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])

    def test_a_429_on_both_models_is_raised(self):
        real = llm_module.urllib.request.urlopen
        llm_module.urllib.request.urlopen = lambda *_a, **_k: (_ for _ in ()).throw(
            http_error(429, '{"error":"rate limit"}')
        )
        self.addCleanup(setattr, llm_module.urllib.request, "urlopen", real)
        client = GroqClient(settings=GroqSettings(api_key="k"))
        with self.assertRaises(RateLimited):
            client.reply_to("hello")

    def test_a_withdrawn_model_name_is_told_apart_from_a_network_fault(self):
        real = llm_module.urllib.request.urlopen
        llm_module.urllib.request.urlopen = lambda *_a, **_k: (_ for _ in ()).throw(
            http_error(404, '{"error":{"code":"model_not_found"}}')
        )
        self.addCleanup(setattr, llm_module.urllib.request, "urlopen", real)
        client = GroqClient(
            settings=GroqSettings(api_key="k", model="gone", fallback_model="gone")
        )
        with self.assertRaises(ModelGone):
            client.reply_to("hello")

    def test_an_unknown_command_name_is_read_as_prose(self):
        self.assertIsNone(parse_command("COMMAND: SELF_DESTRUCT"))
        self.assertIsNone(parse_command("I could say COMMAND: LIGHT_ON but won't"))
        self.assertEqual(parse_command("COMMAND: LIGHT_ON"), "LIGHT_ON")


# ---------------------------------------------------------------------------
# The worker, end to end, with the transport and the speaker replaced
# ---------------------------------------------------------------------------


class RecordingBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, payload):
        self.events.append((topic, payload))


class RecordingSpeech:
    def __init__(self, fail=False):
        self.spoken = []
        self.fail = fail

    def speak(self, text):
        if self.fail:
            raise RuntimeError("no audio device")
        self.spoken.append(text)


class ScriptedLLM:
    configured = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.asked = []

    def reply_to(self, masked_text):
        self.asked.append(masked_text)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class AssistantWorkerTests(unittest.TestCase):
    def drive(self, llm, speech, submissions):
        bus = RecordingBus()
        restored = []
        exchanges = []
        worker = AssistantWorker(
            llm=llm,
            speech=speech,
            bus=bus,
            on_restored=restored.append,
            on_exchange=lambda direction, text, note: exchanges.append(
                (direction, text, note)
            ),
        )
        worker.start()
        for masked, entities in submissions:
            worker.submit(masked, entities)
        worker.shutdown()
        return bus, restored, exchanges

    def test_the_spoken_answer_is_the_restored_one(self):
        llm = ScriptedLLM(["Sure, I'll remind you to call [GIVENNAME_1] tomorrow."])
        speech = RecordingSpeech()
        _, restored, exchanges = self.drive(
            llm,
            speech,
            [(
                "Remind me to call [GIVENNAME_1] tomorrow",
                [entity("[GIVENNAME_1]", "Fahad")],
            )],
        )
        self.assertEqual(speech.spoken, ["Sure, I'll remind you to call Fahad tomorrow."])
        self.assertEqual(restored, ["Sure, I'll remind you to call Fahad tomorrow."])
        # And what crossed the boundary, both ways, was masked.
        for _, text, _ in exchanges:
            self.assertNotIn("Fahad", text)

    def test_two_turns_restore_their_own_values(self):
        llm = ScriptedLLM([
            "Calling [GIVENNAME_1].",
            "Calling [GIVENNAME_2] as well.",
        ])
        speech = RecordingSpeech()
        self.drive(
            llm,
            speech,
            [
                ("Call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Fahad")]),
                ("Call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Ali")]),
            ],
        )
        self.assertEqual(speech.spoken, ["Calling Fahad.", "Calling Ali as well."])
        # The second question was renumbered before it was asked.
        self.assertEqual(llm.asked[1], "Call [GIVENNAME_2]")

    def test_a_missing_key_keeps_the_pipeline_up_and_says_so(self):
        class Unconfigured:
            configured = False

            def reply_to(self, masked_text):
                raise LLMUnavailable("GROQ_API_KEY is not set")

        speech = RecordingSpeech()
        _, restored, exchanges = self.drive(
            Unconfigured(), speech, [("What time is it", [])]
        )
        self.assertEqual(restored, [UNCONFIGURED_REPLY])
        # The masked question is still shown, so the demonstration survives a
        # venue with no wifi and no key.
        self.assertEqual(exchanges[0][0], "unsent")
        self.assertIn("GROQ_API_KEY", exchanges[0][2])

    def test_a_network_failure_degrades_without_taking_the_worker_down(self):
        llm = ScriptedLLM([RuntimeError("Groq is unreachable: timed out")])
        speech = RecordingSpeech()
        _, restored, _ = self.drive(llm, speech, [("What time is it", [])])
        self.assertEqual(restored, [UNREACHABLE_REPLY])

    def test_a_withdrawn_model_says_so_rather_than_blaming_the_wifi(self):
        llm = ScriptedLLM([ModelGone("no such model")])
        speech = RecordingSpeech()
        _, restored, _ = self.drive(llm, speech, [("What time is it", [])])
        self.assertEqual(restored, [MISCONFIGURED_REPLY])

    def test_a_tts_failure_leaves_the_text_answer_intact(self):
        llm = ScriptedLLM(["Calling [GIVENNAME_1]."])
        speech = RecordingSpeech(fail=True)
        _, restored, _ = self.drive(
            llm, speech, [("Call [GIVENNAME_1]", [entity("[GIVENNAME_1]", "Fahad")])]
        )
        self.assertEqual(restored, ["Calling Fahad."])

    def test_a_command_reply_is_carried_out_and_never_spoken(self):
        bus = RecordingBus()
        speech = RecordingSpeech()
        intents = []
        worker = AssistantWorker(
            llm=ScriptedLLM(["COMMAND: LIGHT_ON"]), speech=speech, bus=bus
        )
        worker.start()
        worker.submit("turn the lamp on", [], on_command=intents.append)
        worker.shutdown()
        self.assertEqual(intents, ["LIGHT_ON"])
        self.assertEqual(speech.spoken, [])

    def test_an_unresolvable_placeholder_is_spoken_as_the_token(self):
        llm = ScriptedLLM(["Ask [ACCOUNTNUM_9] about it."])
        speech = RecordingSpeech()
        _, restored, _ = self.drive(llm, speech, [("Ask about it", [])])
        self.assertEqual(restored, ["Ask [ACCOUNTNUM_9] about it."])


# ---------------------------------------------------------------------------
# The no-key path, proved in a real interpreter
# ---------------------------------------------------------------------------


class MissingKeySubprocessTests(unittest.TestCase):
    """Run the real startup path with no key, without touching anybody's .env.

    An exported empty ``GROQ_API_KEY`` wins over the file, because the loader
    uses ``setdefault``. That is the whole reason the no-key mode can be tested
    on a machine that has a key sitting in the repository root: nothing is
    renamed, moved or deleted to produce it.
    """

    def test_the_local_pipeline_stays_up_without_a_key(self):
        import os
        import subprocess

        script = (
            "import os, sys;"
            "sys.path.insert(0, r'" + str(ROOT) + "');"
            "from app.omit_ai_live import load_local_env;"
            "load_local_env();"
            "from app.llm import GroqClient;"
            "from app.speech import detect_backend;"
            "from app.rehydrate import ConversationPlaceholders;"
            "c = GroqClient();"
            "p = ConversationPlaceholders();"
            "sent, _ = p.adopt('call [GIVENNAME_1]',"
            " [{'placeholder': '[GIVENNAME_1]', 'text': 'Fahad'}]);"
            "print('configured', c.configured);"
            "print('masking', sent);"
            "print('restored', p.apply('calling [GIVENNAME_1]'));"
            "print('tts', detect_backend(lambda m: None).name)"
        )
        environment = dict(os.environ)
        environment["GROQ_API_KEY"] = ""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            env=environment,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout

        # The LLM knows it has no key ...
        self.assertIn("configured False", output)
        # ... and everything local still works.
        self.assertIn("masking call [GIVENNAME_1]", output)
        self.assertIn("restored calling Fahad", output)
        self.assertIn("tts ", output)

    def test_an_exported_key_beats_the_file(self):
        import os
        import subprocess

        script = (
            "import os, sys;"
            "sys.path.insert(0, r'" + str(ROOT) + "');"
            "from app.omit_ai_live import load_local_env;"
            "load_local_env();"
            "print('key', os.environ.get('GROQ_API_KEY'));"
            "print('model', os.environ.get('GROQ_MODEL'))"
        )
        environment = dict(os.environ)
        environment["GROQ_API_KEY"] = "exported-wins"
        environment["GROQ_MODEL"] = "exported-model"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("key exported-wins", completed.stdout)
        self.assertIn("model exported-model", completed.stdout)


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------


class SpeakableTextTests(unittest.TestCase):
    """A synthesizer must not be the reason an answer is lost.

    The narrow no-break space in this test is not hypothetical. A live Groq
    reply contained "6\u202fp.m."; handing it to PowerShell on a machine whose
    console codepage is cp1256 raised UnicodeEncodeError, and the reply was
    shown on screen but never spoken.
    """

    def test_model_typography_is_folded_to_plain_characters(self):
        from app.speech import speakable

        self.assertEqual(
            speakable("She\u2019s free after 6\u202fp.m.\u2014ok\u2026"),
            "She's free after 6 p.m.-ok...",
        )

    def test_the_folded_text_survives_the_narrowest_codepage(self):
        from app.speech import speakable

        # cp1256 is what broke it. Nothing left after folding may fail to
        # encode there.
        folded = speakable("She\u2019s free after 6\u202fp.m. \u2013 note\u2026")
        folded.encode("cp1256")

    def test_ordinary_text_is_untouched(self):
        from app.speech import speakable

        for text in ("Calling Fahad.", "", "Plain ASCII, 555-0147."):
            with self.subTest(text=text):
                self.assertEqual(speakable(text), text)


if __name__ == "__main__":
    unittest.main()
