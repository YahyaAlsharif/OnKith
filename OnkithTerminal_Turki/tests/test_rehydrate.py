"""Placeholders go out to the model and real values come back locally."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assistant import AssistantWorker
from app.event_bus import EventBus
from app.rehydrate import (
    RehydrationMap,
    map_from_entities,
    rehydrate,
    rehydrate_with_spans,
    unresolved_placeholders,
)


class RevealedSpanTest(unittest.TestCase):
    """The screen marks the words the masker held back.

    That marking is the demonstration: the room sees the private values
    sitting in the sentence and knows the model was handed a placeholder in
    their place.
    """

    MAP = {"[GIVENNAME_1]": "Turki", "[TELEPHONENUM_1]": "0551234567"}

    def test_a_span_covers_exactly_the_value_that_was_put_back(self) -> None:
        text, spans = rehydrate_with_spans(
            "Call [GIVENNAME_1] on [TELEPHONENUM_1]", self.MAP
        )
        self.assertEqual(text, "Call Turki on 0551234567")
        self.assertEqual(
            [text[span["start"] : span["end"]] for span in spans],
            ["Turki", "0551234567"],
        )
        self.assertEqual([span["kind"] for span in spans], ["GIVENNAME", "TELEPHONENUM"])

    def test_offsets_are_into_the_text_that_gets_drawn(self) -> None:
        """Not into the masked text, whose lengths are entirely different.

        A placeholder is thirteen characters and the name behind it is five,
        so anything measured before the substitution points at the wrong
        words by the second one.
        """

        text, spans = rehydrate_with_spans(
            "[GIVENNAME_1] and [GIVENNAME_1] and [TELEPHONENUM_1]", self.MAP
        )
        self.assertEqual(text, "Turki and Turki and 0551234567")
        self.assertEqual(
            [text[span["start"] : span["end"]] for span in spans],
            ["Turki", "Turki", "0551234567"],
        )

    def test_a_placeholder_with_no_value_is_left_alone_and_not_marked(self) -> None:
        # Nothing was revealed, so there is nothing to point at.
        text, spans = rehydrate_with_spans("About [GIVENNAME_9], yes.", self.MAP)
        self.assertEqual(text, "About [GIVENNAME_9], yes.")
        self.assertEqual(spans, [])

    def test_text_with_nothing_masked_reports_nothing(self) -> None:
        text, spans = rehydrate_with_spans("It is sunny today.", self.MAP)
        self.assertEqual(text, "It is sunny today.")
        self.assertEqual(spans, [])

    def test_the_map_offers_the_same_thing(self) -> None:
        rolling = RehydrationMap()
        rolling.add([{"placeholder": "[GIVENNAME_1]", "text": "Turki"}])
        text, spans = rolling.apply_with_spans("Hello [GIVENNAME_1]")
        self.assertEqual(text, "Hello Turki")
        self.assertEqual(text[spans[0]["start"] : spans[0]["end"]], "Turki")

ENTITIES = [
    {
        "start": 8,
        "end": 13,
        "entity_type": "GIVENNAME",
        "text": "Turki",
        "placeholder": "[GIVENNAME_1]",
    },
    {
        "start": 20,
        "end": 32,
        "entity_type": "TELEPHONENUM",
        "text": "555 123 4567",
        "placeholder": "[TELEPHONENUM_1]",
    },
]


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class RehydrateTest(unittest.TestCase):
    def test_the_map_is_built_from_the_masker_record(self) -> None:
        mapping = map_from_entities(ENTITIES)
        self.assertEqual(mapping["[GIVENNAME_1]"], "Turki")
        self.assertEqual(mapping["[TELEPHONENUM_1]"], "555 123 4567")

    def test_placeholders_are_replaced(self) -> None:
        mapping = map_from_entities(ENTITIES)
        spoken = rehydrate(
            "I will remind [GIVENNAME_1] on [TELEPHONENUM_1].", mapping
        )
        self.assertEqual(spoken, "I will remind Turki on 555 123 4567.")

    def test_a_placeholder_used_twice_is_replaced_each_time(self) -> None:
        mapping = map_from_entities(ENTITIES)
        spoken = rehydrate(
            "[GIVENNAME_1]? Yes, [GIVENNAME_1] is on the list.", mapping
        )
        self.assertEqual(spoken, "Turki? Yes, Turki is on the list.")

    def test_an_unknown_placeholder_is_left_alone(self) -> None:
        mapping = map_from_entities(ENTITIES)
        # Inventing a value would be worse than saying the token, and this
        # way the gap is audible rather than silent.
        spoken = rehydrate("Ask [SURNAME_9] about it.", mapping)
        self.assertEqual(spoken, "Ask [SURNAME_9] about it.")
        self.assertEqual(
            unresolved_placeholders("Ask [SURNAME_9] about it.", mapping),
            [("SURNAME", "9")],
        )

    def test_text_without_placeholders_is_untouched(self) -> None:
        self.assertEqual(
            rehydrate("Nothing private here.", map_from_entities(ENTITIES)),
            "Nothing private here.",
        )

    def test_empty_inputs_are_safe(self) -> None:
        self.assertEqual(rehydrate("", {}), "")
        self.assertEqual(rehydrate("text", {}), "text")
        self.assertEqual(map_from_entities([]), {})
        self.assertEqual(map_from_entities(None), {})

    def test_malformed_entities_are_skipped(self) -> None:
        mapping = map_from_entities(
            [
                {"placeholder": "[A_1]"},
                {"text": "orphan"},
                "not a dict",
                {"placeholder": "", "text": "empty"},
                {"placeholder": "[GOOD_1]", "text": "kept"},
            ]
        )
        self.assertEqual(mapping, {"[GOOD_1]": "kept"})

    def test_the_rolling_map_spans_turns(self) -> None:
        rolling = RehydrationMap()
        rolling.add(ENTITIES)
        rolling.add(
            [
                {
                    "placeholder": "[CITY_1]",
                    "text": "Thuwal",
                    "entity_type": "CITY",
                }
            ]
        )
        # A later reply can still refer to an earlier turn's placeholder.
        self.assertEqual(
            rolling.apply("[GIVENNAME_1] is in [CITY_1]."),
            "Turki is in Thuwal.",
        )

    def test_the_rolling_map_is_bounded(self) -> None:
        rolling = RehydrationMap(limit=3)
        for index in range(10):
            rolling.add(
                [{"placeholder": f"[N_{index}]", "text": str(index)}]
            )
        self.assertEqual(len(rolling), 3)


class SpokenReplyTest(unittest.TestCase):
    """The audio carries real values; surfaces keep the masked text."""

    class Speech:
        def __init__(self) -> None:
            self.spoken: list[str] = []

        def speak(self, text: str) -> None:
            self.spoken.append(text)

    class LLM:
        configured = True

        def __init__(self, reply: str) -> None:
            self.reply = reply
            self.seen: list[str] = []

        def reply_to(self, masked_text: str) -> str:
            self.seen.append(masked_text)
            return self.reply

    def test_the_model_sees_placeholders_and_the_room_hears_names(self) -> None:
        llm = self.LLM("I will call [GIVENNAME_1] on [TELEPHONENUM_1].")
        speech = self.Speech()
        bus = EventBus()
        worker = AssistantWorker(llm=llm, speech=speech, bus=bus)
        worker.start()
        self.addCleanup(worker.shutdown)

        worker.submit("Call [GIVENNAME_1] on [TELEPHONENUM_1]", ENTITIES)
        self.assertTrue(wait_for(lambda: speech.spoken))

        # Nothing private was sent.
        self.assertEqual(llm.seen, ["Call [GIVENNAME_1] on [TELEPHONENUM_1]"])
        self.assertNotIn("Turki", llm.seen[0])
        self.assertNotIn("555", llm.seen[0])

        # The room hears the real values.
        self.assertEqual(
            speech.spoken, ["I will call Turki on 555 123 4567."]
        )

        # The screen is in the room too, and shows what the room hears. The
        # transcript database is the one that keeps the placeholders, because
        # it is the artifact that outlives the process.
        turns = bus.snapshot()["conversation"]["turns"]
        self.assertIn("Turki", turns[1]["text"])
        self.assertNotIn("[GIVENNAME_1]", turns[1]["text"])

    def test_a_reply_with_no_placeholders_is_spoken_unchanged(self) -> None:
        llm = self.LLM("It is sunny today.")
        speech = self.Speech()
        worker = AssistantWorker(llm=llm, speech=speech, bus=EventBus())
        worker.start()
        self.addCleanup(worker.shutdown)
        worker.submit("what is the weather", [])
        self.assertTrue(wait_for(lambda: speech.spoken))
        self.assertEqual(speech.spoken, ["It is sunny today."])


if __name__ == "__main__":
    unittest.main()
