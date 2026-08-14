"""PASSWORD admission: a disclosed secret is PII, talking about passwords is not."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import password_rules
from password_rules import (
    CATEGORY_DISCLOSURE,
    CATEGORY_NO_CUE,
    CATEGORY_NOT_SECRET,
    CATEGORY_POLICY,
    CATEGORY_STRUCTURED,
    classify_password_span,
    filter_password_spans,
    looks_like_secret,
)


def classify(text, value):
    start = text.index(value)
    return classify_password_span(text, start, start + len(value))


class DisclosureTests(unittest.TestCase):
    """The examples that define why this label exists."""

    ACCEPTED = [
        ("My password is BlueHorse!92", "BlueHorse!92"),
        ("The Wi-Fi password is kaust2026", "kaust2026"),
        ("My phone password is 482913", "482913"),
        ("I changed my password to MoonLight44", "MoonLight44"),
        ("Use Summer#2026 to log in", "Summer#2026"),
    ]

    def test_specified_examples_are_accepted(self):
        for text, value in self.ACCEPTED:
            with self.subTest(text=text):
                decision = classify(text, value)
                self.assertTrue(decision.accept, decision)
                self.assertEqual(decision.category, CATEGORY_DISCLOSURE)
                self.assertIsNotNone(decision.cue)

    def test_conversational_credential_contexts(self):
        for text, value in [
            ("the vpn password is Tuwaiq!4417", "Tuwaiq!4417"),
            ("my router password is admin2024x", "admin2024x"),
            ("set the account password to Falcon_88", "Falcon_88"),
            ("log in with Zahra#9021", "Zahra#9021"),
            ("the guest network password is riyadh2026wifi", "riyadh2026wifi"),
            ("my passcode is 7741", "7741"),
            ("i think the password is Cedar%Lantern5 but let me check",
             "Cedar%Lantern5"),
        ]:
            with self.subTest(text=text):
                self.assertTrue(classify(text, value).accept)

    def test_the_span_covers_the_secret_not_the_word_password(self):
        text = "My password is BlueHorse!92"
        secret = classify(text, "BlueHorse!92")
        self.assertTrue(secret.accept)
        word = classify_password_span(text, 3, 11)  # the word "password"
        self.assertFalse(word.accept)


class HardNegativeTests(unittest.TestCase):
    """These sentences contain no secret and belong in the hard negatives."""

    REJECTED = [
        ("I forgot my password", "password"),
        ("How do I reset my password?", "password"),
        ("Never share your password", "password"),
        ("The password must contain eight characters", "eight characters"),
        ("Password authentication is enabled", "Password"),
    ]

    def test_specified_examples_are_rejected(self):
        for text, value in self.REJECTED:
            with self.subTest(text=text):
                decision = classify(text, value)
                self.assertFalse(decision.accept, decision)

    def test_policy_and_help_talk_is_rejected(self):
        for text, value in [
            ("password strength is weak", "password"),
            ("my password manager keeps everything in sync", "password"),
            ("the password expired so i need a new one", "password"),
            ("passwords should never be reused across sites", "passwords"),
            ("the password field is greyed out", "password"),
        ]:
            with self.subTest(text=text):
                decision = classify(text, value)
                self.assertFalse(decision.accept)
                self.assertEqual(decision.category, CATEGORY_POLICY)

    def test_a_policy_sentence_that_still_discloses_keeps_the_span(self):
        # Rejecting this would throw away a real credential.
        text = "i changed my password to MoonLight44 because it must contain a digit"
        self.assertTrue(classify(text, "MoonLight44").accept)


class DocumentDumpTests(unittest.TestCase):
    """Synthetic credential columns do not resemble OnKith speech input."""

    def test_table_and_markup_rows_are_rejected(self):
        for text, value in [
            ("name | email | password\nAna | a@b.co | Secret!99", "Secret!99"),
            ("user\tpass\tstatus\nana\tSecret!99\tactive", "Secret!99"),
            ("<input name='password' value='Secret!99'>", "Secret!99"),
        ]:
            with self.subTest(text=text[:30]):
                decision = classify(text, value)
                self.assertFalse(decision.accept)
                self.assertEqual(decision.category, CATEGORY_STRUCTURED)

    def test_multi_field_records_are_rejected(self):
        text = "username: ana, password: Secret!99, role: admin, status: active"
        decision = classify(text, "Secret!99")
        self.assertFalse(decision.accept)
        self.assertEqual(decision.category, CATEGORY_STRUCTURED)


class SecretShapeTests(unittest.TestCase):
    def test_plausible_credentials(self):
        for value in ("BlueHorse!92", "kaust2026", "482913", "MoonLight44",
                      "Summer#2026", "north-star-2049", "gr33nOasis", "7741"):
            self.assertTrue(looks_like_secret(value), value)

    def test_implausible_credentials(self):
        for value in ("password", "PASSWORD", "none", "redacted", "eight characters",
                      "n/a", "x", "yes", "twelve"):
            self.assertFalse(looks_like_secret(value), value)

    def test_numeric_pins_have_a_sane_length_range(self):
        self.assertFalse(looks_like_secret("12"))
        self.assertTrue(looks_like_secret("4829"))
        self.assertTrue(looks_like_secret("482913"))
        self.assertFalse(looks_like_secret("1" * 20))

    def test_a_secret_without_a_cue_is_not_admitted(self):
        decision = classify("the reference is BlueHorse!92", "BlueHorse!92")
        self.assertFalse(decision.accept)
        self.assertEqual(decision.category, CATEGORY_NO_CUE)

    def test_a_cue_without_a_secret_is_not_admitted(self):
        decision = classify("my password is password", "password")
        self.assertFalse(decision.accept)
        self.assertEqual(decision.category, CATEGORY_NOT_SECRET)


class FilteringTests(unittest.TestCase):
    def test_rejected_password_spans_become_o_and_other_spans_survive(self):
        record = {
            "uid": "x",
            "source_text": "Ana asked how to reset her password on ana@example.com",
            "privacy_mask": [
                {"label": "GIVENNAME", "start": 0, "end": 3, "value": "Ana"},
                {"label": "PASSWORD", "start": 28, "end": 36, "value": "password"},
                {"label": "EMAIL", "start": 39, "end": 54, "value": "ana@example.com"},
            ],
        }
        filtered, decisions = filter_password_spans(record)
        labels = [span["label"] for span in filtered["privacy_mask"]]
        self.assertEqual(labels, ["GIVENNAME", "EMAIL"])
        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0].accept)

    def test_accepted_password_spans_are_kept(self):
        record = {
            "uid": "x",
            "source_text": "my password is BlueHorse!92",
            "privacy_mask": [{"label": "PASSWORD", "start": 15, "end": 27, "value": "BlueHorse!92"}],
        }
        filtered, decisions = filter_password_spans(record)
        self.assertEqual(len(filtered["privacy_mask"]), 1)
        self.assertTrue(decisions[0].accept)

    def test_audit_summarises_decisions(self):
        decisions = [
            classify("my password is BlueHorse!92", "BlueHorse!92"),
            classify("I forgot my password", "password"),
            classify("Never share your password", "password"),
        ]
        audit = password_rules.audit_password_decisions(decisions)
        self.assertEqual(audit["candidate_spans"], 3)
        self.assertEqual(audit["accepted_spans"], 1)
        self.assertEqual(audit["rejected_spans"], 2)
        self.assertIn(CATEGORY_POLICY, audit["by_category"])


class GeneratedDisclosureTests(unittest.TestCase):
    """Generated PASSWORD training rows must clear the same bar as external data."""

    def test_generated_rows_are_span_exact_and_self_consistent(self):
        import augment_pipeline

        resources = augment_pipeline.load_resources()
        rows = augment_pipeline.generate_password_disclosures(
            resources["entity_values"], count=120, seed=7
        )
        self.assertGreater(len(rows), 50)
        for row in rows:
            text = row["source_text"]
            password_spans = [s for s in row["privacy_mask"] if s["label"] == "PASSWORD"]
            self.assertEqual(len(password_spans), 1, row)
            for span in row["privacy_mask"]:
                self.assertEqual(text[span["start"]:span["end"]], span["value"])
            span = password_spans[0]
            self.assertTrue(
                classify_password_span(text, span["start"], span["end"]).accept, text
            )

    def test_generation_is_deterministic(self):
        import augment_pipeline

        resources = augment_pipeline.load_resources()
        first = augment_pipeline.generate_password_disclosures(
            resources["entity_values"], count=40, seed=11
        )
        second = augment_pipeline.generate_password_disclosures(
            resources["entity_values"], count=40, seed=11
        )
        self.assertEqual(
            [row["source_text"] for row in first], [row["source_text"] for row in second]
        )

    def test_password_hard_negatives_exist_in_the_curated_bank(self):
        import json

        bank = json.loads(
            (Path(__file__).resolve().parents[1] / "resources" / "hard_negative_templates.json")
            .read_text(encoding="utf-8")
        )
        password_talk = [row for row in bank if row["category"] == "password_talk"]
        self.assertGreaterEqual(len(password_talk), 10)
        for row in password_talk:
            self.assertIn("password", row["text"].casefold())


if __name__ == "__main__":
    unittest.main()
