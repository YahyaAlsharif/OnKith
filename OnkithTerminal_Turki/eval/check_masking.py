#!/usr/bin/env python3
"""What the device masks and what it lets through, on this machine, right now.

    .venv/bin/python eval/check_masking.py
    ONKITH_PII_ENGINE=tinybert .venv/bin/python eval/check_masking.py

A hand check before handing the build to somebody, and the fastest way to
answer "is COUNTRY still off?" without reading two config files and a
label map. It runs the real adapter with the real configuration, so what
it prints is what the microphone would get.

The released list is not a list of things the model failed to find. Every
one of them is detected and then deliberately let through, because a city
or a country on its own does not identify anybody and redacting it costs
every question about the time, the weather, or the way somewhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.pii_adapter import PIIMaskerAdapter

#: Sentences that must come back untouched. Each one is a thing somebody
#: actually says to this device, and each contains a category the model
#: does detect and the configuration releases.
MUST_PASS_THROUGH = (
    ("CITY", "what time is it right now in Riyadh"),
    ("COUNTRY", "how hot is it in Saudi Arabia today"),
    ("COUNTRY", "is Japan ahead of us in time"),
    ("CITY + COUNTRY", "what is the weather in London and in France"),
    ("TIMEZONE", "what timezone is Greenwich Mean Time"),
    ("TIME", "remind me at half past four"),
    ("TIME", "the meeting is at 9:30 tomorrow morning"),
    ("the device's own name", "Onkith turn the light off"),
    # TITLE is not released. In this ontology it is the honorific, so it
    # fires on Doctor and Mrs and never on the name of a book or a song -
    # those come back untouched already, with no entity at all.
    ("a book title, unprompted", "I am reading The Great Gatsby"),
    ("a film title, unprompted", "have you seen the film The Godfather"),
)

#: Released, but the words beside them must still go. A category is only
#: safe to release when what sits next to it is still protected.
RELEASED_BUT_NEIGHBOUR_MASKED = (
    ("honorific keeps the title, loses the name",
     "please tell Mrs Nowak I called", "Mrs", "Nowak"),
)

#: One probe per category the device redacts. The label on the left is
#: what the sentence is about, not necessarily what the model calls it -
#: the printed result is the authority.
MUST_BE_MASKED = (
    ("name", "my name is Sarah Walusiak"),
    ("phone", "call me on 055 123 4567"),
    ("email", "write to rayyan.hadi@example.com"),
    ("address", "I live at 42 King Fahd Road"),
    ("postcode", "the postcode is SW1A 1AA"),
    ("bank account", "my account is GB72 NWBK 6016 1331 9268 19"),
    ("card number", "the card is 4532 0151 1283 0366"),
    ("national id", "my national id is 123-45-6789"),
    ("id card", "my id card number is ID-88213394"),
    ("passport", "my passport is X4429183"),
    ("driving licence", "my licence number is D1234567890AB"),
    ("tax number", "my tax identifier is TX-940-761-285"),
    ("date of birth", "I was born on the third of May 1998"),
    ("age", "my daughter is nine years old"),
    ("employer", "I work at Aramco Digital"),
    ("salary", "my salary is 28000 riyals a month"),
    ("bank name", "my bank is Al Rajhi Bank"),
    ("web address", "see https://internal.example.org/reports"),
    ("ip address", "the box is on 192.168.1.44"),
    ("password", "my password is hunter2trombone"),
    ("sex", "the patient is female"),
)


def main() -> int:
    config = PipelineConfig()
    adapter = PIIMaskerAdapter(config)
    adapter.load()

    released = set(config.never_masked_entities)
    label_map = json.loads(
        config.pii_label_map_path.read_text(encoding="utf-8")
    )
    known = label_map.get("entities")
    if not isinstance(known, list):
        known = sorted(
            {
                label.split("-", 1)[1]
                for label in label_map.get("id2label", label_map).values()
                if label != "O"
            }
        )

    print("=" * 72)
    print(f"  engine        {config.pii_engine}")
    print(f"  model         {config.pii_model_path.name}  "
          f"({config.pii_model_path.stat().st_size / 1048576:.0f} MB, "
          f"{config.pii_label_count} labels)")
    print(f"  detects       {len(known)} categories")
    print(f"  released      {', '.join(sorted(released)) or 'none'}")
    print("=" * 72)

    unknown = sorted(released - set(known))
    if unknown:
        # A released category the model cannot emit is a config entry that
        # does nothing. Silent, and exactly the kind of thing that survives
        # a model swap by looking unchanged.
        print()
        print(f"  WARNING: {', '.join(unknown)} is in never_mask but is not a")
        print(f"  category this model emits. That entry has no effect.")

    failures = 0

    print()
    print("RELEASED ON PURPOSE - these must come back unchanged")
    print("-" * 72)
    for label, text in MUST_PASS_THROUGH:
        result = adapter.mask_text(text)
        passed = result.masked_text == text
        failures += not passed
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label:<22} {result.masked_text}")

    print()
    print("RELEASED, BUT THE NAME BESIDE IT IS NOT")
    print("-" * 72)
    for label, text, kept, gone in RELEASED_BUT_NEIGHBOUR_MASKED:
        result = adapter.mask_text(text)
        passed = kept in result.masked_text and gone not in result.masked_text
        failures += not passed
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label}")
        print(f"  {'':<8} {result.masked_text}")

    print()
    print("MASKED - these must not come back unchanged")
    print("-" * 72)
    for label, text in MUST_BE_MASKED:
        result = adapter.mask_text(text)
        found = sorted({entity["entity_type"] for entity in result.entities})
        passed = bool(found) and result.masked_text != text
        failures += not passed
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label:<16} {result.masked_text}")
        print(f"  {'':<8} {'':<16} -> {', '.join(found) or 'NOTHING DETECTED'}")

    print()
    print("-" * 72)
    print(f"  {failures} failure(s)" if failures else "  all checks passed")
    print()
    print("  A category named on the left and a different one on the right is")
    print("  a mislabel, not a leak: the words were still replaced. Only")
    print("  'NOTHING DETECTED' means text was released.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
