#!/usr/bin/env python3
"""Run one pipeline's downstream privacy path over fixed text and print JSON.

Point it at ``FullPipeline_V6``, ``archive/FullPipeline_V4`` or
``archive/FullPipeline_V3`` and it does the same thing on any of them: transcript
merge, wake-word repair, inactivity buffer, DeBERTa-v3-xsmall INT8 masking, span
consolidation, spoken-digit rendering. The output is a JSON document, so two
roots can be diffed.

    python model/privacy_evaluation/scripts/masking_parity_probe.py FullPipeline_V6

Why a separate process
----------------------
Every pipeline package is called ``integrated_pipeline``. Importing two of them
into one interpreter would make the second import a no-op and quietly compare a
pipeline against itself.

What it deliberately does not do
--------------------------------
It never loads a speech recognizer. Nothing here imports ``whisper_adapter`` or
``canary_adapter``, so running it against the archived V3 root cannot start
Canary inference. The point of the comparison is that replacing the recognizer
did not change the masking implementation, and that question is answered by
feeding both sides the same text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Fixed probe text. Two shapes per case where it matters: the unpunctuated
#: lowercase surface Canary produced under ``pnc: no``, and the punctuated,
#: capitalised surface Whisper produces. Both go through both pipelines, so a
#: difference in the *masker* shows up separately from a difference in what the
#: recognizer would have handed it.
PROBE_TEXTS = [
    "my name is sarah walusiak and my number is 555 0147",
    "My name is Sarah Walusiak and my number is 555-0147.",
    "email me at sarah.walusiak@example.org before friday",
    "Email me at sarah.walusiak@example.org before Friday.",
    "i live at 42 fairview road in leeds ls1 4dt",
    "I live at 42 Fairview Road in Leeds, LS1 4DT.",
    "my account number is one two three four five six seven eight",
    "My account number is 12345678.",
    "my card is four one one one one one one one one one one one one one one one",
    "My card is 4111 1111 1111 1111.",
    "onkith play bones by imagine dragons",
    "Onkith, play Bones by Imagine Dragons.",
    "on kids what is the weather",
    "the appointment is on the twelfth of march at nine thirty",
    "The appointment is on 12 March at 09:30.",
    "he is forty seven years old and his id is 1234567890",
    "He is 47 years old and his ID is 1234567890.",
    # Hard negatives: nothing here is private, and nothing may be masked.
    "what time is it",
    "turn the music off please",
    "Someone named Ankit called this morning.",
]


def probe(root: Path) -> dict[str, object]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.digit_normalizer import normalize_spoken_digits
    from app.wake_word_normalizer import normalize_wake_word
    from integrated_pipeline.config import PipelineConfig
    from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
    from integrated_pipeline.pii_adapter import PIIMaskerAdapter
    from integrated_pipeline.schemas import TranscriptFragment
    from integrated_pipeline.transcript_merger import merge_segment_transcripts

    config = PipelineConfig(root=root)
    adapter = PIIMaskerAdapter(config)
    adapter.load()

    rows = []
    for index, text in enumerate(PROBE_TEXTS):
        merged = merge_segment_transcripts([text])
        wake_text, variant = normalize_wake_word(merged.text)
        buffer = InactivityTranscriptBuffer(
            inactivity_seconds=config.buffer_inactivity_seconds,
            maximum_tokens=config.maximum_buffer_tokens,
            maximum_duration_seconds=config.maximum_buffer_duration_seconds,
        )
        buffer.add_fragment(
            TranscriptFragment(
                fragment_id=f"probe-{index}",
                text=wake_text,
                arrival_time_seconds=0.0,
                source_audio_id=f"probe-{index}",
                session_id="masking-parity",
                batch_id=f"probe-{index}",
                sequence_number=0,
            )
        )
        flushed = buffer.flush(0.0, reason="manual")
        privacy_input = flushed.text if flushed else wake_text
        result = adapter.mask_text(privacy_input)
        rows.append(
            {
                "input": text,
                "wake_variant": variant,
                "privacy_input": privacy_input,
                "masked_text": result.masked_text,
                "final_output": normalize_spoken_digits(result.masked_text),
                "entities": sorted(
                    (
                        int(entity["start"]),
                        int(entity["end"]),
                        str(entity["entity_type"]),
                        str(entity.get("source", "")),
                    )
                    for entity in result.entities
                ),
            }
        )

    return {
        "root": root.name,
        "privacy_artifact_sha256": str(config.pii["sha256"]),
        "label_count": int(config.pii["label_count"]),
        "word_policy": config.pii_word_policy,
        "structured_enabled": config.enable_structured_pii,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        type=Path,
        help="pipeline package root, e.g. FullPipeline_V6 or archive/FullPipeline_V3",
    )
    args = parser.parse_args(argv)
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root)
    print(json.dumps(probe(root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
