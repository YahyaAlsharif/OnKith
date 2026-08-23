"""The notebook parity comparison must detect every kind of divergence.

The comparison logic is tested directly with synthetic reference data so it is
proven runnable here, independently of whether faster-whisper or real
speech are available.
"""

from __future__ import annotations

import unittest

from scripts.compare_notebook_parity import FAIL, NOT_RUN, PASS, Report, compare


def segment(index, start, unique, end, forced, reason, text):
    return {
        "segment_index": index,
        "segment_start_sample": start,
        "segment_unique_start_sample": unique,
        "segment_end_sample": end,
        "forced_boundary_sample": forced,
        "endpoint_reason": reason,
        "is_final_segment_of_utterance": forced is None,
        "text": text,
    }


def full_run(**overrides):
    base = {
        "silero_onnx_sha256": "abc",
        "segments": [
            segment(0, 12_000, 12_000, 153_824, 148_192, "max_duration", "hello there"),
            segment(1, 142_560, 148_192, 175_328, None, "endpoint_silence", "there world"),
        ],
        "merged_text": "hello there world",
        "blank_segment_indices": [],
        "call_order": ["seg-0", "seg-1"],
    }
    base.update(overrides)
    return base


def statuses(report: Report) -> dict[str, str]:
    return {str(row["check"]): str(row["status"]) for row in report.rows}


class NotebookParityComparisonTests(unittest.TestCase):
    def test_identical_runs_pass_every_dimension(self) -> None:
        report = Report()
        compare(full_run(), full_run(), report)
        result = statuses(report)
        self.assertEqual(result["Segment boundary parity"], PASS)
        self.assertEqual(result["Per-segment transcript parity"], PASS)
        self.assertEqual(result["Final merged transcript parity"], PASS)
        self.assertEqual(result["Blank segment parity"], PASS)
        self.assertEqual(result["STT call ordering parity"], PASS)
        self.assertEqual(result["Endpoint reason parity"], PASS)
        self.assertEqual(result["Same Silero model on both sides"], PASS)
        self.assertFalse(report.failed)

    def test_boundary_difference_is_detected(self) -> None:
        reference = full_run()
        reference["segments"][0]["segment_start_sample"] = 5_856  # calibration leak
        report = Report()
        differences = compare(full_run(), reference, report)
        self.assertEqual(statuses(report)["Segment boundary parity"], FAIL)
        self.assertEqual(len(differences["segment_boundaries"]), 1)

    def test_segment_count_difference_is_detected(self) -> None:
        reference = full_run()
        reference["segments"] = reference["segments"][:1]
        reference["call_order"] = ["seg-0"]
        report = Report()
        compare(full_run(), reference, report)
        result = statuses(report)
        self.assertEqual(result["Segment boundary parity"], FAIL)
        self.assertEqual(result["STT call ordering parity"], FAIL)

    def test_per_segment_transcript_difference_is_detected(self) -> None:
        reference = full_run()
        reference["segments"][1]["text"] = "there word"
        report = Report()
        differences = compare(full_run(), reference, report)
        self.assertEqual(statuses(report)["Per-segment transcript parity"], FAIL)
        self.assertEqual(differences["segment_transcripts"][0]["index"], 1)

    def test_merged_transcript_difference_is_detected(self) -> None:
        report = Report()
        compare(full_run(), full_run(merged_text="hello there word"), report)
        self.assertEqual(statuses(report)["Final merged transcript parity"], FAIL)

    def test_blank_segment_difference_is_detected(self) -> None:
        report = Report()
        compare(full_run(), full_run(blank_segment_indices=[0]), report)
        self.assertEqual(statuses(report)["Blank segment parity"], FAIL)

    def test_endpoint_reason_difference_is_detected(self) -> None:
        reference = full_run()
        reference["segments"][1]["endpoint_reason"] = "end_of_stream"
        report = Report()
        compare(full_run(), reference, report)
        self.assertEqual(statuses(report)["Endpoint reason parity"], FAIL)

    def test_different_silero_model_blocks_a_parity_claim(self) -> None:
        report = Report()
        compare(full_run(), full_run(silero_onnx_sha256="def"), report)
        status = statuses(report)["Same Silero model on both sides"]
        self.assertEqual(status, FAIL)
        self.assertTrue(report.failed)

    def test_missing_local_transcripts_report_not_run_not_pass(self) -> None:
        local = full_run()
        for row in local["segments"]:
            row["text"] = None
        local["merged_text"] = None
        local["blank_segment_indices"] = None
        local["call_order"] = None
        report = Report()
        compare(local, full_run(), report)
        result = statuses(report)
        # Boundaries are still comparable without the recognizer.
        self.assertEqual(result["Segment boundary parity"], PASS)
        # Everything needing transcripts is NOT RUN, never PASS.
        self.assertEqual(result["Per-segment transcript parity"], NOT_RUN)
        self.assertEqual(result["Final merged transcript parity"], NOT_RUN)
        self.assertEqual(result["Blank segment parity"], NOT_RUN)
        self.assertEqual(result["STT call ordering parity"], NOT_RUN)
        self.assertFalse(report.failed)

    def test_empty_reference_reports_not_run(self) -> None:
        report = Report()
        compare(full_run(), {"segments": []}, report)
        result = statuses(report)
        self.assertEqual(result["Segment boundary parity"], NOT_RUN)
        self.assertEqual(result["Endpoint reason parity"], NOT_RUN)
        self.assertFalse(report.failed)

    def test_report_rejects_invented_statuses(self) -> None:
        report = Report()
        with self.assertRaises(ValueError):
            report.add("made up", "PROBABLY FINE")


if __name__ == "__main__":
    unittest.main()
