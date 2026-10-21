from __future__ import annotations

import unittest

import auto_lrc


class RawAsrDriftGuardTests(unittest.TestCase):
    def test_accumulating_offset_is_not_collapsed_into_systematic_drift(self) -> None:
        timestamps = [10.0, 20.0, 30.0]
        report: dict[str, object] = {
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
                {"entry": 3, "timestamp": 30.0, "score": 0.9},
            ],
            "suspicious_alignments": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 9.2, "score": 0.95, "segment": {"start": 9.2}},
                {"entry": 2, "timestamp": 18.6, "score": 0.95, "segment": {"start": 18.6}},
                {"entry": 3, "timestamp": 28.0, "score": 0.95, "segment": {"start": 28.0}},
            ]
        }

        auto_lrc.flag_unresolved_raw_ctc_disagreements(timestamps, report, raw_report)

        self.assertEqual(report["raw_asr_systematic_drift_runs"], [])
        self.assertEqual(report["review_required_count"], 3)
        self.assertTrue(all(
            "unresolved_raw_ctc_disagreement" in row.get("flags", [])
            for row in report["suspicious_alignments"]  # type: ignore[index]
        ))


if __name__ == "__main__":
    unittest.main()
