from __future__ import annotations

import unittest

import auto_lrc


class SeverityMergeTests(unittest.TestCase):
    @staticmethod
    def base_report(count: int) -> dict[str, object]:
        return {
            "timing_entries": count,
            "assignments": [
                {
                    "entry": i + 1,
                    "timestamp": float((i + 1) * 10),
                    "score": 0.9,
                }
                for i in range(count)
            ],
            "suspicious_alignments": [
                {
                    "entry": 1,
                    "severity": "high",
                    "review_required": True,
                    "flags": ["group_placement_unsupported"],
                    "candidate_timestamps": {"output": 10.0},
                }
            ],
        }

    def test_ctc_disagreement_cannot_downgrade_existing_high(self) -> None:
        report = self.base_report(1)
        ctc_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.5, "ctc_score": 0.2}
            ]
        }
        auto_lrc.flag_unresolved_ctc_disagreements([10.0], report, ctc_report)
        row = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertEqual(row["severity"], "high")
        self.assertTrue(row["review_required"])

    def test_raw_disagreement_cannot_downgrade_existing_high(self) -> None:
        report = self.base_report(1)
        raw_report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.8,
                    "score": 0.70,
                    "segment": 1,
                    "borrowed": False,
                }
            ]
        }
        auto_lrc.flag_unresolved_raw_ctc_disagreements(
            [10.0], report, raw_report
        )
        row = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertEqual(row["severity"], "high")
        self.assertTrue(row["review_required"])

    def test_systematic_drift_diagnostic_cannot_clear_existing_review(self) -> None:
        report = self.base_report(3)
        raw_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 9.0, "score": 0.70, "segment": 1, "borrowed": False},
                {"entry": 2, "timestamp": 19.0, "score": 0.70, "segment": 2, "borrowed": False},
                {"entry": 3, "timestamp": 29.0, "score": 0.70, "segment": 3, "borrowed": False},
            ]
        }
        auto_lrc.flag_unresolved_raw_ctc_disagreements(
            [10.0, 20.0, 30.0], report, raw_report
        )
        row = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertIn("raw_asr_systematic_drift", row["flags"])
        self.assertEqual(row["severity"], "high")
        self.assertTrue(row["review_required"])


if __name__ == "__main__":
    unittest.main()
