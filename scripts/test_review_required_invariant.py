"""review_required single-writer contract (LRC 1.5 W4)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402

REVIEW_KEYS = ("review_required", "review_required_count", "review_required_percent", "verification_counts")


def _row(required: bool, classes: list[str] | None, status: str) -> dict[str, object]:
    row: dict[str, object] = {
        "chosen_time": 1.0,
        "review_required": required,
        "verification_status": status,
        "overall_trusted": status == "trusted",
    }
    if classes is not None:
        row["review_reason_classes"] = classes
    return row


class ReviewRequiredInvariantTests(unittest.TestCase):
    def test_finalize_writes_four_keys_from_rows(self) -> None:
        report = {
            "timing_entries": 4,
            "assignments": [
                _row(True, ["actionable-timing-failure"], "review-required"),
                _row(False, [], "trusted"),
                _row(False, [], "unverified"),
                _row(True, None, "review-required"),
            ],
        }
        auto_lrc.finalize_review_summary(report)
        self.assertEqual(report["review_required_count"], 2)
        self.assertEqual(report["review_required_percent"], auto_lrc.ratio_percent(2, 4))
        self.assertTrue(report["review_required"])
        self.assertEqual(report["verification_counts"], {"trusted": 1, "unverified": 1, "review_required": 2})

    def test_tampered_flag_raises(self) -> None:
        report = {"assignments": [_row(True, ["final-self-check"], "review-required")]}
        auto_lrc.finalize_review_summary(report)
        report["assignments"][0]["review_required"] = False
        with self.assertRaises(auto_lrc.LrcError):
            auto_lrc.finalize_review_summary(report)
        report["assignments"][0]["review_required"] = True
        report["assignments"][0]["review_reason_classes"] = []
        with self.assertRaises(auto_lrc.LrcError):
            auto_lrc.finalize_review_summary(report)

    def test_missing_assignments_raises(self) -> None:
        with self.assertRaises(auto_lrc.LrcError):
            auto_lrc.finalize_review_summary({})

    def test_update_report_confidence_metrics_uses_the_single_writer(self) -> None:
        report = {
            "timing_entries": 2,
            "assignments": [
                _row(False, None, "trusted"),
                _row(True, None, "review-required"),
            ],
        }
        auto_lrc.update_report_confidence_metrics(report)
        expected = {key: report[key] for key in REVIEW_KEYS}
        auto_lrc.finalize_review_summary(report)
        self.assertEqual({key: report[key] for key in REVIEW_KEYS}, expected)
        self.assertEqual(report["verification_counts"]["review_required"], report["review_required_count"])
        self.assertEqual(
            report["verification_counts"]["trusted"] + report["verification_counts"]["unverified"]
            + report["verification_counts"]["review_required"],
            len(report["assignments"]),
        )
        auto_lrc.assert_report_review_contract(report)


if __name__ == "__main__":
    unittest.main()
