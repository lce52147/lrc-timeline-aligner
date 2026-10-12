"""P3 real-call regression: Central summary must not overwrite committed review rows."""
from __future__ import annotations

import unittest
from unittest import mock

import auto_lrc
import test_core_logic


class PostCommitSummaryContractTests(unittest.TestCase):
    def test_selected_central_summary_cannot_override_final_row_review_contract(self):
        fixture = test_core_logic.CentralTimingStateTests()
        candidate = fixture.candidate(12.0, fixture.coherent_spans(12.0), current=True)
        report = fixture.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        original = auto_lrc.apply_central_summary_to_report

        def stale_preview_summary(report, summary):
            # Production P3 case: 24 rows, 9 REVIEW, stale summary 10 REVIEW.
            original(report, summary)
            report["review_required_count"] = 1
            report["review_required_percent"] = 100.0
            report["review_required"] = True

        with mock.patch.object(
            auto_lrc, "apply_central_summary_to_report",
            side_effect=stale_preview_summary,
        ):
            auto_lrc.commit_evaluated_backend(selection, report)

        expected = sum(bool(row.get("review_required")) for row in report["assignments"])
        self.assertEqual(report["review_required_count"], expected)
        self.assertEqual(
            report["review_required_percent"],
            auto_lrc.ratio_percent(expected, len(report["assignments"])),
        )
        auto_lrc.assert_report_review_contract(report)


if __name__ == "__main__":
    unittest.main()