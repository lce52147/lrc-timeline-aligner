"""F4-ctc-final-disagreement no longer forces REVIEW on its own (LRC 1.5 W6)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402

F4 = "F4-ctc-final-disagreement"


def _assignment(*, trusted_timing: bool, self_check: list[str], flags: list[str] | None = None) -> dict[str, object]:
    item: dict[str, object] = {
        "chosen_time": 12.34,
        "segment": {"start": 12.34, "end": 14.0},
        "flags": list(flags or []),
        "self_check_flags": list(self_check),
    }
    if trusted_timing:
        item["manual_anchor_hint"] = True
    else:
        item["group_placement_unsupported"] = True
    return item


class F4DemotionTests(unittest.TestCase):
    def test_f4_is_untrusted_class(self) -> None:
        self.assertEqual(auto_lrc.FINAL_LINE_TRUST_CLASSIFICATION[F4], "UNTRUSTED")
        self.assertNotIn(F4, auto_lrc.ACTIONABLE_TIMING_FAILURES)
        self.assertNotIn(F4, auto_lrc.ACTIONABLE_SUSPICIOUS_FLAGS)
        fixture = json.loads((SCRIPTS / "test_fixtures" / "reason_codes_b1_39.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["classification"][F4], "UNTRUSTED")
        for code, cls in fixture["classification"].items():
            self.assertEqual(cls, auto_lrc.FINAL_LINE_TRUST_CLASSIFICATION.get(code, "INFO"), code)

    def _finalize(self, item: dict[str, object]) -> dict[str, object]:
        report = {"timing_entries": 1, "assignments": [item]}
        auto_lrc.update_report_confidence_metrics(report)
        return report["assignments"][0]

    def test_f4_only_untrusted_timing_is_untrusted_not_review(self) -> None:
        row = self._finalize(_assignment(trusted_timing=False, self_check=[F4]))
        decision = auto_lrc.final_line_trust(row, [], central_committed=False)
        self.assertFalse(decision.review_required)
        self.assertFalse(decision.timing_trusted)
        self.assertIn(F4, decision.reason_codes)
        self.assertFalse(row["review_required"])
        self.assertEqual(row["verification_status"], "unverified")
        self.assertIn(F4, row["final_trust_reason_codes"])
        self.assertNotIn(
            "final-self-check",
            auto_lrc._review_reason_classes_from_trust(decision, row, []),
        )

    def test_f4_only_trusted_timing_becomes_trusted(self) -> None:
        item = _assignment(trusted_timing=True, self_check=[F4])
        self.assertTrue(auto_lrc.assignment_timing_is_trusted(item))
        row = self._finalize(item)
        self.assertFalse(row["review_required"])
        self.assertTrue(row["overall_trusted"] or row["verification_status"] == "trusted" or row["timing_audit_trusted"])
        self.assertIn(F4, row["final_trust_reason_codes"])

    def test_f4_with_review_class_code_stays_review(self) -> None:
        review_code = next(iter(sorted(auto_lrc.ACTIONABLE_TIMING_FAILURES)))
        row = self._finalize(_assignment(trusted_timing=False, self_check=[F4], flags=[review_code]))
        self.assertTrue(row["review_required"])
        self.assertEqual(row["verification_status"], "review-required")
        self_check_review = next(
            code for code in ("F1-vocal-absent", "F3-nonmonotonic")
            if auto_lrc.FINAL_LINE_TRUST_CLASSIFICATION.get(code) == "REVIEW"
        )
        row2 = self._finalize(_assignment(trusted_timing=False, self_check=[F4, self_check_review]))
        self.assertTrue(row2["review_required"])
        decision = auto_lrc.final_line_trust(row2, [], central_committed=False)
        self.assertIn("final-self-check", auto_lrc._review_reason_classes_from_trust(decision, row2, []))

    def test_self_check_helper(self) -> None:
        self.assertFalse(auto_lrc.self_check_flags_require_review([F4]))
        self.assertFalse(auto_lrc.self_check_flags_require_review([]))
        self.assertFalse(auto_lrc.self_check_flags_require_review(None))
        self.assertTrue(auto_lrc.self_check_flags_require_review([F4, "F3-nonmonotonic"]))


if __name__ == "__main__":
    unittest.main()
