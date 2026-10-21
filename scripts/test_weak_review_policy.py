"""LRC 1.7: WEAK reason codes are demoted from REVIEW to UNTRUSTED at output time only."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402

WEAK = "candidate_disagreement"
STRONG = "temporal-incoherence"


def _row(entry: int, codes: list[str], *, review: bool) -> dict[str, object]:
    return {
        "entry": entry,
        "timestamp": float(entry * 10),
        "review_required": review,
        "review_reason_classes": ["actionable-suspicious-conflict"] if review else [],
        "overall_trusted": False,
        "timing_audit_trusted": False,
        "content_trusted": True,
        "verification_status": "review-required" if review else "unverified",
        "final_trust_reason_codes": list(codes),
    }


def _report() -> dict[str, object]:
    rows = [
        _row(1, [WEAK], review=True),
        _row(2, [WEAK, STRONG], review=True),
        _row(3, [WEAK, "insufficient-minimum-proof"], review=True),
        _row(4, [], review=False),
        _row(5, [WEAK, "no-strict-opening-proof"], review=True),
    ]
    rows[3]["verification_status"] = "trusted"
    rows[3]["overall_trusted"] = True
    report: dict[str, object] = {
        "timing_entries": 5,
        "assignments": rows,
        "suspicious_alignments": [
            {"entry": 1, "flags": [WEAK], "severity": "medium", "review_required": True},
            {"entry": 2, "flags": [WEAK], "severity": "medium", "review_required": True},
        ],
        "timing_review_entries": [{"entry": 1, "flags": []}, {"entry": 2, "flags": []}, {"entry": 3, "flags": []}],
        "unverified_entries": [],
        "low_confidence_entries": [
            {"entry": 1, "verification_status": "review-required", "trust_failures": ["timing-untrusted", "actionable-review"]},
        ],
        "unverified_count": 0,
    }
    auto_lrc.finalize_review_summary(report)
    return report


class WeakReviewPolicyTests(unittest.TestCase):
    def test_weak_codes_are_classified_review_inside_the_pipeline(self) -> None:
        # The pipeline itself is untouched: weak codes still read as REVIEW there.
        for code in auto_lrc.WEAK_REVIEW_CODES:
            self.assertEqual(auto_lrc.FINAL_LINE_TRUST_CLASSIFICATION.get(code), "REVIEW", code)

    def test_only_weak_only_rows_are_demoted(self) -> None:
        report = _report()
        self.assertEqual(report["review_required_count"], 4)
        demoted = auto_lrc.apply_weak_review_policy(report)
        self.assertEqual(demoted, [1, 3])
        rows = report["assignments"]
        self.assertEqual(rows[0]["verification_status"], "unverified")
        self.assertFalse(rows[0]["review_required"])
        self.assertEqual(rows[0]["review_reason_classes"], [])
        self.assertTrue(rows[0]["weak_review_demoted"])
        self.assertEqual(rows[1]["verification_status"], "review-required")
        self.assertEqual(rows[2]["verification_status"], "unverified")
        self.assertEqual(rows[3]["verification_status"], "trusted")
        # a non-weak companion code (outside the chosen cut) keeps the row in REVIEW
        self.assertEqual(rows[4]["verification_status"], "review-required")
        self.assertEqual(report["review_required_count"], 2)
        self.assertEqual(report["verification_counts"], {"trusted": 1, "unverified": 2, "review_required": 2})
        self.assertEqual(report["unverified_count"], 2)
        self.assertFalse(report["suspicious_alignments"][0]["review_required"])
        self.assertTrue(report["suspicious_alignments"][1]["review_required"])
        self.assertEqual([row["entry"] for row in report["timing_review_entries"]], [2])
        self.assertEqual(report["low_confidence_entries"][0]["trust_failures"], ["timing-untrusted"])
        self.assertEqual([row["entry"] for row in report["unverified_entries"]], [1, 3])
        self.assertEqual(report[auto_lrc.WEAK_REVIEW_POLICY_KEY]["demoted_entries"], [1, 3])

    def test_policy_is_idempotent_and_never_trusts(self) -> None:
        report = _report()
        auto_lrc.apply_weak_review_policy(report)
        snapshot = copy.deepcopy(report)
        self.assertEqual(auto_lrc.apply_weak_review_policy(report), [])
        self.assertEqual(report, snapshot)
        self.assertFalse(any(row.get("weak_review_demoted") and row["verification_status"] == "trusted"
                             for row in report["assignments"]))

    def test_write_alignment_report_applies_policy(self) -> None:
        report = _report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.align-report.json"
            auto_lrc.write_alignment_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["review_required_count"], 2)
        self.assertEqual(saved["assignments"][0]["verification_status"], "unverified")
        self.assertEqual(saved[auto_lrc.WEAK_REVIEW_POLICY_KEY]["demoted_entries"], [1, 3])


if __name__ == "__main__":
    unittest.main()
