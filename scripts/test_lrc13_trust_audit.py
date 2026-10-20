"""Production trust consistency regression tests (R4-1/R4-3)."""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import unittest
import auto_lrc


def verified_report() -> dict[str, object]:
    return {
        "backend": "ctc",
        "timing_entries": 1,
        "timing_state_schema_version": 3,
        "algorithm_source_sha256": hashlib.sha256(Path(auto_lrc.__file__).read_bytes()).hexdigest(),
        "candidate_selection": {
            "status": "selected", "metric_stage": "central-evaluated",
            "selected_backend": "ctc",
        },
        "central_timing_state": [{
            "entry": 1, "timestamp": 10.0, "canonical_written_centiseconds": 1000,
            "candidate_id": "ctc-current", "candidate_source": "ctc-current",
            "timestamp_status": "selected_valid", "timing_trusted": True,
        }],
        "assignments": [{
            "entry": 1, "timestamp": 10.0, "score": 0.9, "timing_repair": "ctc",
            "canonical_written_centiseconds": 1000,
            "timestamp_status": "selected_valid",
            "timestamp_revision": "rev1", "timing_evidence_revision": "rev1",
            "selected_candidate_id": "ctc-current", "provisional_candidate_id": None,
            "candidate_source": "ctc-current",
            "candidate_provenance": {
                "candidate_id": "ctc-current", "source": "ctc-current",
                "canonical_written_centiseconds": 1000,
            },
            "normalized_timing_evidence": {
                "computed_for_centiseconds": 1000,
                "structural_failures": [], "minimum_proof_satisfied": True,
            },
            "content_trusted": True, "timing_trusted": True,
            "timing_audit_trusted": True, "overall_trusted": True,
            "review_required": False, "verification_status": "trusted",
        }],
        "suspicious_alignments": [],
    }


class StaleResolutionRegressionTests(unittest.TestCase):
    def test_final_actionable_defect_invalidates_older_resolution(self) -> None:
        report = verified_report()
        self.assertTrue(auto_lrc.report_has_verified_central_commit(report))
        report["assignments"][0]["flags"] = ["previous-tail-overlap"]
        report["suspicious_alignments"] = [{
            "entry": 1, "flags": ["candidate_disagreement"],
            "severity": "resolved",
            "resolution": "superseded-by-verified-central-decision",
            "review_required": False,
        }]
        auto_lrc.update_report_confidence_metrics(report)
        row = report["suspicious_alignments"][0]
        self.assertNotIn("resolution", row)
        self.assertNotEqual(row["severity"], "resolved")
        self.assertTrue(row["review_required"])
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertEqual(report["review_required_count"], 1)

    def test_uncontested_resolution_is_not_erased(self) -> None:
        report = verified_report()
        report["suspicious_alignments"] = [{
            "entry": 1, "flags": ["candidate_disagreement"],
            "severity": "resolved",
            "resolution": "superseded-by-verified-central-decision",
            "review_required": False,
        }]
        auto_lrc.update_report_confidence_metrics(report)
        self.assertFalse(report["suspicious_alignments"][0]["review_required"])
        self.assertEqual(report["review_required_count"], 0)



class UnifiedFinalTrustRegressionTests(unittest.TestCase):
    def assert_bound(self, report: dict[str, object], *, trusted: bool, review: bool) -> None:
        a = report["assignments"][0]
        c = report["central_timing_state"][0]
        self.assertEqual(a["timing_trusted"], trusted)
        self.assertEqual(a["timing_audit_trusted"], trusted)
        self.assertEqual(c["timing_trusted"], trusted)
        self.assertEqual(a["review_required"], review)
        self.assertEqual(a["verification_status"], "review-required" if review else ("trusted" if trusted else "unverified"))
        self.assertEqual(a["overall_trusted"], trusted and not review)
        self.assertEqual(a["timestamp_status"], c["timestamp_status"])
        self.assertEqual(a["timestamp_status"], "selected_valid" if trusted else "provisional_unresolved")
        self.assertEqual(a["timestamp"], 10.0)
        self.assertEqual(c["timestamp"], 10.0)
        self.assertTrue(auto_lrc.report_has_verified_central_commit(report))
        self.assertEqual(report["review_required_count"], int(review))
        self.assertEqual(report["trusted_percent"], 100.0 if trusted and not review else 0.0)

    def test_evidence_downgrade_updates_central_commit_not_just_assignment(self) -> None:
        report = verified_report()
        report["assignments"][0]["normalized_timing_evidence"]["minimum_proof_satisfied"] = False
        report["assignments"][0]["normalized_timing_evidence"]["structural_failures"] = ["previous-tail-ownership-unknown"]
        self.assertTrue(auto_lrc.report_has_verified_central_commit(report))
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=False, review=False)
        self.assertEqual(report["assignments"][0]["provisional_candidate_id"], "ctc-current")

    def test_actionable_review_is_untrusted_and_consistent(self) -> None:
        report = verified_report()
        report["assignments"][0]["flags"] = ["previous-tail-overlap"]
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=False, review=True)

    def test_info_diagnostic_does_not_change_trust(self) -> None:
        report = verified_report()
        report["assignments"][0]["flags"] = ["manual_anchor_resolved"]
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=True, review=False)

    def test_resolved_precentral_flag_on_assignment_is_not_active_review(self) -> None:
        report = verified_report()
        report["assignments"][0]["flags"] = ["candidate_disagreement"]
        report["suspicious_alignments"] = [{
            "entry": 1, "flags": ["candidate_disagreement"],
            "severity": "resolved",
            "resolution": "superseded-by-verified-central-decision",
            "review_required": False,
        }]
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=True, review=False)

    def test_final_self_check_is_review_even_without_preexisting_boolean(self) -> None:
        report = verified_report()
        report["assignments"][0]["self_check_flags"] = ["F3-nonmonotonic"]
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=False, review=True)

    def test_f4_self_check_alone_no_longer_forces_review(self) -> None:
        # LRC 1.5 W6: F4-ctc-final-disagreement is UNTRUSTED-class, so a row
        # carrying only F4 follows its timing evidence (trusted here) and is
        # not review-required; the code still appears in the reason codes.
        report = verified_report()
        report["assignments"][0]["self_check_flags"] = ["F4-ctc-final-disagreement"]
        auto_lrc.update_report_confidence_metrics(report)
        self.assert_bound(report, trusted=True, review=False)
        self.assertIn("F4-ctc-final-disagreement", report["assignments"][0]["final_trust_reason_codes"])

    def test_final_trust_classification_table_and_idempotence(self) -> None:
        self.assertTrue(callable(auto_lrc.final_line_trust))
        table = auto_lrc.FINAL_LINE_TRUST_CLASSIFICATION
        self.assertEqual(table["previous-tail-overlap"], "REVIEW")
        self.assertEqual(table["previous-tail-ownership-unknown"], "UNTRUSTED")
        self.assertEqual(table["manual_anchor_resolved"], "INFO")
        report = verified_report()
        report["assignments"][0]["flags"] = ["previous-tail-overlap"]
        auto_lrc.update_report_confidence_metrics(report)
        first = copy.deepcopy(report)
        auto_lrc.update_report_confidence_metrics(report)
        self.assertEqual(first, report)


if __name__ == "__main__":
    unittest.main()