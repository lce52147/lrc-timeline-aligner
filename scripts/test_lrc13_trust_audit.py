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


if __name__ == "__main__":
    unittest.main()