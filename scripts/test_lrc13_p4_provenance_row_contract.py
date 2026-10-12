import unittest
from types import SimpleNamespace

import auto_lrc


class BackendProvenanceRowContractTests(unittest.TestCase):
    @staticmethod
    def selection():
        return SimpleNamespace(selected=SimpleNamespace(
            backend="ctc",
            evaluation_revision="eval",
            report_projection_revision="report",
        ))

    def test_mismatched_assignment_and_central_rows_fail_closed(self):
        report = {
            "assignments": [
                {"entry": 1, "overall_trusted": True},
                {"entry": 2, "overall_trusted": True},
            ],
            "central_timing_state": [
                {"candidate_id": "ctc-row-1", "candidate_source": "ctc-current"},
            ],
        }
        with self.assertRaisesRegex(
            auto_lrc.LrcError, "provenance.*row count"
        ):
            auto_lrc._finalize_backend_provenance_projection(
                report, {"candidates": []}, self.selection()
            )

    def test_missing_central_rows_fail_closed(self):
        report = {"assignments": [{"entry": 1, "overall_trusted": True}]}
        with self.assertRaisesRegex(
            auto_lrc.LrcError, "provenance.*row count"
        ):
            auto_lrc._finalize_backend_provenance_projection(
                report, {"candidates": []}, self.selection()
            )

    def test_matching_rows_preserve_trusted_evidence(self):
        report = {
            "assignments": [{"entry": 1, "overall_trusted": True}],
            "central_timing_state": [
                {"candidate_id": "ctc-row-1", "candidate_source": "ctc-current"}
            ],
        }
        projection = auto_lrc._finalize_backend_provenance_projection(
            report, {"candidates": []}, self.selection()
        )
        self.assertEqual(
            projection["trusted_row_evidence"][0]["candidate_id"],
            "ctc-row-1",
        )


if __name__ == "__main__":
    unittest.main()
