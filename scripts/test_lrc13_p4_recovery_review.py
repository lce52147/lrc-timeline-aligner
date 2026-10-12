import inspect
import unittest

import auto_lrc


class RecoveryMovedReviewTests(unittest.TestCase):
    def report(self, *, key: str, old_cs: int, new_cs: int):
        return {
            "assignments": [{
                "entry": 1,
                "flags": [],
                "timing_trusted": True,
                "timing_audit_trusted": True,
                "review_required": False,
            }],
            key: {
                "status": "APPLIED",
                "entries": [{
                    "entry": 1,
                    "status": "TRIGGER",
                    "recovery_disposition": "applied",
                    "old_selected_centiseconds": old_cs,
                    "new_selected_centiseconds": new_cs,
                }],
            },
        }

    def test_hubert_recovery_over_point_three_seconds_requires_review(self):
        report = self.report(
            key=auto_lrc._HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY,
            old_cs=1000,
            new_cs=1031,
        )
        rows = auto_lrc._mark_material_recovery_moves_for_review(report)
        self.assertEqual(len(rows), 1)
        self.assertIn("recovery-moved", report["assignments"][0]["flags"])
        decision = auto_lrc.final_line_trust(
            report["assignments"][0], [], central_committed=False
        )
        self.assertTrue(decision.review_required)
        self.assertFalse(decision.timing_trusted)
        self.assertIn("recovery-moved", decision.reason_codes)

    def test_peer_recovery_over_point_three_seconds_requires_review(self):
        report = self.report(
            key=auto_lrc._PEER_EXISTING_CANDIDATE_RECOVERY_KEY,
            old_cs=2000,
            new_cs=1969,
        )
        rows = auto_lrc._mark_material_recovery_moves_for_review(report)
        self.assertEqual(rows[0]["entry"], 1)
        self.assertEqual(rows[0]["delta_centiseconds"], -31)
        self.assertIn("peer-existing-candidate", rows[0]["recovery_sources"])

    def test_exact_point_three_seconds_does_not_require_recovery_review(self):
        report = self.report(
            key=auto_lrc._HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY,
            old_cs=1000,
            new_cs=1030,
        )
        rows = auto_lrc._mark_material_recovery_moves_for_review(report)
        self.assertEqual(rows, [])
        self.assertNotIn("recovery-moved", report["assignments"][0]["flags"])

    def test_final_output_path_calls_recovery_move_marker(self):
        source = inspect.getsource(auto_lrc.process_audio)
        self.assertIn("_mark_material_recovery_moves_for_review(report)", source)


if __name__ == "__main__":
    unittest.main()
