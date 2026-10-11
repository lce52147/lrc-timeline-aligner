from types import SimpleNamespace
import unittest
import auto_lrc


class CentralReviewRegression(unittest.TestCase):
    def test_far_precentral_candidate_blocks_supersession(self):
        decision = SimpleNamespace(
            evidence=SimpleNamespace(minimum_proof_satisfied=True, structural_failures=()),
            written_time=SimpleNamespace(seconds=20.0, centiseconds=2000),
            entry_index=0,
            rejected_candidates=(),
            candidate=None,
        )
        audit = SimpleNamespace(timing_trusted=True)
        row = {"entry": 1, "flags": ["candidate_disagreement"],
               "candidate_timestamps": {"output": 20.0, "ctc": 18.74}}
        self.assertFalse(auto_lrc._central_decision_supersedes_suspicious_row(decision, audit, row))
        row["candidate_timestamps"]["ctc"] = 19.53
        self.assertTrue(auto_lrc._central_decision_supersedes_suspicious_row(decision, audit, row))

    def test_precentral_disagreement_code_is_reviewable(self):
        self.assertIn("unresolved-precentral-disagreement", auto_lrc.ACTIONABLE_SUSPICIOUS_FLAGS)
