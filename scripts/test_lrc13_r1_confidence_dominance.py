from __future__ import annotations

import unittest

import auto_lrc
from test_core_logic import BackendPeerSimulationTests


class CrossLineageConfidenceDominanceTests(unittest.TestCase):
    def test_distant_cross_lineage_candidate_cannot_win_on_confidence_alone(self) -> None:
        helper = BackendPeerSimulationTests()
        current = helper.candidate(
            0,
            1.0,
            source="ctc-current",
            current=True,
            confidence=0.40,
            direct=False,
        )
        challenger = helper.candidate(
            0,
            2.0,
            source="whispercpp-current",
            current=False,
            confidence=0.95,
            direct=False,
        )
        current_eval = auto_lrc.evaluate_timing_candidate(current)
        challenger_eval = auto_lrc.evaluate_timing_candidate(challenger)
        self.assertTrue(current_eval.valid)
        self.assertTrue(challenger_eval.valid)
        self.assertEqual(
            auto_lrc._candidate_vector(current_eval)[:-1],
            auto_lrc._candidate_vector(challenger_eval)[:-1],
        )
        self.assertFalse(
            auto_lrc.candidate_dominates(challenger_eval, current_eval)
        )


if __name__ == "__main__":
    unittest.main()
