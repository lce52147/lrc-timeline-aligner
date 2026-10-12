from __future__ import annotations

import unittest

import auto_lrc
from test_core_logic import V142CentralConsensusRegressionTests


class ConsensusFamilyDedupTests(unittest.TestCase):
    def test_duplicate_same_family_candidate_cannot_break_cluster_tie(self) -> None:
        helper = V142CentralConsensusRegressionTests()
        current = helper.candidate("ctc-current", 12.0, current=True, independent=False)
        first_raw = helper.candidate("raw-vocal-independent-fusion", 10.00)
        first_raw_duplicate = helper.candidate("raw-vocal-independent-fusion", 10.08)
        first_recurrence = helper.candidate("global-acoustic-recurrence", 10.05)
        second_raw = helper.candidate("raw-vocal-independent-fusion", 11.00)
        second_recurrence = helper.candidate("global-acoustic-recurrence", 11.05)

        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                current,
                first_raw,
                first_raw_duplicate,
                first_recurrence,
                second_raw,
                second_recurrence,
            ),
            previous_candidate=None,
            selection_revision=1,
        )

        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.candidate_id, current.candidate_id)


if __name__ == "__main__":
    unittest.main()
