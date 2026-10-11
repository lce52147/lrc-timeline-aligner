"""P3: mutual acoustic recurrence must remain diagnostic evidence."""
import unittest
from types import SimpleNamespace as NS
from unittest import mock
import auto_lrc


class RecurrenceFinalSelectionTests(unittest.TestCase):
    def test_mutual_recurrence_is_never_final_even_if_it_dominates(self):
        current = NS(candidate_id="ctc", source="ctc-current", current=True,
                     written_time=NS(centiseconds=1000, seconds=10.0))
        recurrence = NS(candidate_id="mutual", source="mutual-acoustic-recurrence",
                        current=False, written_time=NS(centiseconds=1800, seconds=18.0))
        def evaluate(candidate, *args, **kwargs):
            return NS(candidate=candidate, valid=True, evidence=NS(candidate_revision=1),
                      rejection_reasons=())
        for policy in ("default", "ctc-first"):
            with (self.subTest(policy=policy),
                mock.patch.object(auto_lrc, "evaluate_timing_candidate", side_effect=evaluate),
                mock.patch.object(auto_lrc, "candidate_dominates", return_value=True),
            ):
                decision = auto_lrc.select_timing_decision(
                    entry_index=0, candidates=(current, recurrence),
                    previous_candidate=None, selection_revision=1,
                    timing_source_policy=policy,
                )
                self.assertEqual(decision.selected_candidate_id, "ctc")
                self.assertIn("mutual", [e.candidate.candidate_id for e in decision.rejected_candidates])


if __name__ == "__main__":
    unittest.main()
