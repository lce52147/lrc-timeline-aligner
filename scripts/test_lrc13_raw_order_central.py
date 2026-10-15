"""Central reads independent raw onsets as evidence (never a clock): a CTC time
that contradicts their order is neither valid nor kept over a collapsed
fallback, and two agreeing CTC re-alignments at a heard onset outrank an
unproven whole-song time."""
from __future__ import annotations

import unittest

import auto_lrc
from test_lrc13_incoherent_vocal import candidate, steady

FUSION = auto_lrc.RAW_ORDER_ANCHOR_SOURCE


def build(sets):
    return auto_lrc.build_final_timing_state(sets, timing_source_policy="ctc-first")


class RawOrderTests(unittest.TestCase):
    def row_sets(self, middle):
        # Raw onsets: row 1 at 10 s, row 3 at 20 s.  Row 2's whole-song time
        # (30 s) sits after row 3's heard onset.
        return (
            (
                candidate(0, "ctc-current", 10.0, steady(10.0), current=True),
                candidate(0, FUSION, 10.0, steady(10.0)),
            ),
            middle,
            (
                candidate(2, "ctc-current", 40.0, steady(40.0), current=True),
                candidate(2, FUSION, 20.0, steady(20.0)),
            ),
        )

    def test_current_after_a_later_raw_onset_is_flagged(self) -> None:
        current = candidate(1, "ctc-current", 30.0, steady(30.0), current=True)
        sets = self.row_sets((current,))
        # Row 3's own onset is not a bound on its own time.
        self.assertEqual(auto_lrc.ctc_current_out_of_raw_order(sets), (False, True, False))
        ids = auto_lrc.raw_order_contradicting_candidate_ids(sets)
        self.assertEqual(ids[1], frozenset({current.candidate_id}))
        # The onsets themselves are the reference, never contradicting.
        self.assertFalse(any(c.source == FUSION and c.candidate_id in ids[2] for c in sets[2]))

    def test_contradicting_time_is_not_valid_and_yields_to_the_window(self) -> None:
        middle = (
            candidate(1, "ctc-current", 30.0, steady(30.0), current=True),
            candidate(1, "ctc-local-window", 15.0, steady(15.0, 0.01)),
        )
        decision = build(self.row_sets(middle)).decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.source, "ctc-local-window")
        self.assertAlmostEqual(decision.written_time.seconds, 15.0, places=2)

    def test_ordered_current_is_untouched(self) -> None:
        middle = (
            candidate(1, "ctc-current", 15.0, steady(15.0), current=True),
            candidate(1, "ctc-local-window", 17.0, steady(17.0, 0.01)),
        )
        sets = self.row_sets(middle)
        self.assertEqual(auto_lrc.raw_order_contradicting_candidate_ids(sets)[1], frozenset())
        self.assertAlmostEqual(build(sets).decisions[1].written_time.seconds, 15.0, places=2)


class CorroboratedRealignmentTests(unittest.TestCase):
    def row_sets(self, *extra):
        # Row 2's whole-song time is unproven (weak tokens) and nothing is
        # valid; its path is neither collapsed nor out of raw order.
        return (
            (candidate(0, "ctc-current", 10.0, steady(10.0), current=True),),
            (candidate(1, "ctc-current", 30.0, steady(30.0, 0.01), current=True),) + extra,
            (candidate(2, "ctc-current", 40.0, steady(40.0), current=True),),
        )

    def agreeing(self):
        return (
            candidate(1, "ctc-local-window", 27.0, steady(27.0, 0.01)),
            candidate(1, "ctc-opening-local-retry", 27.05, steady(27.05, 0.01)),
        )

    def test_two_passes_at_the_heard_onset_replace_the_whole_song_time(self) -> None:
        sets = self.row_sets(*self.agreeing(), candidate(1, FUSION, 27.1, steady(27.1)))
        decision = build(sets).decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.source, "ctc-local-window")
        self.assertAlmostEqual(decision.written_time.seconds, 27.0, places=2)

    def test_without_the_heard_onset_the_whole_song_time_stays(self) -> None:
        decision = build(self.row_sets(*self.agreeing())).decisions[1]
        self.assertEqual(decision.candidate.source, "ctc-current")

    def test_one_pass_alone_is_not_enough(self) -> None:
        sets = self.row_sets(self.agreeing()[0], candidate(1, FUSION, 27.1, steady(27.1)))
        self.assertEqual(build(sets).decisions[1].candidate.source, "ctc-current")

    def test_passes_that_disagree_are_not_enough(self) -> None:
        sets = self.row_sets(
            candidate(1, "ctc-local-window", 27.0, steady(27.0, 0.01)),
            candidate(1, "ctc-opening-local-retry", 27.25, steady(27.25, 0.01)),
            candidate(1, FUSION, 27.1, steady(27.1)),
        )
        self.assertEqual(build(sets).decisions[1].candidate.source, "ctc-current")


if __name__ == "__main__":
    unittest.main()
