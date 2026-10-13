from __future__ import annotations

import unittest

import auto_lrc


def spans(start: float, extent: float, count: int = 4) -> tuple[auto_lrc.TimingTokenSpan, ...]:
    step = extent / count
    return tuple(
        auto_lrc.TimingTokenSpan(
            start=round(start + step * i, 3),
            end=round(start + step * i + min(step, 0.02), 3),
            score=0.4,
            token=chr(97 + i),
        )
        for i in range(count)
    )


def candidate(
    entry: int,
    time: float,
    *,
    source: str,
    extent: float,
    current: bool = False,
    confidence: float = 0.3,
) -> auto_lrc.TimingCandidate:
    return auto_lrc.make_timing_candidate(
        entry_index=entry,
        entry_text=f"line {entry}",
        source=source,
        raw_time=time,
        spans=spans(time, extent),
        confidence=confidence,
        current=current,
    )


class CollapsedCtcPathFallbackTests(unittest.TestCase):
    def build(self, middle_extent: float, local_time: float) -> auto_lrc.FinalTimingState:
        sets = (
            (candidate(0, 10.0, source="ctc-current", extent=2.0, current=True),),
            (
                candidate(1, 30.0, source="ctc-current", extent=middle_extent, current=True),
                candidate(1, local_time, source="ctc-local-window", extent=2.5),
            ),
            (candidate(2, 40.0, source="ctc-current", extent=2.0, current=True),),
        )
        return auto_lrc.build_final_timing_state(sets)

    def test_extent_detector(self) -> None:
        self.assertTrue(auto_lrc.ctc_line_extent_collapsed(10.0, 10.3))
        self.assertTrue(auto_lrc.ctc_line_extent_collapsed(10.0, 52.0))
        self.assertFalse(auto_lrc.ctc_line_extent_collapsed(10.0, 13.0))

    def test_unresolved_row_in_collapsed_path_uses_local_window(self) -> None:
        state = self.build(middle_extent=0.3, local_time=14.0)
        decision = state.decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.source, "ctc-local-window")
        self.assertAlmostEqual(decision.written_time.seconds, 14.0)
        self.assertIn(
            "ctc-current",
            {item.candidate.source for item in decision.rejected_candidates},
        )

    def test_sane_path_keeps_current(self) -> None:
        state = self.build(middle_extent=3.0, local_time=14.0)
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")

    def test_fallback_never_passes_the_next_sane_line(self) -> None:
        state = self.build(middle_extent=0.3, local_time=41.0)
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")

    def test_fallback_never_precedes_the_previous_line(self) -> None:
        state = self.build(middle_extent=0.3, local_time=9.0)
        self.assertEqual(state.decisions[1].candidate.source, "ctc-current")

    def test_neighbour_collapse_marks_the_region(self) -> None:
        sets = (
            (candidate(0, 10.0, source="ctc-current", extent=20.0, current=True),),
            (
                candidate(1, 30.0, source="ctc-current", extent=3.0, current=True),
                candidate(1, 33.0, source="ctc-local-window", extent=2.5),
            ),
            (candidate(2, 40.0, source="ctc-current", extent=2.0, current=True),),
        )
        state = auto_lrc.build_final_timing_state(sets)
        self.assertEqual(state.decisions[1].candidate.source, "ctc-local-window")


if __name__ == "__main__":
    unittest.main()
