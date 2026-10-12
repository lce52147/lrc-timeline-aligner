import unittest

import auto_lrc


class MultiChallengerSelectionTests(unittest.TestCase):
    def candidate(self, time: float, confidence: float, *, current: bool = False):
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                time + index * 0.08,
                time + index * 0.08 + 0.04,
                0.9,
                chr(97 + index),
            )
            for index in range(4)
        )
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="line",
            source="ctc-current" if current else "ctc-local-window",
            raw_time=time,
            spans=spans,
            confidence=confidence,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            source_artifact={"time": time, "confidence": confidence},
        )

    def test_unique_dominant_challenger_replaces_current_even_with_two_challengers(self):
        current = self.candidate(10.0, 0.4, current=True)
        weaker = self.candidate(10.3, 0.7)
        stronger = self.candidate(10.6, 0.9)

        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, weaker, stronger),
            previous_candidate=None,
            selection_revision=1,
        )

        self.assertEqual(
            decision.selected_candidate_id,
            stronger.candidate_id,
        )
        self.assertEqual(decision.candidate.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
