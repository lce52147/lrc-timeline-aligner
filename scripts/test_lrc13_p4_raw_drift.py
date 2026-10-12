import unittest

import auto_lrc
from auto_lrc import LyricEntry


class RawTimingDriftFalsificationBoundTests(unittest.TestCase):
    def state_with_constant_raw_offset(self, offset: float):
        entries = [LyricEntry(["alpha"]), LyricEntry(["beta"]), LyricEntry(["gamma"])]
        ctc_times = [10.0, 20.0, 30.0]
        raw_assignments = [
            {
                "timestamp": time + offset,
                "score": 0.95,
                "segment": index + 1,
                "borrowed": False,
            }
            for index, time in enumerate(ctc_times)
        ]
        candidate_sets = []
        for index, ctc_time in enumerate(ctc_times):
            spans = tuple(
                auto_lrc.TimingTokenSpan(
                    ctc_time + token * 0.08,
                    ctc_time + token * 0.08 + 0.04,
                    0.95,
                    chr(97 + token),
                )
                for token in range(4)
            )
            ctc = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entries[index]),
                source="ctc-current",
                raw_time=ctc_time,
                spans=spans,
                confidence=0.99,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"ctc": index},
            )
            binding = auto_lrc._entry_occurrence_binding(
                entries, raw_assignments, index
            )
            occurrence = auto_lrc.candidate_occurrence_evidence_from_binding(
                entries,
                index,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=binding,
                upstream_evidence_revision=f"raw-{index}",
            )
            self.assertIsNotNone(occurrence)
            raw = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entries[index]),
                source="raw-vocal-independent-fusion",
                raw_time=ctc_time + offset,
                spans=(),
                confidence=0.8,
                identity_support="supported",
                sequence_support="supported",
                occurrence_evidence=occurrence,
                independent_content_identity=True,
                current=False,
                source_artifact={"raw": index},
            )
            candidate_sets.append((ctc, raw))
        state = auto_lrc.build_final_timing_state(tuple(candidate_sets))
        self.assertTrue(all(audit.timing_trusted for audit in state.audits))
        return state, tuple(entries)

    def test_three_second_constant_offset_does_not_falsify_raw_timing(self):
        state, entries = self.state_with_constant_raw_offset(3.0)
        self.assertEqual(
            auto_lrc._raw_timing_drift_falsification_details(state, entries),
            (),
        )

    def test_one_and_half_second_constant_offset_can_falsify_raw_timing(self):
        state, entries = self.state_with_constant_raw_offset(1.5)
        details = auto_lrc._raw_timing_drift_falsification_details(state, entries)
        self.assertEqual(len(details), 3)
        self.assertTrue(
            all("contiguous-drift" in item["proof_modes"] for item in details)
        )


if __name__ == "__main__":
    unittest.main()
