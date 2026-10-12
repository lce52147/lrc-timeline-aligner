import inspect
import unittest

import auto_lrc


def candidate(entry_index: int, time_s: float):
    spans = tuple(
        auto_lrc.TimingTokenSpan(
            time_s + token * 0.08,
            time_s + token * 0.08 + 0.04,
            0.95,
            chr(97 + token),
        )
        for token in range(4)
    )
    return auto_lrc.make_timing_candidate(
        entry_index=entry_index,
        entry_text="abcd",
        source="ctc-current",
        raw_time=time_s,
        spans=spans,
        confidence=0.99,
        identity_support="supported",
        sequence_support="supported",
        current=True,
        source_artifact={"entry": entry_index, "time": time_s},
    )


class PostcentralFinalTailGuardTests(unittest.TestCase):
    def test_final_candidate_is_rechecked_under_final_predecessor_tail(self):
        previous = candidate(0, 10.0)
        current = candidate(1, 10.5)
        tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id=previous.candidate_id,
            previous_generation_revision=previous.generation_revision,
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="valid",
            reliable_tail_start=10.2,
            reliable_tail_end=10.7,
            uncertainty_seconds=0.05,
            confidence=0.9,
            reason="test",
        )
        self.assertFalse(
            auto_lrc._final_candidate_valid_under_predecessor_tail(
                current, previous, tail
            )
        )
        safe = candidate(1, 11.0)
        self.assertTrue(
            auto_lrc._final_candidate_valid_under_predecessor_tail(
                safe, previous, tail
            )
        )

    def test_postcentral_rebind_calls_final_candidate_tail_guard(self):
        source = inspect.getsource(auto_lrc._postcentral_rebind_existing_peer_conflicts)
        self.assertIn("_final_candidate_valid_under_predecessor_tail(", source)
        self.assertIn("final-selection-invalid-under-final-predecessor-tail", source)

    def test_final_tail_resolver_never_picks_latest_external_tail(self):
        source = inspect.getsource(auto_lrc._resolve_final_peer_predecessor_tail)
        self.assertIn("return intrinsic", source)
        self.assertIn("len(selected) == 1", source)
        self.assertNotIn("selected.sort(key=lambda item:", source)


class HubertProbeWindowCoverageTests(unittest.TestCase):
    def test_window_must_cover_every_hypothesis_with_direct_tolerance(self):
        self.assertTrue(
            auto_lrc._hubert_probe_window_covers_hypotheses(
                9.78, 20.22, (1000, 1500, 2000), 22
            )
        )
        self.assertFalse(
            auto_lrc._hubert_probe_window_covers_hypotheses(
                9.78, 19.90, (1000, 1500, 2000), 22
            )
        )
        self.assertFalse(
            auto_lrc._hubert_probe_window_covers_hypotheses(
                10.00, 20.22, (1000, 1500, 2000), 22
            )
        )

    def test_current_probe_planner_enforces_window_coverage(self):
        source = inspect.getsource(
            auto_lrc._postcentral_plan_known_lyric_hubert_current_probe_requests
        )
        self.assertIn("_hubert_probe_window_covers_hypotheses(", source)

    def test_current_probe_consumer_rechecks_window_coverage(self):
        source = inspect.getsource(auto_lrc._postcentral_acquire_known_lyric_hubert_tails)
        self.assertIn("_hubert_probe_window_covers_request_hypotheses(", source)


if __name__ == "__main__":
    unittest.main()