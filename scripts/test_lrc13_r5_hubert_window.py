from __future__ import annotations

import inspect
import unittest

import auto_lrc


class HubertNeutralWindowTests(unittest.TestCase):
    def test_neutral_window_uses_fixed_leading_context(self) -> None:
        self.assertAlmostEqual(
            auto_lrc._hubert_neutral_probe_window_start(1000),
            10.0 - auto_lrc.LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
        )
        self.assertEqual(auto_lrc._hubert_neutral_probe_window_start(100), 0.0)

    def test_material_conflict_probe_uses_neutral_window_helper(self) -> None:
        source = inspect.getsource(
            auto_lrc._postcentral_plan_known_lyric_hubert_current_probe_requests
        )
        self.assertIn("_hubert_neutral_probe_window_start(", source)
        self.assertNotIn(
            "earliest_seconds - PREFIX_ACOUSTIC_ASSOCIATION_SECONDS",
            source,
        )

    def test_timing_failure_probe_uses_same_neutral_window_helper(self) -> None:
        source = inspect.getsource(
            auto_lrc._postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests
        )
        self.assertIn("_hubert_neutral_probe_window_start(", source)
        self.assertNotIn(
            "earliest_seconds - PREFIX_ACOUSTIC_ASSOCIATION_SECONDS",
            source,
        )

    def test_consumer_rejects_hypothesis_edge_crop(self) -> None:
        base = {
            "hypothesis_bindings": [
                ["a", "ga", 1000],
                ["b", "gb", 1200],
            ],
            "hypothesis_min_centiseconds": 1000,
            "hypothesis_max_centiseconds": 1200,
            "final_selected_centiseconds": 1000,
            "window_end": 12.5,
        }
        biased = {**base, "window_start": 9.78}
        neutral = {
            **base,
            "window_start": auto_lrc._hubert_neutral_probe_window_start(1000),
        }
        tolerance = round(auto_lrc.PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100)
        self.assertFalse(
            auto_lrc._hubert_probe_window_covers_request_hypotheses(
                biased, tolerance
            )
        )
        self.assertTrue(
            auto_lrc._hubert_probe_window_covers_request_hypotheses(
                neutral, tolerance
            )
        )


if __name__ == "__main__":
    unittest.main()
