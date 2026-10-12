from __future__ import annotations

import unittest

import auto_lrc


class FusionConsensusLineageTests(unittest.TestCase):
    def test_whisperx_and_raw_cannot_form_two_independent_votes(self) -> None:
        whisperx_report: dict[str, object] = {
            "backend": "whisperx",
            "strategy": "unit-test",
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.55, "timestamp": 20.0},
            ],
            "low_confidence_entries": [
                {"entry": 2, "score": 0.55, "lyric": "second"}
            ],
            "suspicious_alignments": [],
            "review_required_count": 0,
        }
        ctc_report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 15.0, "ctc_score": 0.0},
                {
                    "entry": 2,
                    "timestamp": 22.5,
                    "ctc_score": 0.1,
                    "ctc_token_spans": [
                        {"char": "s", "start": 22.5, "end": 22.6, "score": 0.8}
                    ],
                },
            ],
        }
        raw_report: dict[str, object] = {
            "backend": "whispercpp_raw",
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.90, "timestamp": 10.2},
                {"entry": 2, "segment": 2, "score": 0.76, "timestamp": 22.3},
            ],
        }

        _, report, changes = auto_lrc.apply_ctc_local_fusion_to_whisperx(
            [10.0, 20.0],
            whisperx_report,
            ctc_report,
            duration=30.0,
            raw_report=raw_report,
        )

        self.assertEqual(len(changes), 1)
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertIsNot(assignments[0].get("timing_trusted"), True)  # type: ignore[index]
        self.assertTrue(assignments[1]["timing_trusted"])  # type: ignore[index]
        self.assertEqual(
            set(assignments[1]["timing_trusted_sources"]),  # type: ignore[index]
            {"ctc", "raw"},
        )


if __name__ == "__main__":
    unittest.main()
