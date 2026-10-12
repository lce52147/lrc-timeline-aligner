from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import auto_lrc


class TimestampMutationTrustTests(unittest.TestCase):
    def test_opening_backtrack_invalidates_prior_timing_trust(self) -> None:
        features = auto_lrc.AudioFeatures(
            duration=40.0,
            frame_times=np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(400, -20.0, dtype=np.float32),
            onset_strength=np.zeros(400, dtype=np.float32),
            segments=[],
        )
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0},
                {
                    "entry": 2,
                    "timestamp": 20.0,
                    "ctc_local_fusion": True,
                    "ctc_score": 0.20,
                    "timing_trusted": True,
                    "timing_audit_trusted": True,
                    "overall_trusted": True,
                    "review_required": False,
                    "timing_trusted_reason": "old-clock-proof",
                    "timing_trusted_sources": ["ctc"],
                    "ctc_token_spans": [
                        *[
                            {"start": 20.0 + i * 0.02, "end": 20.01 + i * 0.02, "score": 0.01}
                            for i in range(4)
                        ],
                        {"start": 20.08, "end": 20.09, "score": 0.40},
                    ],
                },
                {"entry": 3, "timestamp": 30.0},
            ],
            "suspicious_alignments": [],
        }
        old_analyze = auto_lrc.analyze_audio
        old_first = auto_lrc.first_local_onset
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        auto_lrc.first_local_onset = lambda *_args, **_kwargs: 19.5  # type: ignore[assignment]
        try:
            _, result, changes = auto_lrc.apply_hybrid_ctc_opening_onset_backtrack(
                Path("dummy.flac"), [10.0, 20.0, 30.0], report, 40.0
            )
        finally:
            auto_lrc.analyze_audio = old_analyze  # type: ignore[assignment]
            auto_lrc.first_local_onset = old_first  # type: ignore[assignment]
        self.assertEqual(len(changes), 1)
        row = result["assignments"][1]  # type: ignore[index]
        self.assertFalse(row["timing_trusted"])
        self.assertFalse(row["timing_audit_trusted"])
        self.assertFalse(row["overall_trusted"])
        self.assertEqual(row["verification_status"], "unverified")
        self.assertNotIn("timing_trusted_sources", row)

    def test_vocal_onset_move_invalidates_prior_timing_trust(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.02), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 18.8))[0][0]] = 0.95
        onset_strength[np.where(np.isclose(frame_times, 20.2))[0][0]] = 0.40
        features = auto_lrc.AudioFeatures(
            30.0,
            frame_times,
            np.full_like(frame_times, -20.0),
            onset_strength,
            [],
        )
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0},
                {
                    "entry": 2,
                    "timestamp": 20.0,
                    "timing_trusted": True,
                    "timing_audit_trusted": True,
                    "overall_trusted": True,
                    "review_required": False,
                    "timing_trusted_reason": "old-clock-proof",
                    "timing_trusted_sources": ["whisperx"],
                },
                {"entry": 3, "timestamp": 25.0},
            ]
        }
        ctc_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0},
                {"entry": 2, "timestamp": 18.8},
                {"entry": 3, "timestamp": 25.0},
            ]
        }
        old = auto_lrc.vocal_onset_features
        auto_lrc.vocal_onset_features = lambda *_args, **_kwargs: (features, None)  # type: ignore[assignment]
        try:
            _, result, changes = auto_lrc.apply_vocal_onset_tiebreak(
                [10.0, 20.0, 25.0],
                report,
                ctc_report,
                Path("dummy.flac"),
                30.0,
                object(),  # type: ignore[arg-type]
            )
        finally:
            auto_lrc.vocal_onset_features = old  # type: ignore[assignment]
        self.assertEqual(len(changes), 1)
        row = result["assignments"][1]  # type: ignore[index]
        self.assertFalse(row["timing_trusted"])
        self.assertFalse(row["timing_audit_trusted"])
        self.assertFalse(row["overall_trusted"])
        self.assertEqual(row["verification_status"], "unverified")
        self.assertNotIn("timing_trusted_sources", row)


if __name__ == "__main__":
    unittest.main()