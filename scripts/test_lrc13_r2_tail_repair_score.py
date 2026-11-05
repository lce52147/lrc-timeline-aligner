from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import auto_lrc
import split_patch


class TailRepairScoreTests(unittest.TestCase):
    def test_tail_repair_does_not_promote_content_score(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["anchor"]),
            auto_lrc.LyricEntry(["tail one"]),
            auto_lrc.LyricEntry(["tail two"]),
        ]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 12.0, "score": 0.2},
                {"entry": 3, "timestamp": 14.0, "score": 0.3},
            ],
            "timing_entries": 3,
        }
        features = auto_lrc.AudioFeatures(
            duration=40.0,
            frame_times=np.array([0.0], dtype=np.float32),
            rms_db=np.array([-20.0], dtype=np.float32),
            onset_strength=np.array([0.0], dtype=np.float32),
            segments=[(0.0, 40.0)],
        )
        restores = []
        restores.append(split_patch.assign("low_confidence_tail_start", lambda _report: 1))  # type: ignore[assignment]
        restores.append(split_patch.assign("has_repeated_asr_tail", lambda *_args, **_kwargs: True))  # type: ignore[assignment]
        restores.append(split_patch.assign("analyze_audio", lambda *_args, **_kwargs: features))  # type: ignore[assignment]
        restores.append(split_patch.assign("first_sustained_onset", lambda *_args, **_kwargs: 25.0))  # type: ignore[assignment]
        restores.append(split_patch.assign("weighted_rough_times", lambda *_args, **_kwargs: [25.0, 30.0]))  # type: ignore[assignment]
        restores.append(split_patch.assign("snap_to_onsets", lambda times, *_args, **_kwargs: list(times)))  # type: ignore[assignment]
        try:
            repaired, changes = auto_lrc.repair_tail_hallucination(
                Path("dummy.flac"),
                entries,
                [10.0, 12.0, 14.0],
                report,
                [],
                40.0,
            )
        finally:
            for restore in reversed(restores):
                restore()
        self.assertEqual(repaired, [10.0, 25.0, 30.0])
        self.assertEqual(len(changes), 2)
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertEqual(assignments[1]["score"], 0.2)  # type: ignore[index]
        self.assertEqual(assignments[2]["score"], 0.3)  # type: ignore[index]


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
