import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_recurrence import (
    AcousticRepeatMatch, AcousticRepeatScan,
    vocal_cepstral_features, scan_vocal_recurrences, unique_repeat_anchor,
)


class SegmentRecurrenceTests(unittest.TestCase):
    def test_audio_features_do_not_require_scipy(self):
        rng = np.random.default_rng(1)
        frames = rng.normal(size=(90, 512)).astype(np.float32)
        result, db = vocal_cepstral_features(frames, 16000)
        self.assertEqual(result.shape, (90, 15))
        self.assertEqual(db.shape, (90,))
        self.assertTrue(np.isfinite(result).all())

    def test_identical_acoustic_phrase_recovers_cross_time_repeat(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1200, 15)).astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        x[400:550] = x[100:250]
        rms = np.full(1200, -20, dtype=np.float32)
        scan = scan_vocal_recurrences(
            x, rms, hop_seconds=.02, template_start=2., template_end=5.,
            min_similarity=.95,
        )
        self.assertEqual(scan.status, "RESEARCH_UNVERIFIED_OCCURRENCE")
        self.assertTrue(any(abs(item.time - 8.) < .03 for item in scan.matches))

    def test_no_audio_activity_cannot_authenticate_repeat(self):
        x = np.ones((600, 15), dtype=np.float32) / np.sqrt(15)
        rms = np.full(600, -100., dtype=np.float32)
        scan = scan_vocal_recurrences(
            x, rms, hop_seconds=.02, template_start=2., template_end=5.,
        )
        self.assertEqual(scan.status, "BLOCKED_NO_VOCAL_ACTIVITY")
        self.assertEqual(scan.matches, ())

    def test_ambiguous_occurrences_fail_closed(self):
        scan = AcousticRepeatScan(
            "RESEARCH_UNVERIFIED_OCCURRENCE", 5.7,
            (AcousticRepeatMatch(155.18, .84),
             AcousticRepeatMatch(164.78, .63),
             AcousticRepeatMatch(174.38, .67)),
        )
        self.assertIsNone(unique_repeat_anchor(
            scan, earliest=152., latest=169., min_similarity=.60,
        ))
        found = unique_repeat_anchor(
            scan, earliest=162., latest=165., min_similarity=.60,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.time, 164.78)

    def test_poor_similarity_does_not_count(self):
        scan = AcousticRepeatScan(
            "RESEARCH_UNVERIFIED_OCCURRENCE", 5.7,
            (AcousticRepeatMatch(164.78, .42),),
        )
        self.assertIsNone(unique_repeat_anchor(scan, earliest=163., latest=167.))

    def test_reject_impossible_reference(self):
        x = np.ones((100, 15), dtype=np.float32)
        rms = np.full(100, -20., dtype=np.float32)
        result = scan_vocal_recurrences(
            x, rms, hop_seconds=.02, template_start=0., template_end=4.,
        )
        self.assertEqual(result.status, "BLOCKED_TEMPLATE_BOUNDS")


if __name__ == "__main__":
    unittest.main()
