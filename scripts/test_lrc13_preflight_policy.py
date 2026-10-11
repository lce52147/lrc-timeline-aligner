"""Regression contracts for the declared LRC 1.3 lyric preflight gates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import auto_lrc


class LyricPreflightPolicyTests(unittest.TestCase):
    def test_raw_asr_below_half_coverage_marks_input_suspect(self):
        lines = ["alpha", "bravo", "charlie", "delta", "echo"]
        entries = [auto_lrc.LyricEntry([line]) for line in lines]
        raw = [auto_lrc.AsrSegment(0, 10, "alpha bravo")]
        result = auto_lrc.raw_asr_lyric_coverage(entries, raw)
        self.assertEqual(result["matched_lines"], 2)
        self.assertEqual(result["coverage_percent"], 40.0)
        self.assertEqual(result["status"], "LOW_COVERAGE")

    def test_more_than_three_order_reversals_marks_input_suspect(self):
        lines = ["alpha", "bravo", "charlie", "delta", "echo"]
        entries = [auto_lrc.LyricEntry([line]) for line in lines]
        raw = [auto_lrc.AsrSegment(0, 10, "echo delta charlie bravo alpha")]
        result = auto_lrc.raw_asr_lyric_coverage(entries, raw)
        self.assertEqual(result["coverage_percent"], 100.0)
        self.assertGreater(result["order_reversals"], 3)
        self.assertEqual(result["status"], "ORDER_INCONSISTENT")

    def test_preflight_reports_forum_credits_and_instrumental_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.lrc"
            path.write_text(
                "[00:01.00]【間奏】\n"
                "[00:03.00]【心得】【歌曲推薦】樓主\n"
                "[00:05.00]作詞: example\n"
                "[00:07.00]bilibili BV1ABCDEFGH\n"
                "[00:09.00]翻譯: sample\n"
                "[00:11.00]♪\n",
                encoding="utf-8",
            )
            document = auto_lrc.load_lyrics(path)
            with mock.patch.object(auto_lrc, "audio_track_labels", return_value=("Artist", "Song")):
                result = auto_lrc.build_lyric_preflight(
                    path, document, document.entries, Path("song.flac")
                )
        self.assertEqual(result["status"], "INPUT_SUSPECT")
        self.assertIn("SUSPECT_GARBAGE", result["reason_codes"])
        self.assertGreaterEqual(len(result["garbage_entries"]), 4)
        kinds = [row["kind"] for row in result["transformations"]]
        self.assertIn("instrumental-marker-removed", kinds)


if __name__ == "__main__":
    unittest.main()
