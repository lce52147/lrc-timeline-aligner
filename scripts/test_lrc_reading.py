import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lrc_reading


class ReadingTests(unittest.TestCase):
    def test_regression_lines_are_exact(self):
        self.assertEqual(lrc_reading.romanize_line("解ってるはずなのに私"), "wakatteru hazu nanoni watashi")
        self.assertEqual(lrc_reading.romanize_line("君を好きなまま"), "kimi wo suki na mama")

    def test_particles_and_geminates(self):
        self.assertEqual(lrc_reading.romanize_line("今日は大丈夫"), "kyou wa daijoubu")
        self.assertEqual(lrc_reading.romanize_line("海へ行こう"), "umi e ikou")
        self.assertEqual(lrc_reading.romanize_line("走って走って"), "hashitte hashitte")
        self.assertEqual(lrc_reading.romanize_line("待ってて"), "mattete")

    def test_kana_to_romaji(self):
        self.assertEqual(lrc_reading.kana_to_romaji("ラーメン"), "raamen")
        self.assertEqual(lrc_reading.kana_to_romaji("シャワー"), "shawaa")
        self.assertEqual(lrc_reading.kana_to_romaji("いっしょ"), "issho")
        self.assertEqual(lrc_reading.kana_to_romaji("まっちゃ"), "matcha")

    def test_compact_has_no_spaces_and_non_japanese_is_empty_safe(self):
        self.assertEqual(lrc_reading.romanize_compact("君を好きなまま"), "kimiosukinamama")
        self.assertEqual(lrc_reading.romanize_line(""), "")
        self.assertFalse(lrc_reading.has_japanese("hello"))

    def test_fallback_without_analyzer(self):
        lrc_reading._tagger.cache_clear()
        original = lrc_reading._tagger
        lrc_reading._tagger = lambda: None
        try:
            self.assertEqual(lrc_reading.romanize_line("走って走って"), "hashitte hashitte")
            self.assertIn(" wa ", " " + lrc_reading.romanize_line("私は歩く") + " ")
        finally:
            lrc_reading._tagger = original


if __name__ == "__main__":
    unittest.main()
