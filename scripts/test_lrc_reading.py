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


class ReadingDictionaryTests(unittest.TestCase):
    def test_bj_shipped_dictionary_fixes_the_islet_line(self):
        self.assertEqual(lrc_reading.romanize_line("飛沫く波と嘶き"), "shibuku nami to inanaki")
        self.assertEqual(lrc_reading.romanize_line("飛沫いて"), "shibuite", "conjugated forms follow the stem")
        self.assertEqual(lrc_reading.romanize_line("微睡みの中"), "madoromi no naka")
        self.assertEqual(lrc_reading.romanize_line("漁った"), "asatta")
        self.assertEqual(lrc_reading.romanize_line("過去形"), "kakokei")

    def test_longest_match_conjugation_and_context(self):
        entries = lrc_reading.parse_overrides("# c\n飛沫\tひまつ\n飛沫く\tしぶく\n謎語\tなぞご\t鍵\nbad\tkana\n壊\tkana\n")
        self.assertEqual([item[0] for item in entries], ["飛沫く", "飛沫", "謎語"], "longest surface first; bad rows dropped")
        self.assertEqual(lrc_reading.apply_overrides("飛沫く", entries), ("しぶく", 1))
        self.assertEqual(lrc_reading.apply_overrides("飛沫かない", entries)[0], "しぶかない")
        self.assertEqual(lrc_reading.apply_overrides("飛沫", entries)[0].strip(chr(0xE000) + chr(0xE001)), "ひまつ", "the plain noun entry still applies")
        self.assertEqual(lrc_reading.apply_overrides("謎語", entries)[1], 0, "context text missing")
        self.assertEqual(lrc_reading.apply_overrides("謎語 鍵", entries)[1], 1)

    def test_alignment_reading_ignores_the_dictionary_unless_flagged(self):
        import os
        self.assertNotEqual(lrc_reading.romanize_compact("飛沫く波と嘶き"), "shibukunamitoinanaki")
        self.assertEqual(lrc_reading.romanize_compact("飛沫く波と嘶き", use_dictionary=True), "shibukunamitoinanaki")
        os.environ[lrc_reading.ALIGN_DICT_ENV] = "1"
        try:
            self.assertEqual(lrc_reading.romanize_compact("飛沫く波と嘶き"), "shibukunamitoinanaki")
        finally:
            del os.environ[lrc_reading.ALIGN_DICT_ENV]

    def test_typed_reading_to_romaji(self):
        self.assertEqual(lrc_reading.kana_text_to_romaji("しぶく なみ ToInanaki"), "shibuku nami toinanaki")


if __name__ == "__main__":
    unittest.main()
