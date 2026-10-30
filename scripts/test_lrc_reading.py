import os
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


class FuriganaTests(unittest.TestCase):
    TOKI = lrc_reading.clean_segments([["時間", "とき"], ["の", ""], ["流", "なが"], ["れ", ""]])

    def test_furigana_beats_the_analyzer_reading(self):
        self.assertEqual(lrc_reading.romanize_line("時間の流れ"), "jikan no nagare")
        self.assertEqual(lrc_reading.romanize_line("時間の流れ", furigana=self.TOKI), "toki no nagare")
        self.assertEqual(lrc_reading.romanize_line("運命を信じて", furigana=lrc_reading.clean_segments(
            [["運命", "さだめ"], ["を", ""], ["信", "しん"], ["じて", ""]])), "sadame wo shinjite")

    def test_furigana_is_mapped_onto_tokens_so_particles_and_spacing_stay(self):
        seg = lrc_reading.clean_segments([["私", "わたし"], ["は", ""], ["行", "い"], ["く", ""]])
        self.assertEqual(lrc_reading.romanize_line("私は行く", furigana=seg), "watashi wa iku")

    def test_reading_priority_manual_furigana_dictionary_auto(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = Path(tmp) / "d.tsv"
            dictionary.write_text("時間\tじかん\n", encoding="utf-8")
            old = os.environ.get(lrc_reading.OVERRIDES_ENV)
            os.environ[lrc_reading.OVERRIDES_ENV] = str(dictionary)
            try:
                furigana = {lrc_reading.line_key("時間の流れ"): self.TOKI}
                self.assertEqual(lrc_reading.reading_source("時間の流れ", furigana, {"時間の流れ": "x"}), "manual")
                self.assertEqual(lrc_reading.reading_source("時間の流れ", furigana), "furigana")
                self.assertEqual(lrc_reading.reading_source("時間の流れ", {}), "dict")
                self.assertEqual(lrc_reading.reading_source("流れ", {}), "auto")
                # ruby that leaves a kanji uncovered does not claim the line
                partial = {lrc_reading.line_key("時間の流れ"): lrc_reading.clean_segments([["時間", "とき"], ["の流れ", ""]])}
                self.assertEqual(lrc_reading.reading_source("時間の流れ", partial), "dict")
                self.assertEqual(lrc_reading.romanize_line("時間の流れ", furigana=self.TOKI), "toki no nagare", "furigana wins over the dictionary")
            finally:
                if old is None:
                    os.environ.pop(lrc_reading.OVERRIDES_ENV, None)
                else:
                    os.environ[lrc_reading.OVERRIDES_ENV] = old

    def test_non_kana_ruby_is_dropped_and_alignment_target_is_unchanged_without_flag(self):
        self.assertEqual(lrc_reading.clean_segments([["時間", "jikan"]]), [("時間", "")])
        self.assertEqual(lrc_reading.romanize_compact("時間の流れ"), "jikannonagare")
        self.assertEqual(lrc_reading.romanize_compact("時間の流れ", furigana=self.TOKI), "tokinonagare")

    def test_sidecar_round_trip(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get(lrc_reading.FURIGANA_ENV)
            os.environ[lrc_reading.FURIGANA_ENV] = tmp
            try:
                (Path(tmp) / "song.furigana.json").write_text(json.dumps({"lines": {"時間の流れ": [["時間", "とき"], ["の", ""]]}}, ensure_ascii=False), encoding="utf-8")
                self.assertEqual(lrc_reading.load_furigana("song")[lrc_reading.line_key("時間の流れ")][0], ("時間", "とき"))
                self.assertEqual(lrc_reading.load_furigana("missing"), {})
            finally:
                if old is None:
                    os.environ.pop(lrc_reading.FURIGANA_ENV, None)
                else:
                    os.environ[lrc_reading.FURIGANA_ENV] = old

    def test_asr_mismatch_flag(self):
        import furigana_audit as audit
        reading = lrc_reading.reading_kana("時間の流れ", self.TOKI)
        heard_same = [(0.0, 5.0, "トキノナガレ")]
        distance, _heard, _coverage = audit.asr_check("", heard_same, 0.0, 5.0, reading)
        self.assertLess(distance, audit.ASR_FLAG_DISTANCE)
        self.assertFalse(audit.is_flagged(distance, _coverage))
        heard_near = [(0.0, 5.0, "ジカンノナガレ")]  # the ASR heard the same line with another reading of 時間
        distance, heard, coverage = audit.asr_check("", heard_near, 0.0, 5.0, reading)
        self.assertIn("ノナガレ", heard)
        self.assertTrue(audit.is_flagged(distance, coverage))
        heard_other = [(0.0, 5.0, "ミライヲシンジテ")]  # a different line: the ASR missed it, nothing to compare
        distance, _heard, coverage = audit.asr_check("", heard_other, 0.0, 5.0, reading)
        self.assertFalse(audit.is_flagged(distance, coverage))
        heard_other = heard_near
        self.assertIsNone(audit.asr_check("", heard_other, 50.0, 60.0, reading), "no ASR text in the window means no check")

    def test_ruby_parser_reads_utaten_markup(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lyricsee"))
        from utaten_source import parse_utaten_ruby
        page = ('<div class="hiragana">\n <span class="ruby"><span class="rb">時間</span><span class="rt">とき</span></span>の'
                '<span class="ruby"><span class="rb">流</span><span class="rt">なが</span></span>れ<br />\n</div>')
        self.assertEqual(parse_utaten_ruby(page), [[("時間", "とき"), ("の", ""), ("流", "なが"), ("れ", "")]])


if __name__ == "__main__":
    unittest.main()
