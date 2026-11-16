"""zh_traditional.convert_text: phrase segmentation from both directions, the corrections table and the variant preference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import zh_traditional


class SegmentationTests(unittest.TestCase):
    def test_required_words(self) -> None:
        expected = {
            "一起干杯": "一起乾杯", "一起干活": "一起幹活", "干杯": "乾杯", "头发": "頭髮", "发现": "發現", "后来": "後來",
            "皇后": "皇后", "里面": "裡面", "公里": "公里", "面条": "麵條", "一只": "一隻", "只有": "只有", "台风": "颱風",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_stock_forward_match_is_wrong_where_the_both_direction_match_is_right(self) -> None:
        import opencc
        self.assertEqual(opencc.OpenCC("s2tw").convert("一起干杯"), "一起幹杯")
        self.assertEqual(zh_traditional.convert_text("一起干杯"), "一起乾杯")

    def test_tie_goes_to_the_right_match(self) -> None:
        self.assertEqual(zh_traditional.convert_text("总是只有我"), "總是只有我")   # 是只 (left) vs 只有 (right)
        self.assertEqual(zh_traditional.convert_text("一直发现着"), "一直發現著")   # 直发 (left) vs 发现 (right)

    def test_a_correction_phrase_beats_an_unrelated_longer_match(self) -> None:
        self.assertEqual(zh_traditional.convert_text("夹着书签在里面"), "夾著書籤在裡面")
        self.assertEqual(zh_traditional.convert_text("鞋带又松开了"), "鞋帶又鬆開了")

    def test_measure_word_only_versus_only(self) -> None:
        for text, want in {"这只猫": "這隻貓", "那只狗": "那隻狗", "这只是": "這只是", "这只有": "這只有", "这只能": "這只能", "这只想": "這只想"}.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_sentences_with_several_ambiguous_characters(self) -> None:
        self.assertEqual(zh_traditional.convert_text("后来我们一起干杯，头发里面有发现"), "後來我們一起乾杯，頭髮裡面有發現")

    def test_unchanged_and_unrelated_text(self) -> None:
        self.assertEqual(zh_traditional.convert_text("abc 123，。"), "abc 123，。")
        self.assertEqual(zh_traditional.convert_text(""), "")
        self.assertIsNone(zh_traditional.convert_line("這是第一句"))


class VariantTests(unittest.TestCase):
    def test_the_users_more_common_variant_is_written(self) -> None:
        self.assertEqual(zh_traditional.convert_text("舞台上的灯"), "舞台上的燈")      # 台, not 臺
        self.assertEqual(zh_traditional.convert_text("命中注定"), "命中注定")          # 注, not 註
        self.assertEqual(zh_traditional.convert_text("淋湿"), "淋濕")                  # 濕, not 溼
        self.assertEqual(zh_traditional.convert_text("怀念"), "懷念")                  # 念, not 唸
        self.assertEqual(zh_traditional.convert_text("计划"), "計畫")
        self.assertEqual(zh_traditional.convert_text("了解"), "了解")
        self.assertEqual(zh_traditional.convert_text("里面"), "裡面")                  # 裡 (230) beats 裏 (4)

    def test_preference_does_not_touch_traditional_text_that_was_already_there(self) -> None:
        self.assertEqual(zh_traditional.convert_text("臺北的灯"), "臺北的燈")
        self.assertEqual(zh_traditional.convert_text("向往"), "嚮往")                  # a correction phrase keeps its own spelling


class OneToManyTests(unittest.TestCase):
    def test_marks_list_the_alternatives(self) -> None:
        after, marks = zh_traditional.convert_marked("干杯")
        self.assertEqual(after, "乾杯")
        self.assertEqual([(m["at"], m["from"]) for m in marks][0], (0, "干"))
        self.assertEqual(set(marks[0]["options"]), {"幹", "乾", "干"})
        self.assertEqual(zh_traditional.convert_marked("爱你")[1], [])
        self.assertIn("发", zh_traditional.one_to_many())
        self.assertNotIn("爱", zh_traditional.one_to_many())

    def test_without_the_dictionaries_nothing_is_marked(self) -> None:
        zh_traditional.one_to_many.cache_clear()
        self.addCleanup(zh_traditional.one_to_many.cache_clear)
        with mock.patch.object(zh_traditional, "_dicts", False):
            self.assertIsNone(zh_traditional.convert_marked("干杯"))
            self.assertEqual(zh_traditional.one_to_many(), {})


if __name__ == "__main__":
    unittest.main()
