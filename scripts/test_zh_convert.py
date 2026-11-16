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


class WordLevelVariantTests(unittest.TestCase):
    """註 / 注 and 嚮 / 向 are variants only in some words: they are word rules, not part of the character preference."""

    def test_the_character_preference_holds_only_characters_that_are_variants_in_every_word(self) -> None:
        self.assertNotIn("註", zh_traditional.VARIANT_PREFERENCE)
        self.assertNotIn("嚮", zh_traditional.VARIANT_PREFERENCE)
        self.assertEqual(set(zh_traditional.VARIANT_PREFERENCE), set("臺溼汙燬絃悽歎唸"))

    def test_words_that_are_written_with_zhu_stay_zhu_and_the_others_are_converted(self) -> None:
        expected = {
            "注册": "註冊", "注解": "註解", "注释": "註釋", "注明": "註明", "注销": "註銷", "备注": "備註", "附注": "附註",
            "批注": "批註", "标注": "標註", "脚注": "腳註", "注生娘娘": "註生娘娘",
            "注定": "注定", "命中注定": "命中注定", "命运注定": "命運注定", "注意": "注意", "专注": "專注",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_words_with_xiang(self) -> None:
        expected = {
            "向导": "嚮導", "心向往之": "心嚮往之", "向往": "嚮往", "方向": "方向", "女生外向": "女生外向",
            "面向对象语言": "面向對象語言",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_a_phrase_cannot_take_the_last_character_of_the_word_before_it(self) -> None:
        expected = {
            "关注明天": "關注明天", "关注销量": "關注銷量", "专注上课": "專注上課", "关注文章": "關注文章", "请注明日期": "請註明日期",
            "迈向明天": "邁向明天", "走向往日": "走向往日", "方向导航": "方向導航", "朝向明日": "朝向明日",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_dried_and_did(self) -> None:
        expected = {
            "一起干杯": "一起乾杯", "干了这杯酒": "乾了這杯酒", "喉咙干了": "喉嚨乾了", "眼泪干了": "眼淚乾了",
            "他干了一件大事": "他幹了一件大事", "苦干了一辈子": "苦幹了一輩子", "干了什么": "幹了什麼",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_hair_now_and_the_discovery_reading_of_the_same_characters(self) -> None:
        self.assertEqual(zh_traditional.convert_text("头发现在"), "頭髮現在")
        self.assertEqual(zh_traditional.convert_text("一直发现"), "一直發現")
        self.assertEqual(zh_traditional.convert_text("总是只有"), "總是只有")

    def test_classical_words_keep_the_character_the_dictionary_writes(self) -> None:
        self.assertEqual(zh_traditional.convert_text("毁炎"), "燬炎")
        self.assertEqual(zh_traditional.convert_text("毁犀"), "燬犀")
        self.assertEqual(zh_traditional.convert_text("四库禁毁书丛刋"), "四庫禁燬書叢刋")
        self.assertEqual(zh_traditional.convert_text("烧毁"), "燒毀")                  # an ordinary word follows the preference
        self.assertEqual(zh_traditional.convert_text("琴弦"), "琴弦")
        self.assertEqual(zh_traditional.convert_text("叹息"), "嘆息")

    def test_every_dictionary_word_that_produces_zhu_or_xiang_is_either_converted_or_protected(self) -> None:
        """No OpenCC phrase may turn 注 / 向 into 註 / 嚮 unless it is a word Taiwan writes that way: the dictionary words are
        compared with the converter, and the only phrases that come out with 注 / 向 are the listed corrections."""
        zh_traditional.one_to_many.cache_clear()
        directory = zh_traditional._dictionary_dir()
        raw = {k: v[0] for k, v in zh_traditional._read_dictionary(directory, "STPhrases.txt").items()}
        kept = {key for key, value in raw.items() if ("註" in value and "註" not in zh_traditional.convert_text(key))
                or ("嚮" in value and "嚮" not in zh_traditional.convert_text(key))}
        self.assertEqual(kept, {"命中注定", "命运注定", "注定", "五百姻缘天注定", "女生外向", "面向对象的技术", "面向对象语言", "向明"})


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
