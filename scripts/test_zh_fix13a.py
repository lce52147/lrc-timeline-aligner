"""zh_traditional, round 13A: whole-line entries, preference words that must not take a character away from the stock reading,
and the closed list of Chinese-only words in kana songs (no word that Japanese kanji words contain)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import zh_traditional


def _rebuild() -> None:
    zh_traditional._dicts = None
    zh_traditional.one_to_many.cache_clear()


class WholeLineTests(unittest.TestCase):
    def test_a_whole_line_is_written_the_lyric_way(self) -> None:
        self.assertGreaterEqual(len(zh_traditional.WHOLE_LINES), 2)
        for entry in zh_traditional.WHOLE_LINES:
            self.assertTrue(entry.why.strip(), entry.key)
            self.assertEqual(len(entry.key), len(entry.value), entry.key)
            self.assertEqual(zh_traditional.convert_text(entry.key), entry.value, entry.key)
            self.assertEqual(zh_traditional.convert_text("  " + entry.key + "\t"), "  " + entry.value + "\t", entry.key)
            self.assertEqual(zh_traditional.convert_line(entry.key), entry.value, entry.key)

    def test_the_same_words_in_a_longer_line_go_the_normal_way(self) -> None:
        for entry in zh_traditional.WHOLE_LINES:
            text, want = entry.negative
            self.assertEqual(zh_traditional.convert_text(text), want, text)
            with mock.patch.object(zh_traditional, "WHOLE_LINES", ()):
                _rebuild()
                normal = zh_traditional.convert_text(entry.key)
                self.addCleanup(_rebuild)
            _rebuild()
            for longer in ("啊" + entry.key, entry.key + "啊", "他说，" + entry.key, entry.key + "，他说", entry.key + "\n" + entry.key):
                with mock.patch.object(zh_traditional, "WHOLE_LINES", ()):
                    _rebuild()
                    want = zh_traditional.convert_text(longer)
                _rebuild()
                self.assertEqual(zh_traditional.convert_text(longer), want, longer)
            self.assertNotEqual(normal, entry.value, "the entry changes nothing: " + entry.key)

    def test_the_two_named_lines_and_their_look_alikes(self) -> None:
        self.assertEqual(zh_traditional.convert_text("干了就没事了！"), "乾了就沒事了！")
        self.assertEqual(zh_traditional.convert_text("干了就没事了"), "乾了就沒事了")
        self.assertEqual(zh_traditional.convert_text("海里有各种鱼类在游动"), "海裡有各種魚類在游動")
        self.assertEqual(zh_traditional.convert_text("这份作业，干了就没事了"), "這份作業，幹了就沒事了")
        self.assertEqual(zh_traditional.convert_text("干了就没事了吧"), "幹了就沒事了吧")
        self.assertEqual(zh_traditional.convert_text("方圆十海里有各种鱼类在游动"), "方圓十海里有各種魚類在游動")
        self.assertEqual(zh_traditional.convert_text("海里有各种鱼类在游动了"), "海里有各種魚類在游動了")
        self.assertEqual(zh_traditional.convert_text("海里有各种鱼类在游动，很多"), "海里有各種魚類在游動，很多")

    def test_marks_follow_the_converted_text(self) -> None:
        for entry in zh_traditional.WHOLE_LINES:
            for lead in ("", " ", "\t "):
                text, marks = zh_traditional.convert_marked(lead + entry.key + "  ")
                self.assertEqual(text, lead + entry.value + "  ")
                for mark in marks:
                    self.assertEqual(entry.key[mark["at"] - len(lead)], mark["from"])
                    self.assertIn(text[mark["at"]], mark["options"] + [text[mark["at"]]])
        _text, marks = zh_traditional.convert_marked("干了就没事了！")
        self.assertEqual([mark["at"] for mark in marks][:2], [0, 1])

    def test_the_review_paths_use_it(self) -> None:
        info = zh_traditional.song_candidates([{"entry": 1, "lyrics": ["干了就没事了！"]}, {"entry": 2, "lyrics": ["海里有各种鱼类在游动"]},
                                               {"entry": 3, "lyrics": ["这份作业，干了就没事了"]}], ())
        self.assertEqual([item["after"] for item in info["lines"]], ["乾了就沒事了！", "海裡有各種魚類在游動", "這份作業，幹了就沒事了"])
        self.assertEqual(zh_traditional.convert_header("[ti:海里有各种鱼类在游动]"), ("海里有各种鱼类在游动", "海裡有各種魚類在游動"))

    def test_with_the_table_empty_it_is_the_engine_again(self) -> None:
        with mock.patch.object(zh_traditional, "WHOLE_LINES", ()):
            _rebuild()
            self.addCleanup(_rebuild)
            self.assertEqual(zh_traditional.convert_text("干了就没事了！"), "幹了就沒事了！")
            self.assertEqual(zh_traditional.convert_text("海里有各种鱼类在游动"), "海里有各種魚類在游動")


# (text, converted): a stock phrase crosses the border of the preference word, so the word is not taken (a word stock has keeps the stock value,
# a word stock does not have is left out of the segmentation). One row per PREFERENCES word.
BORDER = (
    ("咸水鱼类在游动", "鹹水魚類在遊動"), ("修正游动在", "修正遊動在"), ("修正游动其中", "修正遊動其中"), ("发交织游动吗", "發交織遊動嗎"),
    ("一游动着", "一遊動著"), ("一游动著", "一遊動著"), ("一游过群星之间", "一遊過群星之間"), 
    ("预计划分三组进行", "預計劃分三組進行"), ("我估计划算不来", "我估計劃算不來"),
    ("点赞同", "點贊同"), 
    ("一回归分析", "一回歸分析"), ("丝线性回归", "絲線性回歸"),
    ("辟逻辑回归", "辟邏輯回歸"), ("一回归模型", "一回歸模型"), ("一回归方程", "一回歸方程"), ("一周而复始", "一週而復始"),
    ("注定义出", "注定義出"), 
)
# A preference word that stock also has is used wherever the segmentation picks it, unless a crossing stock phrase writes the shared characters differently
# and the character in front is in the word's closed set (KEEP_STOCK_AFTER). One row per word that moved from the stock reading (round 13A) to the preference.
PICKED = (
    ("糊里糊涂帐", "糊裡糊塗帳"), ("克制中", "克制中"), ("修女生外向", "修女生外向"), ("了解出", "了解出"), ("胡同一", "胡同一"), ("万死胡同", "萬死胡同"), ("丧尽可能", "喪盡可能"), ("背包伙", "背包夥"), ("背负向", "背負向"), ("回归于", "回歸於"), ("熏染发", "燻染發"), ("出人命中注定", "出人命中注定"), ("出人命运注定", "出人命運注定"), ("于五百姻缘天注定", "於五百姻緣天注定"),
    ("我了解了", "我了解了"), ("你了解了吗", "你了解了嗎"), ("终于了解了", "終於了解了"), ("了解了一切", "了解了一切"), ("克制住自己", "克制住自己"), ("注定了", "注定了"), ("你注定了孤单", "你注定了孤單"),
    ("这个计划一定会成功", "這個計畫一定會成功"), ("计划一下", "計畫一下"), ("计划分三步走", "計畫分三步走"), ("胡同学", "胡同學"), ("胡同志", "胡同志"), ("耗尽可能不够用", "耗盡可能不夠用"), ("胡同学你好", "胡同學你好"),
)
# The closed set decides the real dispute (计划|分 against 划分): the same words after a character of the set keep the stock reading.
CONTESTED = (
    ("预计划分三组进行", "預計劃分三組進行"), ("我估计划算不来", "我估計劃算不來"), ("统计划分", "統計劃分"), ("设计划船路线清晰", "設計劃船路線清晰"), ("总统计划分三步", "總統計劃分三步"),
    ("一周而复始", "一週而復始"),
)
# The same words standing by themselves keep the preference.
STANDING = (
    ("我们的计划", "我們的計畫"), ("这个计划很好", "這個計畫很好"), ("计划", "計畫"), ("我了解你", "我了解你"), ("点赞", "點讚"), ("重点赞美", "重點讚美"),
    ("旅游动着", "旅遊動著"), ("他游动着", "他游動著"), ("我们正游动物园", "我們正遊動物園"), ("线性回归", "線性迴歸"), ("香港回归", "香港回歸"),
    ("死胡同", "死胡同"), ("命中注定", "命中注定"), ("周而复始", "周而復始"), ("背包袱", "揹包袱"),
)
# How each of the 30 PREFERENCES words is built into the engine.
VALUE_REPLACED = {"糊里糊涂", "克制", "女生外向", "计划", "了解", "胡同", "死胡同", "尽可能", "背包", "背负", "回归", "熏染", "周而复始", "注定", "命中注定", "命运注定", "五百姻缘天注定"}
LITERAL_PHRASES = {"鱼类在游动", "正游动在", "正游动其中", "交织游动吗", "游动着", "游动著", "游过群星之间", "点赞", "回归分析", "线性回归", "逻辑回归", "回归模型", "回归方程"}


class PreferenceBorderTests(unittest.TestCase):
    def test_every_preference_is_classified_and_the_classes_are_right(self) -> None:
        keys = {entry.key for entry in zh_traditional.PREFERENCES}
        self.assertEqual(keys, VALUE_REPLACED | LITERAL_PHRASES)
        self.assertEqual(len(keys), 30)
        stock = zh_traditional._base()[0]
        self.assertEqual({key for key in keys if key in stock}, VALUE_REPLACED)
        tables = zh_traditional._tables()
        self.assertEqual(set(tables[8]), VALUE_REPLACED)
        self.assertEqual(set(tables[10]), LITERAL_PHRASES)

    def test_a_stock_phrase_across_the_border_keeps_the_stock_reading(self) -> None:
        for text, want in BORDER + CONTESTED:
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_the_word_on_its_own_keeps_the_preference(self) -> None:
        for text, want in STANDING + PICKED:
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_the_segmentation_is_stock_wherever_the_word_is_stock(self) -> None:
        import opencc
        stock = opencc.OpenCC("s2tw")
        for text in ("预计划分三组进行", "我估计划算不来", "我了解你", "注定了", "他背包去了", "命中注定"):
            self.assertEqual(len(zh_traditional.convert_text(text)), len(stock.convert(text)), text)
            # a stock word that is replaced keeps its place: only the characters of the word itself differ from stock
            ours, theirs = zh_traditional.convert_text(text), stock.convert(text)
            differing = [index for index, (a, b) in enumerate(zip(ours, theirs)) if a != b]
            self.assertTrue(all(any(entry.key in text and text.find(entry.key) <= index < text.find(entry.key) + len(entry.key)
                                    for entry in zh_traditional.PREFERENCES) for index in differing), (text, differing))

    def test_a_literal_phrase_needs_no_stock_phrase_across_its_border(self) -> None:
        for text, want in (("点赞同", "點贊同"), ("点赞助", "點贊助"), ("旅游动着", "旅遊動著"), ("修正游动在", "修正遊動在")):
            self.assertEqual(zh_traditional.convert_text(text), want, text)
        for text, want in (("点赞", "點讚"), ("他点赞了", "他點讚了"), ("正游动在海里", "正游動在海里")):
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_with_the_preferences_empty_nothing_is_contested(self) -> None:
        import opencc
        stock = opencc.OpenCC("s2tw")
        with mock.patch.object(zh_traditional, "PREFERENCES", ()), mock.patch.object(zh_traditional, "CORRECTIONS", ()), \
                mock.patch.object(zh_traditional, "WHOLE_LINES", ()), mock.patch.object(zh_traditional, "CONTEXT_RULES", ()), \
                mock.patch.object(zh_traditional, "DROPPED_PHRASES", frozenset()), mock.patch.object(zh_traditional, "VARIANT_PREFERENCE", {}):
            _rebuild()
            self.addCleanup(_rebuild)
            for text, _want in BORDER + STANDING + PICKED + CONTESTED:
                self.assertEqual(zh_traditional.convert_text(text), stock.convert(text), text)


# Japanese kanji words that contain two characters of the closed list side by side (the 〜的 adjectives, 京都是): none may check a kana song line.
JAPANESE_WORDS = ("排他的", "利他的", "自我的", "自主的", "個人的", "目的", "具体的", "一般的", "基本的", "京都是", "首都是")


class KanaListTests(unittest.TestCase):
    def test_no_japanese_kanji_word_holds_a_word_of_the_list(self) -> None:
        for word in zh_traditional.CHINESE_FUNCTION_WORDS:
            for japanese in JAPANESE_WORDS:
                self.assertNotIn(word, japanese, (word, japanese))
        self.assertNotIn("他的", zh_traditional.CHINESE_FUNCTION_WORDS)
        for text in ("排他的条件", "利他的条件", "自我的条件", "京都是"):
            self.assertEqual(zh_traditional._chinese_function_words(text), 0, text)

    def test_a_japanese_word_with_a_kana_sibling_is_listed_unchecked(self) -> None:
        rows = [{"entry": 1, "lyrics": ["さよなら", "排他的条件"]}, {"entry": 2, "lyrics": ["きみ", "利他的条件"]}, {"entry": 3, "lyrics": ["きみ", "自我的条件"]},
                {"entry": 4, "lyrics": ["きみ", "怎么了"]}]
        info = zh_traditional.song_candidates(rows, ())
        got = {item["before"]: item["default_checked"] for item in info["lines"]}
        self.assertEqual(got, {"排他的条件": False, "利他的条件": False, "自我的条件": False, "怎么了": True})

    def test_the_chinese_translation_lines_are_still_checked(self) -> None:
        rows = [{"entry": 1, "lyrics": ["さよなら", "我不会走"]}, {"entry": 2, "lyrics": ["きみ", "怎么了"]}]
        info = zh_traditional.song_candidates(rows, ())
        got = {item["before"]: item["default_checked"] for item in info["lines"]}
        self.assertTrue(got["我不会走"])
        self.assertTrue(got["怎么了"])


if __name__ == "__main__":
    unittest.main()
