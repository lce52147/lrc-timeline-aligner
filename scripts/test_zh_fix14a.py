"""zh_traditional, round 14A: a preference applies where stock picks the word (two-step dispute check), the kana list
counts 我的 他的 都是 only at a word start, and a whole-line entry with the review page's closing mark."""
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


def _disputes():
    """``{word: [crossing stock phrase]}`` for every value-replacement preference: a stock phrase that overlaps the word without being
    inside it and writes the overlapping characters differently (the first step of the check, computed straight from STPhrases)."""
    stock = zh_traditional._base()[0]
    found = {}
    for entry in zh_traditional.PREFERENCES:
        if entry.key not in stock:
            continue
        key, value = entry.key, entry.value
        rows = []
        for phrase, written in stock.items():
            if phrase == key or phrase in zh_traditional.DROPPED_PHRASES:
                continue
            for offset in range(-len(phrase) + 1, len(key)):
                low, high = offset, offset + len(phrase)
                if (low >= 0 and high <= len(key)) or phrase[max(low, 0) - low:min(high, len(key)) - low] != key[max(low, 0):min(high, len(key))]:
                    continue
                if len(written) != len(phrase) or any(written[index - low] != value[index] for index in range(max(low, 0), min(high, len(key)))):
                    rows.append(phrase)
        if rows:
            found[key] = rows
    return found


# (text, converted). The stock segmentation picks the preference word: the preference applies.
PICKED = (
    ("我了解了", "我了解了"), ("你了解了吗", "你了解了嗎"), ("终于了解了", "終於了解了"), ("了解了一切", "了解了一切"), ("克制住自己", "克制住自己"),
    ("注定了", "注定了"), ("你注定了孤单", "你注定了孤單"), ("这个计划一定会成功", "這個計畫一定會成功"), ("计划一下", "計畫一下"),
    ("计划分三步走", "計畫分三步走"), ("胡同学", "胡同學"), ("胡同志", "胡同志"), ("耗尽可能不够用", "耗盡可能不夠用"), ("胡同学你好", "胡同學你好"),
    ("点赞同意", "點贊同意"), ("旅游动着", "旅遊動著"),
)
# The dispute is real (计划|分 against 划分) and the character in front is in the closed set: the stock reading.
KEPT = (
    ("预计划分三组进行", "預計劃分三組進行"), ("我估计划算不来", "我估計劃算不來"), ("统计划分", "統計劃分"), ("设计划船路线清晰", "設計劃船路線清晰"),
    ("总统计划分三步", "總統計劃分三步"), ("一周而复始", "一週而復始"),
)


class PreferenceDisputeTests(unittest.TestCase):
    def test_which_crossing_phrases_dispute_which_word(self) -> None:
        found = _disputes()
        self.assertEqual(set(found), {"糊里糊涂", "克制", "女生外向", "计划", "了解", "胡同", "尽可能", "背包", "背负", "回归", "熏染", "周而复始", "注定"})
        self.assertIn("划分", found["计划"])
        self.assertNotIn("解了", found.get("了解", ()))   # the same 解: no dispute
        self.assertNotIn("划分", found.get("了解", ()))
        for word in ("死胡同", "命中注定", "命运注定", "五百姻缘天注定"):
            self.assertNotIn(word, found)

    def test_the_closed_sets_belong_to_disputed_words(self) -> None:
        found = _disputes()
        for word, letters in zh_traditional.KEEP_STOCK_AFTER.items():
            self.assertIn(word, found)
            self.assertTrue(letters and len(set(letters)) == len(letters), word)

    def test_the_word_the_segmentation_picks_keeps_the_preference(self) -> None:
        for text, want in PICKED:
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_a_real_dispute_after_a_character_of_the_set_keeps_stock(self) -> None:
        for text, want in KEPT:
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_with_the_tables_empty_it_is_stock(self) -> None:
        import opencc
        stock = opencc.OpenCC("s2tw")
        with mock.patch.object(zh_traditional, "PREFERENCES", ()), mock.patch.object(zh_traditional, "CORRECTIONS", ()), \
                mock.patch.object(zh_traditional, "WHOLE_LINES", ()), mock.patch.object(zh_traditional, "CONTEXT_RULES", ()), \
                mock.patch.object(zh_traditional, "DROPPED_PHRASES", frozenset()), mock.patch.object(zh_traditional, "VARIANT_PREFERENCE", {}), \
                mock.patch.object(zh_traditional, "KEEP_STOCK_AFTER", {}):
            _rebuild()
            self.addCleanup(_rebuild)
            for text, _want in PICKED + KEPT:
                self.assertEqual(zh_traditional.convert_text(text), stock.convert(text), text)


class KanaWordStartTests(unittest.TestCase):
    def _checked(self, *texts):
        rows = [{"entry": 1, "lyrics": ["さよなら"] + list(texts)}]
        info = zh_traditional.song_candidates(rows, ())
        return {item["before"]: item["default_checked"] for item in info["lines"]}

    def test_at_a_word_start_the_words_count(self) -> None:
        got = self._checked("我的未来", "他的声音", "都是我的错", "我不会走", "怎么了")
        self.assertEqual(got, {"我的未来": True, "他的声音": True, "都是我的错": True, "我不会走": True, "怎么了": True})

    def test_after_a_han_character_they_do_not(self) -> None:
        texts = ("排他的条件", "利他的条件", "自我的条件", "自主的学会", "合理的体制", "京都是古都")
        got = self._checked(*texts)
        self.assertTrue(set(got) <= set(texts))
        self.assertFalse(any(got.values()), got)
        for text in texts:
            self.assertEqual(zh_traditional._chinese_function_words(text), 0, text)
        self.assertEqual(zh_traditional._chinese_function_words("，我的未来"), 1)
        self.assertEqual(zh_traditional._chinese_function_words("想 他的声音"), 1)


class ReviewMarkTests(unittest.TestCase):
    def test_a_whole_line_with_the_closing_mark(self) -> None:
        for entry in zh_traditional.WHOLE_LINES:
            for mark in (" *", " **", "\t*", "*", "  * ", " ***"):
                text, marks = zh_traditional.convert_marked(entry.key + mark)
                self.assertEqual(text, entry.value + mark, (entry.key, mark))
                for item in marks:
                    self.assertEqual(entry.key[item["at"]], item["from"])
            text, marks = zh_traditional.convert_marked("  " + entry.key + " *")
            self.assertEqual(text, "  " + entry.value + " *")
            self.assertTrue(all(entry.key[item["at"] - 2] == item["from"] for item in marks))
            self.assertEqual(zh_traditional.convert_line(entry.key + " *"), entry.value + " *")

    def test_a_longer_line_with_the_mark_is_stock(self) -> None:
        self.assertEqual(zh_traditional.convert_text("这份作业，干了就没事了 *"), "這份作業，幹了就沒事了 *")
        self.assertEqual(zh_traditional.convert_text("干了就没事了吧 **"), "幹了就沒事了吧 **")
        self.assertEqual(zh_traditional.convert_text("方圆十海里有各种鱼类在游动 *"), "方圓十海里有各種魚類在游動 *")
        self.assertEqual(zh_traditional.convert_text("干了 就没事了 *"), "幹了 就沒事了 *")


if __name__ == "__main__":
    unittest.main()
