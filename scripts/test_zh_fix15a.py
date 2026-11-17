"""zh_traditional, round 15A: a preference spelling also applies inside a longer stock phrase that was written with the stock spelling
(derived mechanically from STPhrases, no hand-written list), and the closed set of characters in front of 计划 also accepts Traditional forms."""
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


# stock picked a longer word that holds the preference word: the preference spelling applies
INSIDE = (
    ("我不了解你", "我不了解你"), ("据了解", "據了解"), ("我克制不住", "我克制不住"), ("无法克制", "無法克制"), ("不克制", "不克制"),
    ("我克制住了眼泪", "我克制住了眼淚"), ("你要克制一下自己", "你要克制一下自己"), ("没有计划", "沒有計畫"), ("计划书", "計畫書"),
    ("有计划地进行", "有計畫地進行"), ("计划好了就出发", "計畫好了就出發"), ("你的计划好吗", "你的計畫好嗎"), ("回归线", "回歸線"),
    ("早有计划", "早有計畫"), ("北回归线", "北回歸線"),
)
# unchanged by this round
BEFORE = (
    ("预计划分三组进行", "預計劃分三組進行"), ("我估计划算不来", "我估計劃算不來"), ("设计划船路线清晰", "設計劃船路線清晰"),
    ("点赞同意", "點贊同意"), ("一周而复始", "一週而復始"), ("我的计划", "我的計畫"), ("很了解你", "很了解你"), ("难以克制", "難以克制"),
)
# an explicit word wins over the derivation; a partly overlapping word that writes the overlap differently blocks it; the closed exception list
EXPLICIT = (("回归分析", "迴歸分析"), ("线性回归", "線性迴歸"), ("逻辑回归", "邏輯迴歸"), ("相克制", "相剋制"), ("背包袱", "揹包袱"))
# a mixed line: the character in front of 计划 written Traditional
MIXED = (("預计划分三组", "預計劃分三組"), ("统计划分", "統計劃分"))

# BH2-FIX16A: a derived phrase also respects KEEP_STOCK_AFTER (M1) and the mixed 周而复始 (N1); 会计划好 is the accepted N1 trade-off of the doc
KEEP = (
    ("线我估计划好了", "線我估計劃好了"), ("预计划好了路线", "預計劃好了路線"), ("我们核计划拨的款项", "我們核計劃撥的款項"), ("审核计划", "審核計劃"),
    ("计划好了就出发", "計畫好了就出發"), ("你的计划好吗", "你的計畫好嗎"), ("我们会计划好的", "我們會計劃好的"), ("兩周而复始", "兩週而復始"),
    ("一周而复始", "一週而復始"), ("周而复始", "周而復始"),
)
LINE4 = (("我不了解生命的意义", "我不瞭解生命的意義"), ("他说我不了解生命的意义", "他說我不了解生命的意義"), ("我不了解你", "我不了解你"))


class DerivedPreferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        _rebuild()
        self.addCleanup(_rebuild)

    def check(self, rows) -> None:
        for text, want in rows:
            with self.subTest(text=text):
                self.assertEqual(zh_traditional.convert_text(text), want)

    def test_preference_inside_a_longer_stock_phrase(self) -> None:
        self.check(INSIDE)

    def test_earlier_results_unchanged(self) -> None:
        self.check(BEFORE)

    def test_explicit_entry_exclusion_and_closed_list(self) -> None:
        self.check(EXPLICIT)

    def test_mixed_line_keeps_stock_after_traditional_set_characters(self) -> None:
        self.check(MIXED)

    def test_derived_phrases_respect_keep_stock_after(self) -> None:
        self.check(KEEP)
        stock = {key: value for key, value in zh_traditional._base()[0].items() if key not in zh_traditional.DROPPED_PHRASES}
        derived = zh_traditional.derived_preferences(stock)
        self.assertNotIn("核计划", derived)

    def test_whole_line_four(self) -> None:
        self.check(LINE4)

    def test_derivation_follows_the_rule(self) -> None:
        stock = {key: value for key, value in zh_traditional._base()[0].items() if key not in zh_traditional.DROPPED_PHRASES}
        derived = zh_traditional.derived_preferences(stock)
        self.assertGreater(len(derived), 50)
        explicit = zh_traditional.entries()
        words = {entry.key: entry for entry in zh_traditional.PREFERENCES if entry.key in stock and stock[entry.key] != entry.value}
        for key, value in derived.items():
            with self.subTest(key=key):
                self.assertNotIn(key, explicit)
                self.assertNotIn(key, zh_traditional.NOT_DERIVED)
                self.assertEqual(len(value), len(key))
                self.assertNotEqual(value, stock[key])
                # every character that differs lies inside one occurrence of a preference word whose stock value was written there
                changed = {index for index, (a, b) in enumerate(zip(value, stock[key])) if a != b}
                covered = set()
                for word, entry in words.items():
                    at = key.find(word)
                    while at >= 0:
                        if stock[key][at:at + len(word)] == stock[word]:
                            covered.update(range(at, at + len(word)))
                        at = key.find(word, at + 1)
                self.assertTrue(changed <= covered)
        for key in ("相克制", "背包袱", "回归分析"):
            self.assertNotIn(key, derived)

    def test_phrase_with_two_preference_words_changes_both(self) -> None:
        stock = {"了解计划": "瞭解計劃"}
        stock.update({key: value for key, value in zh_traditional._base()[0].items() if key in ("了解", "计划")})
        self.assertEqual(zh_traditional.derived_preferences(stock), {"了解计划": "了解計畫"})

    def test_empty_preferences_derive_nothing_and_equal_stock(self) -> None:
        import opencc
        stock = zh_traditional._base()[0]
        with mock.patch.object(zh_traditional, "PREFERENCES", ()):
            self.assertEqual(zh_traditional.derived_preferences(stock), {})
            _rebuild()
            with mock.patch.multiple(zh_traditional, CORRECTIONS=(), DROPPED_PHRASES=frozenset(), CONTEXT_RULES=(), VARIANT_PREFERENCE={}, WHOLE_LINES=()):
                _rebuild()
                cc = opencc.OpenCC("s2tw")
                for text, _ in INSIDE + BEFORE + EXPLICIT + MIXED:
                    with self.subTest(text=text):
                        self.assertEqual(zh_traditional.convert_text(text), cc.convert(text))


if __name__ == "__main__":
    unittest.main()
