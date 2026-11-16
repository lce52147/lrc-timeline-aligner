"""zh_traditional.song_candidates in songs with kana: every line s2tw changes is listed; the checked ones are the lines that are surely Chinese."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import zh_traditional


def _rows(*entries):
    """Each argument is the list of lines of one row (one timestamp)."""
    return [{"entry": index + 1, "lyrics": list(lines)} for index, lines in enumerate(entries)]


def _listed(*entries, uncertain=False):
    info = zh_traditional.song_candidates(_rows(*entries), (), uncertain)
    return info["count"], info["optional"], [(item["entry"], item["sub"], item["before"], item["default_checked"]) for item in info["lines"]]


class CertainSimplifiedMaTests(unittest.TestCase):
    def test_ma_after_the_question_words_is_certainly_simplified(self) -> None:
        for text in ("怎么了", "什么", "这么多", "那么", "多么", "要么", "甚么"):
            self.assertTrue(zh_traditional.looks_simplified(text), text)
            self.assertTrue(zh_traditional.looks_simplified(text, japanese=True), text)

    def test_the_traditional_words_with_ma_are_not(self) -> None:
        for text in ("老么", "么儿", "么妹", "么弟", "么女", "老么回來了", "怎麼了", "什麼"):
            self.assertFalse(zh_traditional.looks_simplified(text), text)
            self.assertFalse(zh_traditional.looks_simplified(text, japanese=True), text)
        for text, want in {"老么": "老么", "么儿": "么兒", "么妹": "么妹", "什么": "什麼", "怎么了": "怎麼了"}.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_a_song_without_kana_opens_the_gate_with_ma_alone(self) -> None:
        self.assertEqual(_listed(["怎么了"], ["与你同行"]), (2, 0, [(1, 0, "怎么了", True), (2, 0, "与你同行", True)]))


class KanaSongTests(unittest.TestCase):
    def test_a_simplified_translation_with_only_ma_is_listed_and_checked(self) -> None:
        count, optional, lines = _listed(["さよなら", "怎么了"], ["きみのこえ"])
        self.assertEqual((count, optional, lines), (1, 0, [(1, 1, "怎么了", True)]))

    def test_the_gate_still_needs_one_certain_line(self) -> None:
        self.assertEqual(_listed(["きみ", "回来"], ["初恋"], ["一期一会"]), (0, 0, []))
        self.assertEqual(_listed(["きみ", "回来"], ["初恋"], ["一期一会"], ["老么"]), (0, 0, []))

    def test_two_chinese_function_words_check_the_line(self) -> None:
        count, optional, lines = _listed(["きみ"], ["我们一起走"], ["这个还对"], ["不在后面"])
        self.assertEqual(lines, [(2, 0, "我们一起走", True), (3, 0, "这个还对", True), (4, 0, "不在后面", False)])
        self.assertEqual((count, optional), (2, 1))

    def test_words_japanese_also_uses_do_not_check_a_line(self) -> None:
        for text in ("他国", "目的", "不安", "存在", "有名", "我是", "他在有", "回来", "着物", "没収"):
            self.assertEqual(zh_traditional._chinese_function_words(text), 0, text)
        count, optional, lines = _listed(["きみ", "他国"], ["きみ", "目的"], ["きみ", "不安"], ["きみ", "存在"], ["きみ", "有名"], ["这么说"])
        self.assertIn((1, 1, "他国", False), lines)   # only 他国 changes under s2tw; 目的 不安 存在 有名 are not candidates at all
        self.assertEqual({item[2] for item in lines}, {"他国", "这么说"})
        self.assertTrue(all(not item[3] for item in lines if item[2] in ("他国", "目的", "不安", "存在", "有名")))

    def test_a_function_word_next_to_a_kana_line_of_the_same_row_checks_the_line(self) -> None:
        count, optional, lines = _listed(["さよなら", "我不会走"], ["きみ", "我后面"], ["我后面"], ["我们一起走"])
        self.assertEqual(lines, [(1, 1, "我不会走", False), (2, 1, "我后面", False), (3, 0, "我后面", False), (4, 0, "我们一起走", True)])
        self.assertEqual((count, optional), (1, 3))
        count, optional, lines = _listed(["さよなら", "我不会说"], ["きみ", "这还会走"], ["这还会走"])
        self.assertEqual(lines, [(1, 1, "我不会说", True), (2, 1, "这还会走", True), (3, 0, "这还会走", True)])

    def test_a_kanji_only_line_is_listed_unchecked(self) -> None:
        count, optional, lines = _listed(["きみ", "回来"], ["初恋"], ["一期一会"], ["我们一起走"])
        self.assertEqual(lines, [(1, 1, "回来", False), (2, 0, "初恋", False), (3, 0, "一期一会", False), (4, 0, "我们一起走", True)])
        self.assertEqual((count, optional), (1, 3))

    def test_a_line_with_a_character_cp932_cannot_encode_is_checked(self) -> None:
        self.assertTrue(zh_traditional._surely_not_japanese("仿"))
        self.assertFalse(zh_traditional._surely_not_japanese("国境"))
        self.assertFalse(zh_traditional._surely_not_japanese("初恋"))
        count, optional, lines = _listed(["きみ"], ["我们一起走"], ["后来仿"])
        self.assertEqual(lines, [(2, 0, "我们一起走", True), (3, 0, "后来仿", True)])

    def test_a_traditional_only_character_keeps_the_line_unchecked(self) -> None:
        count, optional, lines = _listed(["きみ"], ["我们一起走"], ["我們不在后面"])
        self.assertEqual(lines, [(2, 0, "我们一起走", True), (3, 0, "我們不在后面", False)])

    def test_a_traditional_translation_has_nothing_to_convert(self) -> None:
        self.assertEqual(_listed(["さよなら", "我不會走"], ["きみ", "我们一起走"]), (1, 0, [(2, 1, "我们一起走", True)]))

    def test_an_uncertain_encoding_checks_nothing(self) -> None:
        count, optional, lines = _listed(["さよなら", "怎么了"], ["きみ", "我不会走"], uncertain=True)
        self.assertEqual((count, optional), (0, 2))
        self.assertEqual([item[3] for item in lines], [False, False])

    def test_a_line_with_kana_is_never_listed(self) -> None:
        self.assertEqual(_listed(["我不会走さよなら", "怎么了"]), (1, 0, [(1, 1, "怎么了", True)]))

    def test_headers_follow_the_same_rule(self) -> None:
        info = zh_traditional.song_candidates(_rows(["きみ", "怎么了"]), [(0, "[ti:回来]"), (1, "[ar:我不会走]")], False)
        self.assertEqual([(item["before"], item["default_checked"]) for item in info["headers"]], [("回来", False), ("我不会走", False)])

    def test_a_song_without_kana_is_unchanged(self) -> None:
        count, optional, lines = _listed(["你在哪里"], ["回来"], ["運氣在裏面"], ["与你同行"])
        self.assertEqual(lines, [(1, 0, "你在哪里", True), (2, 0, "回来", True), (3, 0, "運氣在裏面", False), (4, 0, "与你同行", True)])


if __name__ == "__main__":
    unittest.main()
