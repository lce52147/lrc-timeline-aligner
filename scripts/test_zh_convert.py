"""zh_traditional.convert_text: stock OpenCC s2tw plus four kinds of rules, each with a reason and a positive and a negative example."""
from __future__ import annotations

import random
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


class RulesPatched(unittest.TestCase):
    """Base class: ``self.rules(CORRECTIONS=(), ...)`` swaps rule tables for the test and puts them back afterwards."""

    def rules(self, **tables) -> None:
        for name, value in tables.items():
            patcher = mock.patch.object(zh_traditional, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        _rebuild()
        self.addCleanup(_rebuild)


class StockEqualityTests(RulesPatched):
    """With the four rule tables empty the converter is stock OpenCC s2tw, character for character (gate G1)."""

    def setUp(self) -> None:
        import opencc
        self.stock = opencc.OpenCC("s2tw")
        self.rules(CORRECTIONS=(), PREFERENCES=(), DROPPED_PHRASES=frozenset(), CONTEXT_RULES=(), VARIANT_PREFERENCE={})

    def test_every_stphrases_word(self) -> None:
        phrases = list(zh_traditional._base()[0])
        self.assertGreater(len(phrases), 49000)
        different = [key for key in phrases if zh_traditional.convert_text(key) != self.stock.convert(key)]
        self.assertEqual(different[:10], [])

    def test_fixed_random_sentences(self) -> None:
        base = zh_traditional._base()
        words, chars = sorted(base[0]), sorted(base[1])
        generator = random.Random(20261116)
        for _ in range(300):
            parts = [generator.choice(words) for _ in range(generator.randint(1, 3))] + [generator.choice(chars) for _ in range(2)]
            generator.shuffle(parts)
            text = "".join(piece + generator.choice(["", "", "，", " ", "a"]) for piece in parts)
            self.assertEqual(zh_traditional.convert_text(text), self.stock.convert(text), text)

    def test_the_known_stock_mistakes_are_back(self) -> None:
        self.assertEqual(zh_traditional.convert_text("一起干杯"), "一起幹杯")
        self.assertEqual(zh_traditional.convert_text("关系紧密"), "關係緊密")
        self.assertEqual(zh_traditional.convert_text("他干了一件大事"), "他幹了一件大事")


class EngineTests(unittest.TestCase):
    def test_the_longest_phrase_wins_then_the_leftmost(self) -> None:
        phrases = {"ab": 1, "bc": 1, "abc": 1, "cd": 1}
        self.assertEqual(zh_traditional._segment("xabcd", phrases, 3), [(1, 4)])
        self.assertEqual(zh_traditional._segment("abcd", {"ab": 1, "bc": 1, "cd": 1}, 2), [(0, 2), (2, 4)])   # equal length: the left one
        self.assertEqual(zh_traditional._segment("abxbc", {"ab": 1, "bc": 1}, 2), [(0, 2), (3, 5)])

    def test_the_left_parse_wins_a_tie(self) -> None:
        self.assertEqual(zh_traditional.convert_text("头发现在"), "頭髮現在")
        self.assertEqual(zh_traditional.convert_text("理发现场"), "理髮現場")
        self.assertEqual(zh_traditional.convert_text("太后来过"), "太后來過")
        self.assertEqual(zh_traditional.convert_text("天后宫"), "天后宮")
        self.assertEqual(zh_traditional.convert_text("吃面前先洗手"), "吃麵前先洗手")

    def test_a_separator_ends_every_phrase_like_stock(self) -> None:
        # stock splits at punctuation and spaces before it looks for phrases
        self.assertEqual(zh_traditional.convert_text("一起，干杯"), "一起，乾杯")
        self.assertEqual(zh_traditional.convert_text("头 发"), "頭 發")

    def test_required_words(self) -> None:
        expected = {
            "一起干杯": "一起乾杯", "一起干活": "一起幹活", "干杯": "乾杯", "头发": "頭髮", "发现": "發現", "后来": "後來",
            "皇后": "皇后", "里面": "裡面", "公里": "公里", "面条": "麵條", "一只": "一隻", "只有": "只有", "台风": "颱風",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_fixed_phrases_of_the_review(self) -> None:
        expected = {
            "注册": "註冊", "注解": "註解", "注释": "註釋", "注明": "註明", "注销": "註銷", "备注": "備註", "附注": "附註", "脚注": "腳註",
            "注生娘娘": "註生娘娘", "注定": "注定", "命中注定": "命中注定", "注意": "注意", "向导": "嚮導", "心向往之": "心嚮往之", "向往": "嚮往",
            "方向": "方向", "关系紧密": "關係緊密", "把这杯干了吧": "把這杯乾了吧", "这杯酒我干了": "這杯酒我乾了",
            "他把这件事干了吧": "他把這件事幹了吧", "他干了一件大事": "他幹了一件大事", "关注明天": "關注明天", "迈向明天": "邁向明天",
            "一起干杯": "一起乾杯", "干了这杯酒": "乾了這杯酒", "如果喉咙干了": "如果喉嚨乾了", "眼泪干了": "眼淚乾了", "总是只有一只鸟": "總是只有一隻鳥",
            "一直发现着": "一直發現著", "一头发光": "一頭髮光", "苦干了一辈子": "苦幹了一輩子", "干了什么": "幹了什麼",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_unchanged_and_unrelated_text(self) -> None:
        self.assertEqual(zh_traditional.convert_text("abc 123，。"), "abc 123，。")
        self.assertEqual(zh_traditional.convert_text(""), "")
        self.assertIsNone(zh_traditional.convert_line("這是第一句"))


class RuleTableTests(RulesPatched):
    """Every rule has a reason, a positive and a negative example, and changes an output (gates for the rule tables)."""

    def test_entries_have_a_reason_and_their_examples_hold(self) -> None:
        for entry in zh_traditional.CORRECTIONS + zh_traditional.PREFERENCES:
            self.assertTrue(entry.why.strip(), entry.key)
            self.assertGreaterEqual(len(entry.key), 2, entry.key)
            self.assertEqual(zh_traditional.convert_text(entry.key), entry.value, "positive: " + entry.key)
            sample, want = entry.sample or (entry.key, entry.value)
            self.assertEqual(zh_traditional.convert_text(sample), want, "sample: " + entry.key)
            negative, want = entry.negative
            self.assertEqual(zh_traditional.convert_text(negative), want, "negative: " + entry.key)
        keys = [entry.key for entry in zh_traditional.CORRECTIONS + zh_traditional.PREFERENCES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_an_entry_that_changes_nothing_does_not_belong_in_the_table(self) -> None:
        every = zh_traditional.CORRECTIONS + zh_traditional.PREFERENCES
        for entry in every:
            sample = (entry.sample or (entry.key, entry.value))[0]
            before = zh_traditional.convert_text(sample)
            others = [item for item in every if item is not entry]
            with mock.patch.object(zh_traditional, "CORRECTIONS", tuple(item for item in others if item in zh_traditional.CORRECTIONS)), \
                    mock.patch.object(zh_traditional, "PREFERENCES", tuple(item for item in others if item in zh_traditional.PREFERENCES)):
                _rebuild()
                after = zh_traditional.convert_text(sample)
            _rebuild()
            self.assertNotEqual(before, after, "the entry changes nothing: " + entry.key)

    def test_context_rules_have_a_reason_and_their_examples_hold(self) -> None:
        names = [rule.name for rule in zh_traditional.CONTEXT_RULES]
        self.assertEqual(len(names), len(set(names)))
        for rule in zh_traditional.CONTEXT_RULES:
            self.assertTrue(rule.why.strip(), rule.name)
            self.assertTrue(rule.positive and rule.negative, rule.name)
            for text, want in rule.positive + rule.negative:
                self.assertEqual(zh_traditional.convert_text(text), want, "%s: %s" % (rule.name, text))

    def test_a_context_rule_changes_the_result_of_one_of_its_positive_examples(self) -> None:
        every = zh_traditional.CONTEXT_RULES
        for rule in every:
            before = [zh_traditional.convert_text(text) for text, _want in rule.positive]
            with mock.patch.object(zh_traditional, "CONTEXT_RULES", tuple(item for item in every if item is not rule)):
                _rebuild()
                after = [zh_traditional.convert_text(text) for text, _want in rule.positive]
            _rebuild()
            self.assertNotEqual(before, after, "the rule changes nothing: " + rule.name)

    def test_dropped_phrases_have_a_positive_and_a_negative_example(self) -> None:
        self.assertEqual(set(zh_traditional.DROPPED_NOTES), set(zh_traditional.DROPPED_PHRASES))
        for key in zh_traditional.DROPPED_PHRASES:
            positive, negative = zh_traditional.DROPPED_NOTES[key]
            self.assertEqual(zh_traditional.convert_text(positive[0]), positive[1], key)
            self.assertEqual(zh_traditional.convert_text(negative[0]), negative[1], key)
            with mock.patch.object(zh_traditional, "DROPPED_PHRASES", zh_traditional.DROPPED_PHRASES - {key}):
                _rebuild()
                kept = zh_traditional.convert_text(positive[0])
            _rebuild()
            self.assertNotEqual(kept, positive[1], "dropping the phrase changes nothing: " + key)

    def test_the_variant_preference_holds_only_characters_that_are_variants_in_every_word(self) -> None:
        self.assertEqual(set(zh_traditional.VARIANT_PREFERENCE), set("臺溼汙燬絃悽歎唸"))
        expected = {"舞台上的灯": "舞台上的燈", "淋湿": "淋濕", "怀念": "懷念", "琴弦": "琴弦", "叹息": "嘆息", "烧毁": "燒毀", "里面": "裡面"}
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_the_preference_does_not_touch_traditional_text_that_was_already_there(self) -> None:
        self.assertEqual(zh_traditional.convert_text("臺北的灯"), "臺北的燈")

    def test_the_preference_does_not_touch_an_entry_or_a_context_span(self) -> None:
        self.assertEqual(zh_traditional.convert_text("这杯酒我干了"), "這杯酒我乾了")
        self.assertEqual(zh_traditional.convert_text("毁炎"), "毀炎")

    def test_preferences_are_kept_apart_from_corrections(self) -> None:
        corrected = {entry.key for entry in zh_traditional.CORRECTIONS}
        preferred = {entry.key for entry in zh_traditional.PREFERENCES}
        self.assertFalse(corrected & preferred)
        for key in ("计划", "了解", "游动着", "回归", "注定", "克制", "糊里糊涂", "女生外向"):
            self.assertIn(key, preferred)
        for key in ("了如列眉", "短短几", "一直发现着"):
            self.assertIn(key, corrected)

    def test_users_preferences(self) -> None:
        expected = {"计划": "計畫", "了解": "了解", "回归": "回歸", "游动着": "游動著", "游过群星之间": "游過群星之間", "点赞": "點讚", "尽可能": "盡可能", "背负": "背負", "周而复始": "周而復始"}
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)


def _source_index(text: str, after: str, position: int) -> int:
    """Both texts here have the same length (every rule and phrase maps one character to one)."""
    assert len(text) == len(after)
    return position


class OneToManyTests(unittest.TestCase):
    def test_marks_list_the_alternatives(self) -> None:
        after, marks = zh_traditional.convert_marked("干杯")
        self.assertEqual(after, "乾杯")
        self.assertEqual([(m["at"], m["from"]) for m in marks][0], (0, "干"))
        self.assertEqual(set(marks[0]["options"]), {"幹", "乾", "干"})
        self.assertEqual(zh_traditional.convert_marked("爱你")[1], [])
        self.assertIn("发", zh_traditional.one_to_many())
        self.assertNotIn("爱", zh_traditional.one_to_many())

    def test_marks_stay_correct_inside_a_context_span_and_after_a_separator(self) -> None:
        for text, want in {"把这杯干了吧，头发": "把這杯乾了吧，頭髮", "总是只有我": "總是只有我", "他把这件事干了吧，发现": "他把這件事幹了吧，發現"}.items():
            after, marks = zh_traditional.convert_marked(text)
            self.assertEqual(after, want)
            self.assertTrue(marks)
            for mark in marks:
                self.assertEqual(text[_source_index(text, after, mark["at"])], mark["from"])
                self.assertIn(after[mark["at"]], mark["options"])
        after, marks = zh_traditional.convert_marked("把这杯干了吧，头发")
        self.assertEqual({(m["at"], m["from"]) for m in marks if m["from"] in "干发"}, {(3, "干"), (8, "发")})

    def test_without_the_dictionaries_nothing_is_marked(self) -> None:
        zh_traditional.one_to_many.cache_clear()
        self.addCleanup(zh_traditional.one_to_many.cache_clear)
        with mock.patch.object(zh_traditional, "_dicts", False):
            self.assertIsNone(zh_traditional.convert_marked("干杯"))
            self.assertEqual(zh_traditional.one_to_many(), {})


class RuleExamplesTests(unittest.TestCase):
    """The task's named cases (G5, G6), run from a table."""

    TABLE = {
        # simplified -> Traditional
        "关系紧密": "關係緊密", "把这杯干了吧": "把這杯乾了吧", "这杯酒我干了": "這杯酒我乾了",
        "他把这件事干了吧": "他把這件事幹了吧", "他干了一件大事": "他幹了一件大事",
        "熬干了汤": "熬幹了湯", "喉咙干了，想喝水": "喉嚨幹了，想喝水", "干了就没事了": "幹了就沒事了", "这活儿干了就没事了": "這活兒幹了就沒事了",
        "你今天干了啥": "你今天干了啥", "吃完饼干了啥都不想做": "吃完餅乾了啥都不想做", "风后来停了": "風后來停了", "黄帝的大臣风后": "黃帝的大臣風后", "吸入湿润的风后": "吸入濕潤的風後",
        "离岸几海里": "離岸幾海里", "航行了二十海里": "航行了二十海里", "他在海里游泳": "他在海里游泳", "海里有各种鱼": "海里有各種魚", "每海里收费十元": "每海里收費十元",
        "试卷卷面整洁": "試卷卷面整潔", "卷卷而上的火焰": "捲捲而上的火焰", "志向往往会改变": "志向往往會改變", "朝南的方向往北走": "朝南的方向往北走",
        "旅游动态更新": "旅遊動態更新", "他游过了河": "他遊過了河",
        "鱼游动着": "魚游動著", "线性回归模型": "線性迴歸模型", "回归线穿过这里": "迴歸線穿過這裡", "香港回归": "香港回歸",
        "他在新书签名会上": "他在新書簽名會上", "夹着书签在里面": "夾著書籤在裡面", "青松开花了": "青松開花了", "老松开我的手": "老鬆開我的手",
        "鞋带又松开了": "鞋帶又鬆開了", "理发现场很热闹": "理髮現場很熱鬧", "舞台灯光": "舞台燈光", "这只会飞的鸟": "這隻會飛的鳥", "这只会唱歌又会跳舞的鸟": "這隻會唱歌又會跳舞的鳥",
        "这只会更糟": "這隻會更糟", "那只有毒的蛇": "那只有毒的蛇", "这只猫": "這隻貓", "总是只有我": "總是只有我", "没有只要祈求": "沒有只要祈求", "克制自己的情绪": "克制自己的情緒",
        "刮胡子": "刮鬍子", "刮了一阵风": "颳了一陣風", "余先生在家": "余先生在家", "老么": "老麼", "什么": "什麼",
    }

    def test_table(self) -> None:
        for text, want in self.TABLE.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_the_fixed_word_list_of_the_review(self) -> None:
        expected = {
            "注册": "註冊", "注解": "註解", "注释": "註釋", "注明": "註明", "注销": "註銷", "备注": "備註", "附注": "附註", "批注": "批註",
            "标注": "標註", "脚注": "腳註", "注生娘娘": "註生娘娘", "注定": "注定", "命中注定": "命中注定", "注意": "注意", "专注": "專注",
            "向导": "嚮導", "心向往之": "心嚮往之", "向往": "嚮往", "方向": "方向", "女生外向": "女生外向", "面向对象语言": "面嚮對象語言",
            "命运注定": "命運注定", "关注明天": "關注明天", "迈向明天": "邁向明天", "方向导航": "方向導航", "干了这杯酒": "乾了這杯酒",
            "如果喉咙干了": "如果喉嚨乾了", "一起干杯": "一起乾杯",
        }
        for text, want in expected.items():
            self.assertEqual(zh_traditional.convert_text(text), want, text)

    def test_every_dictionary_word_that_produces_zhu_or_xiang_is_either_converted_or_listed(self) -> None:
        raw = zh_traditional._base()[0]
        kept = {key for key, value in raw.items() if ("註" in value and "註" not in zh_traditional.convert_text(key))
                or ("嚮" in value and "嚮" not in zh_traditional.convert_text(key))}
        self.assertEqual(kept, {"命中注定", "命运注定", "注定", "五百姻缘天注定", "女生外向", })


if __name__ == "__main__":
    unittest.main()
