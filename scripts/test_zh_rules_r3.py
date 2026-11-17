"""Every non-preference ENTRIES word and every CONTEXT_RULES rule of zh_traditional: two sentences where it fires and two look-alikes
(a different word with the same ending or beginning) where it must not. Rows: (keys, [(simplified, expected)] x2 positive, [(simplified, expected)] x2 negative).
A rule that gets hit by a counter example is deleted, not patched (the review rule of BH2-FIX11A)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import zh_traditional

GROUPS = (
    (("一直发现着", "一直发现著"),
     [("他一直发现着新东西", "他一直發現著新東西"), ("灯一直发现著光", "燈一直發現著光")],
     [("她梳着一直发", "她梳著一直髮"), ("她梳着一直发现在很流行", "她梳著一直髮現在很流行")]),
    (("短短几",),
     [("短短几秒", "短短幾秒"), ("短短几天", "短短幾天")],
     [("茶几", "茶几"), ("一张红木短几", "一張紅木短几")]),
    (("卷卷而上",),
     [("卷卷而上的火焰", "捲捲而上的火焰"), ("烟卷卷而上", "煙捲捲而上")],
     [("问卷卷成了筒状", "問卷捲成了筒狀"), ("案卷卷起了边角", "案卷捲起了邊角")]),
    (("没干透吧",),
     [("衣服还没干透吧", "衣服還沒乾透吧"), ("道路还没干透吧", "道路還沒乾透吧")],
     [("他没干透露机密的事", "他沒幹透露機密的事"), ("他没干坏事", "他沒幹壞事")]),
    (("湿润的风后",),
     [("吸入湿润的风后", "吸入濕潤的風後"), ("闻到湿润的风后的泥土味", "聞到濕潤的風後的泥土味")],
     [("黄帝的大臣风后", "黃帝的大臣風后"), ("他叫风后", "他叫風后")]),
    (("还是只想", "总是只有", "若是只有", "总是只在", "没有只要", "别只是", "不是只有一次"),
     [("总是只有我", "總是只有我"), ("别只是旁观", "別只是旁觀")],
     [("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("那是只会飞的蝴蝶", "那是隻會飛的蝴蝶")]),
    (("闪烁的星海里",),
     [("漂浮在闪烁的星海里", "漂浮在閃爍的星海裡"), ("闪烁的星海里有光", "閃爍的星海裡有光")],
     [("十二海里有一座岛", "十二海里有一座島"), ("每海里收费十元", "每海里收費十元")]),
    (("一起干杯",),
     [("大家一起干杯", "大家一起乾杯"), ("我们一起干杯吧", "我們一起乾杯吧")],
     [("一起干活", "一起幹活"), ("他们一起干了件大事", "他們一起幹了件大事")]),
    (("这杯干了", "这杯酒我干了"),
     [("把这杯干了吧", "把這杯乾了吧"), ("这杯酒我干了", "這杯酒我乾了")],
     [("他把这件事干了吧", "他把這件事幹了吧"), ("卖酒我干了十年", "賣酒我幹了十年")]),
    (("如果喉咙干了",),
     [("如果喉咙干了，就去喝水", "如果喉嚨乾了，就去喝水"), ("他说如果喉咙干了就喝水", "他說如果喉嚨乾了就喝水")],
     [("捂着喉咙干了坏事", "捂著喉嚨幹了壞事"), ("他喉咙干了三杯", "他喉嚨幹了三杯")]),
    (("了如列眉",),
     [("一目了如列眉", "一目瞭如列眉"), ("事情明了如列眉", "事情明瞭如列眉")],
     [("我明白了如何做", "我明白了如何做"), ("看了如下内容", "看了如下內容")]),
    (("唯有云雀的声音",),
     [("唯有云雀的声音传来", "唯有雲雀的聲音傳來"), ("耳边唯有云雀的声音", "耳邊唯有雲雀的聲音")],
     [("古人有云雀跃不已", "古人有云雀躍不已"), ("古人有云", "古人有云")]),
    (tuple("被发" + x for x in "送生布表"),
     [("文件被发送了", "文件被發送了"), ("他被发表了一篇文章", "他被發表了一篇文章")],
     [("被发乱首而行", "被髮亂首而行"), ("被发而为狂", "被髮而為狂")]),
    (("四库禁毁书丛刋",),
     [("四库禁毁书丛刋", "四庫禁燬書叢刋"), ("这套四库禁毁书丛刋", "這套四庫禁燬書叢刋")],
     [("我的计划", "我的計畫"), ("烧毁了文件", "燒毀了文件"), ("统计划分成两组", "統計劃分成兩組")]),
    (("surname-yu",),
     [("余先生在家", "余先生在家"), ("我找余老师", "我找余老師")],
     [("剩余先生产", "剩餘先生產"), ("多余老师的话", "多餘老師的話")]),
)
FAMILY_STOCK = ()   # every non-preference word has its own group above and its 6 edge attacks in test_zh_edges_r4


class RuleContrastTests(unittest.TestCase):
    def test_every_group_row(self) -> None:
        for keys, positives, negatives in GROUPS:
            self.assertGreaterEqual(len(positives), 2, keys)
            self.assertGreaterEqual(len(negatives), 2, keys)
            for text, want in positives + negatives:
                self.assertEqual(zh_traditional.convert_text(text), want, "%s: %s" % (keys[0], text))

    def test_every_non_preference_word_and_rule_is_covered(self) -> None:
        covered = {key for keys, _p, _n in GROUPS for key in keys}
        for entry in zh_traditional.CORRECTIONS:
            self.assertTrue(entry.key in covered or entry.key in FAMILY_STOCK, entry.key)
        for rule in zh_traditional.CONTEXT_RULES:
            self.assertIn(rule.name, covered)

    def test_the_positives_are_not_what_stock_gives_unless_the_rule_keeps_stock(self) -> None:
        import opencc
        stock = opencc.OpenCC("s2tw")
        changed = 0
        for keys, positives, _negatives in GROUPS:
            changed += sum(1 for text, want in positives if stock.convert(text) != want)
        self.assertGreater(changed, 25)

    def test_the_three_review_sets_have_no_regression_against_stock(self) -> None:
        import opencc
        stock = opencc.OpenCC("s2tw")
        for keys, _positives, negatives in GROUPS:
            for text, want in negatives:
                if stock.convert(text) == want:
                    self.assertEqual(zh_traditional.convert_text(text), want, text)


if __name__ == "__main__":
    unittest.main()
