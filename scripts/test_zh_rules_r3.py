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
    (("一直发现", "一直发光", "一直发生", "一直发展", "一直发出", "一直发愁", "一直发呆", "一直发抖", "一直发烧", "一直发笑", "一直发亮", "一直发怒", "一直发火", "一直发狂", "一直发挥", "一直发布", "一直发表", "一直发誓"),
     [("他一直发现新东西", "他一直發現新東西"), ("灯一直发光", "燈一直發光")],
     [("她梳着一直发", "她梳著一直髮"), ("他留着一直发垂到肩", "他留著一直髮垂到肩")]),
    (("一头发光", "一头发亮"),
     [("一头发光的小鹿", "一頭發光的小鹿"), ("一头发亮的黑毛", "一頭發亮的黑毛")],
     [("他有一头发白的长发", "他有一頭髮白的長髮"), ("一头发丝飘起来", "一頭髮絲飄起來")]),
    (("风后来", "风后面"),
     [("风后来停了", "風後來停了"), ("风后面跟着雨", "風後面跟著雨")],
     [("黄帝的大臣风后", "黃帝的大臣風后"), ("风后是一个人名", "風后是一個人名")]),
    (("回归线",),
     [("太阳直射回归线", "太陽直射回歸線"), ("回归线以南", "回歸線以南")],
     [("线性回归模型", "線性迴歸模型"), ("回归分析", "迴歸分析")]),
    (("今天干了啥", "昨天干了啥", "明天干了啥"),
     [("你今天干了啥", "你今天幹了啥"), ("昨天干了啥都忘了", "昨天幹了啥都忘了")],
     [("吃完饼干了啥都不想做", "吃完餅乾了啥都不想做"), ("池塘的水干了", "池塘的水乾了")]),
    (("新书签名", "图书签名"),
     [("新书签名会", "新書簽名會"), ("图书签名活动", "圖書簽名活動")],
     [("在书签名称栏里", "在書籤名稱欄裡"), ("夹着书签在里面", "夾著書籤在裡面")]),
    (("短短几",),
     [("短短几秒", "短短幾秒"), ("短短几天", "短短幾天")],
     [("茶几", "茶几"), ("一张红木短几", "一張紅木短几")]),
    (("一起干杯",),
     [("大家一起干杯", "大家一起乾杯"), ("我们一起干杯吧", "我們一起乾杯吧")],
     [("一起干活", "一起幹活"), ("他们一起干了件大事", "他們一起幹了件大事")]),
    (("没干透", "没干净", "没干燥"),
     [("衣服还没干透", "衣服還沒乾透"), ("手没干净", "手沒乾淨")],
     [("他没干坏事", "他沒幹壞事"), ("他没干完作业", "他沒幹完作業")]),
    (("这杯干了", "酒我干了"),
     [("把这杯干了吧", "把這杯乾了吧"), ("这杯酒我干了", "這杯酒我乾了")],
     [("他把这件事干了吧", "他把這件事幹了吧"), ("他干了一件大事", "他幹了一件大事")]),
    (("熬干了", "嘴巴干了"),
     [("汤熬干了", "湯熬乾了"), ("嘴巴干了想喝水", "嘴巴乾了想喝水")],
     [("他熬夜干了活", "他熬夜幹了活"), ("他张大嘴巴说干了就干", "他張大嘴巴說幹了就幹")]),
    (("在海里", "大海里", "到海里", "入海里"),
     [("鱼在海里生活", "魚在海裡生活"), ("大海里有鲸", "大海裡有鯨")],
     [("船行驶了三海里", "船行駛了三海里"), ("离岸二十海里的岛", "離岸二十海里的島")]),
    (("海里有各种", "星海里", "海里的鱼"),
     [("海里有各种鱼", "海裡有各種魚"), ("星海里的光", "星海裡的光")],
     [("十二海里有一座岛", "十二海里有一座島"), ("每海里收费十元", "每海里收費十元")]),
    (("有云层", "有云雾", "有云彩", "有云朵", "有云纹", "有云形", "有云状", "有云雷", "有云雀", "天上有云", "天空有云", "今天有云", "明天有云"),
     [("今天有云层", "今天有雲層"), ("山间有云雾", "山間有雲霧")],
     [("古人有云", "古人有云"), ("孔子有云", "孔子有云")]),
    (tuple("被发" + x for x in "送生布表展动明挥出回下给到往掘配育票放射起言达扬来过去走了"),
     [("文件被发送了", "文件被發送了"), ("他被发了一张好人卡", "他被發了一張好人卡")],
     [("被发乱首而行", "被髮亂首而行"), ("被发而为狂", "被髮而為狂")]),
    (("游标卡尺", "兼容并包"),
     [("游标卡尺量一下", "游標卡尺量一下"), ("兼容并包的态度", "兼容並包的態度")],
     [("游标是一个箭头", "游標是一個箭頭"), ("并包含所有费用", "並包含所有費用")]),
    (("统计划分",),
     [("统计划分成两组", "統計劃分成兩組"), ("按统计划分区域", "按統計劃分區域")],
     [("我的计划", "我的計畫"), ("计划赶不上变化", "計畫趕不上變化")]),
    (("不是只有", "若是只有", "总是只有", "总是只在", "还是只想", "没有只要", "别只是"),
     [("总是只有我", "總是只有我"), ("别只是旁观", "別只是旁觀")],
     [("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("那是只会飞的蝴蝶", "那是隻會飛的蝴蝶")]),
    (("卷卷而上",),
     [("卷卷而上的火焰", "捲捲而上的火焰"), ("烟卷卷而上", "煙捲捲而上")],
     [("问卷卷成了筒状", "問卷捲成了筒狀"), ("案卷卷起了边角", "案卷捲起了邊角")]),
    (("湿润的风后",),
     [("吸入湿润的风后", "吸入濕潤的風後"), ("闻到湿润的风后的泥土味", "聞到濕潤的風後的泥土味")],
     [("黄帝的大臣风后", "黃帝的大臣風后"), ("他叫风后", "他叫風后")]),
    (("古松开",),
     [("古松开始变黄", "古松開始變黃"), ("古松开出新叶", "古松開出新葉")],
     [("鞋带又松开了", "鞋帶又鬆開了"), ("老松开我的手", "老鬆開我的手")]),
    (("tour",),
     [("旅游动态更新", "旅遊動態更新"), ("漫游过去的事", "漫遊過去的事"), ("畅游动物园", "暢遊動物園")],
     [("鱼游动着", "魚游動著"), ("他游过了河", "他游過了河")]),
    (("surname-yu",),
     [("余先生在家", "余先生在家"), ("我找余老师", "我找余老師")],
     [("剩余先生产", "剩餘先生產"), ("多余老师的话", "多餘老師的話")]),
    (("dry-throat",),
     [("如果喉咙干了，就去喝水", "如果喉嚨乾了，就去喝水"), ("喉咙干了就喝水", "喉嚨乾了就喝水")],
     [("捂住喉咙干了坏事", "捂住喉嚨幹了壞事"), ("他喉咙干了三杯", "他喉嚨幹了三杯")]),
    (("toast-start",),
     [("干了就没事了", "乾了就沒事了"), ("干了就没事！", "乾了就沒事！")],
     [("这活儿干了就没事了", "這活兒幹了就沒事了"), ("他干了就没事", "他幹了就沒事")]),
)
FAMILY_STOCK = ("了如列眉", "周几", "游标", "控制台", "老么", "么儿", "么妹", "么弟", "么女", "面向对象的技术", "面向对象语言", "毁炎", "毁犀", "四库禁毁书丛刋")   # tested one by one in test_zh_convert (sample + negative)


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
