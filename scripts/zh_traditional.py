"""Simplified -> Traditional (Taiwan) detection for the review page.

OpenCC ``s2tw`` dictionaries only (never ``s2twp``: it rewrites vocabulary). A line with any kana is Japanese and is
never converted. Without OpenCC every function answers "nothing to convert", so the review never depends on it.

The conversion is stock OpenCC ``s2tw`` (opencc-python-reimplemented) run on our own tables: split at the separators,
then recursively "take the longest phrase of STPhrases, the leftmost one among equally long ones, and convert the two
sides the same way", the remaining characters through STCharacters, then TWVariants. With the four rule tables below
emptied the output is character for character the stock output (the test suite checks every STPhrases word).

Everything that differs from stock is one of four kinds of rule, each with a reason and a positive and a negative
example that the tests run (docs/ZH_S2TW_ROUNDTRIP.md lists them):

* ``ENTRIES``: a word added to or replaced in the dictionary; it takes part in the segmentation like any word, with no priority;
* ``DROPPED_PHRASES``: a stock phrase that is not used;
* ``CONTEXT_RULES``: a regular expression over the simplified text that fixes a span and its Traditional form where only the
  surrounding text can tell (a fixed span is a boundary: no phrase crosses it);
* ``VARIANT_PREFERENCE``: a character-level variant choice applied to the characters the conversion produced.
"""
from __future__ import annotations

import functools
import os
import re
from typing import Callable, NamedTuple, Union

# Hiragana, katakana (not the middle dot U+30FB or the long-vowel mark U+30FC: Chinese lyrics use both), katakana extensions and halfwidth katakana.
_KANA_RE = re.compile(r"[぀-ヺヽ-ヿㇰ-ㇿｦ-ﾟ]")
# ID tags whose text is prose: [ti:] [ar:] [al:] [au:] [by:] [re:] [ve:]. Only the text between the colon and the closing bracket is converted.
HEADER_TEXT_RE = re.compile(r"^(\[(?:ti|ar|al|au|by|re|ve):)(.*?)(\][ 	]*)$", re.I | re.S)
# CJK ideographs: extension A, unified, compatibility.
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TAIWAN_CHURCH_CHAR = "祢"   # U+7962, the standard pronoun in Taiwan church lyrics; Big5 has no code for it
# Characters that are simplified in every word they appear in even though Big5 can encode them: 么 is only ever written 麼 in
# Traditional text when it follows one of these (老么 / 么儿 / 么妹 are Traditional and do not match).
_CERTAIN_MA_RE = re.compile(r"[什怎这那多要甚]么")
_converter = None
_loaded = False


class Entry(NamedTuple):
    """One dictionary word. ``negative`` is a look-alike that must NOT be affected: ``(simplified, expected)``."""
    key: str
    value: str
    why: str
    negative: tuple
    sample: tuple = ()   # a sentence where the word changes the result: ``(simplified, expected)``; empty = the word itself


class Context(NamedTuple):
    """One context rule. ``pattern`` is searched in each clause of the simplified text; the span of group ``fix`` (the whole match
    without it) is fixed and written as ``value`` (a string, or a function of the match and of the rest of the clause)."""
    name: str
    pattern: str
    value: Union[str, Callable]
    why: str
    positive: tuple
    negative: tuple


# --------------------------------------------------------------------------------------------------------------------
# a. ENTRIES. Two kinds, kept apart: the stock dictionary is wrong (CORRECTIONS) or both forms are standard and the user's
# own lyrics use the one written here (PREFERENCES; "count in the lyrics: this form / the stock form").
# --------------------------------------------------------------------------------------------------------------------
CORRECTIONS: tuple = (
    Entry("糊里糊涂", "糊裡糊塗", "stock leaves 里: the Taiwan form of the idiom is 糊裡糊塗", ("糊里", "糊裡")),
    Entry("一直", "一直", "not a stock phrase: 直发 (直髮) swallows the 发 of 一直|发现 / 一直|发光", ("直发", "直髮"), ("一直发现", "一直發現")),
    Entry("一头", "一頭", "not a stock phrase: 头发 (頭髮) swallows the 发 of 一头|发光", ("长头发", "長頭髮"), ("一头发光", "一頭發光")),
    Entry("风后来", "風後來", "风后 is the name 風后 (kept); 风+后来 is wind + 'later'", ("风后", "風后")),
    Entry("风后面", "風後面", "风后 is the name 風后 (kept); 风+后面 is wind + 'behind'", ("风后", "風后")),
    Entry("回归线", "回歸線", "Taiwan writes 回歸線 (tropic); stock gives 迴歸線", ("线性回归", "線性迴歸")),
    Entry("克制", "克制", "Taiwan writes 克制 (restrain); stock gives 剋制", ("克星", "剋星")),
    Entry("干了啥", "幹了啥", "stock's phrase 天干 (heavenly stem) swallows the 干 of 今天|干了啥", ("池塘干了啥都不剩", "池塘乾了啥都不剩"), ("你今天干了啥", "你今天幹了啥")),
    Entry("书签名", "書簽名", "书签名 = signing a book (签名), not the bookmark 书签 of the preference below", ("书签", "書籤")),
    Entry("周几", "週幾", "week + how many ('which day of the week'): Taiwan writes 週幾; stock writes 周幾 (9 / 0 in the Taiwan corpus)", ("周末", "週末")),
    Entry("游标", "游標", "cursor / vernier: 游標; stock writes 遊標 (12 / 0 in the Taiwan corpus)", ("旅游", "旅遊")),
    Entry("控制台", "控制台", "console: Taiwan writes 控制台; stock's 控制檯 is the table-like 檯 (24 / 0 in the Taiwan corpus)", ("台灯", "檯燈")),
    Entry("老么", "老么", "么 is the Traditional character in 老么 / 么儿 / 么妹 / 么弟 / 么女 (youngest child); 么 after 什 怎 这 那 多 要 甚 is 麼", ("什么", "什麼")),
    Entry("么儿", "么兒", "youngest son: 么 is Traditional here", ("什么", "什麼")),
    Entry("么妹", "么妹", "youngest sister: 么 is Traditional here", ("什么", "什麼")),
    Entry("么弟", "么弟", "youngest brother: 么 is Traditional here", ("什么", "什麼")),
    Entry("么女", "么女", "youngest daughter: 么 is Traditional here", ("什么", "什麼")),
    Entry("女生外向", "女生外向", "Taiwan does not write 外嚮; stock turns 向 into 嚮 here", ("心向往之", "心嚮往之")),
    Entry("面向对象的技术", "面向對象的技術", "Taiwan does not write 面嚮; stock turns 向 into 嚮 here", ("心向往之", "心嚮往之")),
    Entry("面向对象语言", "面向對象語言", "Taiwan does not write 面嚮; stock turns 向 into 嚮 here", ("心向往之", "心嚮往之")),
    Entry("毁炎", "燬炎", "classical words keep 燬 (the character preference to 毀 does not apply)", ("烧毁", "燒毀")),
    Entry("毁犀", "燬犀", "classical words keep 燬 (the character preference to 毀 does not apply)", ("烧毁", "燒毀")),
    Entry("四库禁毁书丛刋", "四庫禁燬書叢刋", "a book series title written with 燬", ("烧毁", "燒毀")),
)
PREFERENCES: tuple = (
    Entry("计划", "計畫", "both are standard; 计划 -> 計畫 4 / 計劃 0 in the lyrics", ("合计划拨", "合計劃撥")),
    Entry("了解", "了解", "both are standard; 了解 4 / 瞭解 1", ("了如指掌", "瞭如指掌")),
    Entry("点赞", "點讚", "both are standard; 點讚 1 / 點贊 0", ("赞美", "讚美")),
    Entry("胡同", "胡同", "both are standard; 胡同 1 / 衚衕 0", ("胡子", "鬍子")),
    Entry("死胡同", "死胡同", "both are standard; 死胡同 1 / 死衚衕 0", ("胡子", "鬍子")),
    Entry("尽可能", "盡可能", "both are standard; 盡可能 1 / 儘可能 0", ("尽管", "儘管")),
    Entry("游动", "游動", "swimming, not touring: 游動 7 / 遊動 0", ("旅游动态", "旅遊動態")),
    Entry("游过", "游過", "swam across, not toured: 游過 / 遊過", ("旅游过程", "旅遊過程")),
    Entry("游移", "游移", "drifting, not touring: 游移 / 遊移", ("旅游移民", "旅遊移民")),
    Entry("背包", "背包", "both are standard; 背包 29 / 揹包 1", ("背叛", "背叛")),
    Entry("背负", "背負", "both are standard; 背負 / 揹負", ("背叛", "背叛")),
    Entry("回归", "回歸", "both are standard; 回歸 1 / 迴歸 0 in the lyrics", ("线性回归", "線性迴歸")),
    Entry("回归分析", "迴歸分析", "carve-out of 回归: statistics writes 迴歸", ("回归线", "回歸線")),
    Entry("线性回归", "線性迴歸", "carve-out of 回归: statistics writes 迴歸", ("回归线", "回歸線")),
    Entry("逻辑回归", "邏輯迴歸", "carve-out of 回归: statistics writes 迴歸", ("回归线", "回歸線")),
    Entry("回归模型", "迴歸模型", "carve-out of 回归: statistics writes 迴歸", ("回归线", "回歸線")),
    Entry("回归方程", "迴歸方程", "carve-out of 回归: statistics writes 迴歸", ("回归线", "回歸線")),
    Entry("熏染", "燻染", "both are standard; 燻染 1 / 薰染 0", ("熏陶", "薰陶")),
    Entry("周而复始", "周而復始", "both are standard; 周而復始 3 / 週而復始 0", ("周末", "週末")),
    Entry("注定", "注定", "both are standard; 注定 25 / 註定 0 in the lyrics (註冊 註解 備註 are 註 in Taiwan and stay stock)", ("注册", "註冊")),
    Entry("命中注定", "命中注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
    Entry("命运注定", "命運注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
    Entry("五百姻缘天注定", "五百姻緣天注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
)
# --------------------------------------------------------------------------------------------------------------------
# b. DROPPED_PHRASES: stock phrases that hide a better match across the same characters.
# --------------------------------------------------------------------------------------------------------------------
DROPPED_PHRASES = frozenset({"一起干", "了如", "短几", "有云", "如干", "心向", "并包", "并产", "被发", "先尝"})   # 一起干杯 is 一起+干杯 (stock: 一起干 hides 干杯); 了如 is only meant for 了如指掌, which stock has as a whole phrase
DROPPED_NOTES: dict = {
    "一起干": (("一起干杯", "一起乾杯"), ("一起干活", "一起幹活")),
    "了如": (("我明白了如何做", "我明白了如何做"), ("了如指掌", "瞭如指掌")),
    "短几": (("短短几秒", "短短幾秒"), ("茶几", "茶几")),       # 短几 (a short low table) is stock's phrase and swallows 短短|几秒
    "有云": (("唯有云雀", "唯有雲雀"), ("今天有云", "今天有雲")),   # 有云 ('it is said') swallows 有|云雀 and 有|云层
    "如干": (("倒不如干脆", "倒不如乾脆"), ("若干", "若干")),     # 如干 swallows 不如|干脆
    "心向": (("内心向往", "內心嚮往"), ("心向党", "心向黨")),     # 心向 swallows 内心|向往
    # found in the Taiwan corpus (G4: the Taiwan wording in 5 or more places, the stock wording in none): each swallows the first character of the next word
    "并包": (("并包含", "並包含"), ("合并", "合併")),             # 并包 (幷包) swallows 并|包含: 11 / 0
    "并产": (("并产生", "並產生"), ("合并", "合併")),             # 并产 (併產) swallows 并|产生: 7 / 0
    "被发": (("被发送", "被發送"), ("被发现", "被發現")),         # 被发 (被髮, 'with loose hair') swallows 被|发送 / 被|发现: 19 / 0
    "先尝": (("先尝试", "先嘗試"), ("尝试", "嘗試")),             # 先尝 (先嚐) swallows 先|尝试: 7 / 0
}
# --------------------------------------------------------------------------------------------------------------------
# c. CONTEXT_RULES. Constants used by them:
# --------------------------------------------------------------------------------------------------------------------
_NUMERALS = "一二三四五六七八九十百千万两几数多零〇0-9０-９"   # 海里 after one of these is the nautical mile
# Words counted with 隻 (animals, paired body parts) that follow ``的``: 这只会飞的鸟 is "this bird that can fly"; 这只会更糟 is "only"
ZHI_NOUNS = ("鸟", "猫", "狗", "鸡", "鸭", "鹅", "羊", "兔", "虫", "蚁", "蜂", "蝶", "蛙", "虎", "狼", "狐", "熊", "猴", "鹿", "鼠", "鹰", "雁",
             "鸦", "雀", "燕", "鸥", "鹤", "鹦鹉", "孔雀", "蛇", "鱼", "龙", "蚊", "蝇", "蜘蛛", "松鼠", "刺猬", "蝙蝠", "乌龟", "老虎", "狮", "象", "牛", "马", "猪", "手", "脚", "眼", "耳")
_CLAUSE_END = "，。！？!?,.;；、…\n"
_ZHI_AFTER = re.compile(r"[^%s]{0,6}?的[小大老白黑黄花红灰]?(?:%s)" % (re.escape(_CLAUSE_END), "|".join(ZHI_NOUNS)))


def _zhi(match: "re.Match", rest: str) -> str:
    """这只 / 那只 / 是只 followed by 有 能 会 想 要: 隻 when the same clause has 的 + a noun counted with 隻, otherwise 只 (the user's wording)."""
    measure = not rest.startswith("是") and _ZHI_AFTER.match(rest) is not None
    lead = {"这": "這", "那": "那"}.get(match.groupdict().get("lead") or "", "")
    return lead + ("隻" if measure else "只")


_VERB_AFTER = "有能会想要是在"
CONTEXT_RULES: tuple = (
    # --- 海里: nautical mile after a number, otherwise "in the sea"
    Context("sea", r"(?<![%s])(?P<fix>海里)" % _NUMERALS, "海裡",
            "海里 after a numeral is the nautical mile (stays 海里); anywhere else it is 海+里 'in the sea' -> 海裡",
            (("在海里游泳", "在海裡游泳"), ("山里海里", "山裡海裡")), (("离岸几海里", "離岸幾海里"), ("航行了二十海里", "航行了二十海里"))),
    # --- 干了: stock reads "do" (幹); "dried / drained / finished a drink" (乾) only in closed contexts
    Context("dry-drink", r"(?:杯|酒|汤|碗|瓶|一口|一饮).{0,3}(?P<fix>干)了(?!一(?:件|番|场|票|桩|笔|单|仗|架|辈))", "乾",
            "a cup / wine / soup / bowl / bottle (or 一口 一饮) within four characters before 干了 is a drink finished: 乾了",
            (("把这杯干了吧", "把這杯乾了吧"), ("这杯酒我干了", "這杯酒我乾了")), (("他把这件事干了吧", "他把這件事幹了吧"), ("他干了一件大事", "他幹了一件大事"))),
    Context("dry-verb", r"(?:喝|晾|晒|流|烧|熬|风|汤|碗|喉咙|嘴巴?|口|泪水?|眼泪|衣服)(?P<fix>干)了", "乾",
            "喝干 晾干 晒干 流干 烧干 熬干 风干, and 汤 / 碗 / 喉咙 / 嘴 / 口 / 泪 / 眼泪 / 衣服 + 干了, are dried: 乾了 (stock writes 熬幹了, 喉嚨幹了, 衣服幹了)",
            (("喝干了", "喝乾了"), ("熬干了", "熬乾了"), ("喉咙干了", "喉嚨乾了"), ("衣服干了", "衣服乾了")), (("他干了坏事", "他幹了壞事"), ("苦干了一辈子", "苦幹了一輩子"))),
    Context("dry-toast", r"(?P<fix>干)了(?:这杯|杯|就没事)", "乾",
            "干了这杯 / 干了杯 / 干了就没事: the drink or the thirst is gone: 乾了",
            (("干了这杯酒", "乾了這杯酒"), ("干了就没事", "乾了就沒事")), (("干了这件事", "幹了這件事"), ("干了一杯", "幹了一杯"))),
    Context("dry-place", r"(?:河床|池塘|河水?|湖水?|井水?|泉水?|水|熬)(?:都|全|已经)?(?P<fix>干)了", "乾",
            "a river / pond / well / spring / water word right before 干了 is dried up (熬干了 is boiled dry): 乾了",
            (("河床干了什么鱼都没有", "河床乾了什麼魚都沒有"), ("池塘干了啥都不剩", "池塘乾了啥都不剩")), (("他干了水电的活", "他幹了水電的活"),)),
    # --- 这只 / 那只 / 是只 + 有 能 会 想 要: "only" (只) unless the clause has 的 + an animal (隻)
    Context("zhi-demonstrative", r"(?P<fix>(?P<lead>[这那])只)(?=[%s])" % _VERB_AFTER, _zhi,
            "stock reads 这只 as 這隻 (measure word); before 有 能 会 想 要 the user's lyrics mean 'only' (這只), unless the clause has 的 + an animal noun (這隻會飛的鳥)",
            (("这只有五分钟", "這只有五分鐘"), ("这只会飞的鸟", "這隻會飛的鳥")), (("这只猫", "這隻貓"), ("这只会唱歌的鹦鹉", "這隻會唱歌的鸚鵡"))),
    Context("zhi-adverb", r"(?<![一二三四五六七八九十两几每各这那哪半整多少\d])(?P<fix>只)(?=[%s])" % _VERB_AFTER, _zhi,
            "只 before 有 能 会 想 要 是 在 and not after a numeral or demonstrative is 'only' (没有只要 别只是 总是只有), 隻 when the clause has 的 + an animal noun (是只有毒的蛇); stock's 有只 / 别只 / 是只 always read it as 隻",
            (("没有只要祈求", "沒有只要祈求"), ("别只是旁观", "別只是旁觀"), ("总是只有我", "總是只有我"), ("他是只有毒的蛇", "他是隻有毒的蛇")), (("一只会飞的鸟", "一隻會飛的鳥"), ("两只是黑的", "兩隻是黑的"))),
    Context("wind-after", r"(?<=[的了着吹随被把乘借顺逆迎])(?P<fix>风后)", "風後",
            "风后 after 的 了 着 吹 ... is 'after the wind' (風後); alone it is the name 風后 (stock)",
            (("吸入湿润的风后", "吸入濕潤的風後"),), (("黄帝的大臣风后", "黃帝的大臣風后"),)),
    Context("dry-compound", r"(?<=[没不未])(?P<fix>干)(?=[透净燥涸枯爽])", "乾",
            "没干透 / 没干净: dried (乾); stock's phrase 没干 -> 沒幹 swallows the 干",
            (("道路还没干透吧", "道路還沒乾透吧"),), (("他没干坏事", "他沒幹壞事"),)),
    Context("curl", r"(?<![试长画书手经压考答])(?P<fix>卷卷)", "捲捲",
            "卷卷 = curling (捲捲而上, 捲捲的头发); after 试 长 画 书 手 经 压 考 答 the first 卷 belongs to the word before (试卷|卷面)",
            (("卷卷而上的火焰", "捲捲而上的火焰"),), (("试卷卷面整洁", "試卷卷面整潔"), ("长卷卷首有题字", "長卷卷首有題字"))),
    # --- 松: the pine of a tree word followed by 开 (青松|开花, 古松|开始); stock's phrase 松开 (鬆開) would swallow the 开
    Context("pine", r"(?<=[青古老孤苍劲乔雪])(?P<fix>松)(?=开)", "松",
            "after 青 古 老 孤 苍 劲 乔 雪 the 松 is a pine tree and the 开 starts the next word (青松|开花); stock's phrase 松开 -> 鬆開 hides it",
            (("青松开花了", "青松開花了"), ("古松开始变黄", "古松開始變黃")), (("鞋带又松开了", "鞋帶又鬆開了"), ("松开我的手", "鬆開我的手"))),
    # --- 游: the 游 of a tourism word before 动 过 移 (旅游|动态, 漫游|过去, 导游|过来)
    Context("tour", r"(?<=[旅漫导周巡夜远出畅郊])(?P<fix>游)(?=[动过移])", "遊",
            "after 旅 漫 导 周 巡 夜 远 出 畅 郊 the 游 is 遊 (tour) and 动 过 移 starts the next word; the preference 游动/游过/游移 (swim) would take it",
            (("漫游过去的事", "漫遊過去的事"), ("导游过来接我们", "導遊過來接我們")), (("他游过了河", "他游過了河"), ("鱼游动着", "魚游動著"))),
    # --- 刮: shaving / scraping (刮), not the wind (颳)
    Context("scrape", r"(?:(?<=胡子)|(?<=[胡须脸毛腿头眉]))(?P<fix>刮)", "刮",
            "after 胡 须 脸 毛 腿 头 眉 the 刮 is shave / scrape (刮); stock's phrase 刮了 is the wind 颳了",
            (("胡子刮了", "鬍子刮了"), ("脸刮干净", "臉刮乾淨")), (("刮了一阵风", "颳了一陣風"),)),
    # --- 余: the surname before a title
    Context("surname-yu", r"(?:^|(?<=[姓叫是找问请跟和对给让被把由说谢与同]))(?P<fix>余)(?=先生|小姐|老师|医生|教授|女士|同学|太太|博士|经理|师傅)", "余",
            "the surname 余 at the start of a clause (or after 姓 叫 是 找 ...) before a title stays 余; stock writes 餘",
            (("余先生在家", "余先生在家"), ("我找余老师", "我找余老師")), (("剩余先生产", "剩餘先生產"),)),
)
# --------------------------------------------------------------------------------------------------------------------
# d. VARIANT_PREFERENCE. Characters where stock writes one standard Traditional form and the user's lyrics use the other
# form more often (count of the form written here / of the form stock writes). Applied only to characters the conversion
# produced from a different character, never to Traditional text that was already in the line, never to an ENTRIES word or
# a CONTEXT_RULES span. Only a character whose two forms are variants of each other in EVERY word belongs here (audit of
# every OpenCC word that produces the character: docs/ZH_S2TW_ROUNDTRIP.md); 註 / 注 and 嚮 / 向 are not.
# --------------------------------------------------------------------------------------------------------------------
VARIANT_PREFERENCE: dict = {
    "臺": "台",   # 17 / 3
    "溼": "濕",   # 11 / 0
    "汙": "污",   # 3 / 2
    "燬": "毀",   # 8 / 0
    "絃": "弦",   # 4 / 0
    "悽": "淒",   # 2 / 0
    "歎": "嘆",   # 12 / 0
    "唸": "念",   # 50 / 0
}

_dicts = None


def _dictionary_dir():
    try:
        import opencc
    except Exception:
        return None
    path = os.path.join(os.path.dirname(opencc.__file__), "dictionary")
    return path if os.path.isdir(path) else None


def _read_dictionary(directory, name):
    table = {}
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            table[key] = value.split(" ")
    return table


def entries():
    """``{simplified: Traditional}`` of ENTRIES (corrections and preferences)."""
    return {item.key: item.value for item in CORRECTIONS + PREFERENCES}


@functools.lru_cache(maxsize=1)
def _base():
    """The OpenCC dictionaries as read from disk, shared by every table build, or None."""
    directory = _dictionary_dir()
    if directory is None:
        return None
    try:
        import opencc
        phrases = {key: value[0] for key, value in _read_dictionary(directory, "STPhrases.txt").items()}
        characters = _read_dictionary(directory, "STCharacters.txt")
        variants = {key: value[0] for key, value in _read_dictionary(directory, "TWVariants.txt").items()}
        reverse = _read_dictionary(directory, "TSCharacters.txt")
        split = opencc.OpenCC("s2tw").split_chars_re
    except Exception:
        return None
    traditional_only = frozenset(key for key, value in reverse.items() if value[0] != key)
    return phrases, characters, variants, traditional_only, split


def _tables():
    """``(phrases, characters, variants, traditional_only, longest, split, context, entry_keys)`` or None when the OpenCC dictionaries cannot be read.

    ``phrases`` is STPhrases without DROPPED_PHRASES plus ENTRIES, ``context`` the compiled CONTEXT_RULES. The rule tables are read
    when the tables are first built; set ``_dicts = None`` after changing them (the tests do)."""
    global _dicts
    if _dicts is None:
        _dicts = False
        base = _base()
        if base is not None:
            phrases = dict(base[0])
            for key in DROPPED_PHRASES:
                phrases.pop(key, None)
            phrases.update(entries())
            context = tuple((re.compile(rule.pattern), rule.value) for rule in CONTEXT_RULES)
            _dicts = (phrases, base[1], base[2], base[3], max(map(len, phrases)), base[4], context, frozenset(entries()))
    return _dicts or None


def converter():
    """The shared s2tw converter, or None when OpenCC is not installed."""
    global _converter, _loaded
    if not _loaded:
        _loaded = True
        try:
            import opencc
            _converter = opencc.OpenCC("s2tw")
        except Exception:
            _converter = None
    return _converter


def has_kana(text):
    return bool(_KANA_RE.search(text))


def _encodes(char, codec):
    try:
        char.encode(codec)
        return True
    except UnicodeEncodeError:
        return False


def _segment(text, phrases, longest):
    """Phrase spans ``(start, end)`` of ``text`` the way stock OpenCC finds them: the longest phrase in the text, the leftmost
    among those of the same length, then the same inside the left part and inside the right part."""
    spans = []
    pending = [(0, len(text))]
    while pending:
        low, high = pending.pop()
        for size in range(min(longest, high - low), 1, -1):
            for start in range(low, high - size + 1):
                if text[start:start + size] in phrases:
                    spans.append((start, start + size))
                    pending.append((low, start))
                    pending.append((start + size, high))
                    break
            else:
                continue
            break
    spans.sort()
    return spans


def _fixed_spans(piece, context):
    """``[(start, end, Traditional)]`` of the CONTEXT_RULES matches in one clause, no overlaps (leftmost, then longest, then rule order)."""
    found = []
    for order, (pattern, value) in enumerate(context):
        position = 0
        while True:
            match = pattern.search(piece, position)
            if match is None:
                break
            start, end = match.span("fix") if "fix" in pattern.groupindex else match.span()
            if start == end:
                position = match.end() + 1
                continue
            written = value(match, piece[match.end():]) if callable(value) else value
            found.append((start, -(end - start), order, end, written))
            position = start + 1
    found.sort()
    spans, last = [], 0
    for start, _negative, _order, end, written in found:
        if start >= last:
            spans.append((start, end, written))
            last = end
    return spans


@functools.lru_cache(maxsize=1)
def one_to_many():
    """``{simplified char: [Traditional candidates]}`` for every character STCharacters maps to more than one form."""
    tables = _tables()
    if tables is None:
        return {}
    return {key: value for key, value in tables[1].items() if len(value) > 1}


def _convert_clause(piece, tables):
    """``[(source, target, fixed)]`` tokens covering one clause (no separators in it)."""
    phrases, characters, _variants_table, _only, longest, _split, context, entry_keys = tables
    tokens = []

    def plain(low, high):
        text = piece[low:high]
        spans = {start: end for start, end in _segment(text, phrases, longest)}
        position = 0
        while position < len(text):
            end = spans.get(position)
            if end is not None:
                source = text[position:end]
                tokens.append((source, phrases[source], source in entry_keys))
            else:
                end = position + 1
                tokens.append((text[position], characters.get(text[position], [text[position]])[0], False))
            position = end

    last = 0
    for start, end, written in _fixed_spans(piece, context):
        plain(last, start)
        tokens.append((piece[start:end], written, True))
        last = end
    plain(last, len(piece))
    return tokens


def convert_marked(text):
    """``(converted text, marks)``; ``marks`` is ``[{"at": index in the converted text, "from": char, "options": [...]}]``
    for every character of ``text`` that STCharacters maps to several Traditional forms. None without OpenCC."""
    tables = _tables()
    if tables is None:
        return None
    variants, split = tables[2], tables[5]
    many = one_to_many()
    out, marks, length = [], [], 0
    for index, piece in enumerate(split.split(text)):
        if index % 2:
            out.append(piece)
            length += len(piece)
            continue
        for source, target, fixed in _convert_clause(piece, tables):
            target = _variants(target, variants)
            if VARIANT_PREFERENCE and not fixed and len(target) == len(source):
                target = "".join(VARIANT_PREFERENCE.get(new, new) if new != old else new for new, old in zip(target, source))
            for offset, char in enumerate(source):
                if char in many and len(target) == len(source):
                    marks.append({"at": length + offset, "from": char, "options": list(many[char])})
            out.append(target)
            length += len(target)
    return "".join(out), marks


def _variants(text, variants):
    """The s2tw second stage: TWVariants applied to the result of the first one (single characters only in this dictionary)."""
    return "".join(variants.get(char, char) for char in text)


def convert_text(text):
    """The Traditional (Taiwan) text for ``text``; ``text`` itself when OpenCC is missing or nothing changes."""
    pair = convert_marked(text)
    if pair is None:
        cc = converter()
        return cc.convert(text) if cc is not None else text
    return pair[0]


def has_traditional_only(text):
    """True when ``text`` has a character only Traditional uses (OpenCC t2s rewrites it: 運 們 裡)."""
    tables = _tables()
    return tables is not None and any(char in tables[3] for char in text)


def looks_simplified(text, japanese=False):
    """True when ``text`` has a character that is certainly simplified: a Han character Big5 (cp950) cannot encode and
    that s2tw rewrites, or 么 after 什 怎 这 那 多 要 甚 (Traditional text writes 麼; Big5 does encode 么). A Traditional
    character s2tw only turns into a Taiwan variant (範, 臺, 裡) is in Big5, and a Hong Kong character (哋 嘅 喺) is
    missing from Big5 but has no s2tw rewrite. In a song with kana the character must also be missing from cp932, which
    keeps Japanese kanji such as 気 国 独."""
    cc = converter()
    if cc is None:
        return False
    if _CERTAIN_MA_RE.search(text):
        return True
    for char in _HAN_RE.findall(text):
        if char == _TAIWAN_CHURCH_CHAR or _encodes(char, "cp950"):
            continue
        if japanese and _encodes(char, "cp932"):
            continue
        if cc.convert(char) != char:
            return True
    return False


# Chinese function words. A line that has two different ones is not Japanese kanji-only text (初恋 一期一会 回来 have none).
# Characters Japanese words use a lot on their own (的 在 有 不 ...) count only in pairs; the set is the common
# Mandarin particles, pronouns, auxiliaries and prepositions of lyrics.
CHINESE_FUNCTION_CHARS = frozenset("的了是不我你他她在有也就都很这那把被给让没还和吗呢吧啊嘛着过么")


def _surely_not_japanese(text):
    """True when ``text`` has a Han character cp932 cannot encode (a Japanese line never has one)."""
    return any(not _encodes(char, "cp932") for char in _HAN_RE.findall(text))


def _chinese_function_words(text):
    return len({char for char in text if char in CHINESE_FUNCTION_CHARS})


def convert_line(text):
    """The s2tw text of a non-kana line when it differs from ``text``; None otherwise."""
    if not text or has_kana(text):
        return None
    if converter() is None:
        return None
    after = convert_text(text)
    return after if after != text else None


def convert_header(text):
    """``(before, after)`` of the text inside an ID tag line (``[ti:...]``) when it converts; None otherwise (same kana rule as a lyric line)."""
    match = HEADER_TEXT_RE.match(text.rstrip("\r"))
    if not match:
        return None
    after = convert_line(match.group(2))
    return (match.group(2), after) if after is not None else None


def _marks_for(text):
    pair = convert_marked(text)
    return pair[1] if pair is not None else []


def song_candidates(rows, headers=(), uncertain=False):
    """``{count, optional, japanese, uncertain, lines, headers}`` for the payload rows, or None when OpenCC is missing.

    Candidates are single lines: ``lines`` holds ``{entry, sub, before, after, marks, default_checked}`` where ``sub``
    is the 0-based position of the line inside its entry (a translation line under a lyric line is its own candidate).
    ``headers`` are ``(line_index, text)`` of the ID tag lines.

    The song gate: nothing is listed unless some line has a character that is certainly simplified (see
    ``looks_simplified``; 么 after 什 怎 这 那 多 要 甚 counts, in songs with and without kana). Past the gate, line by line
    (a line with kana is never a candidate) every line s2tw would change is listed:
    * a certainly simplified character: checked;
    * otherwise, in a song without kana: checked when the line has no Traditional-only character (``后面``, ``一起干杯``),
      unchecked when it has one (``命運若早已注定``: 注 -> 註 only swaps a variant);
    * otherwise, in a song with kana, the line has no Traditional-only character and is checked when it has a Han character
      cp932 cannot encode, or two different Chinese function words (CHINESE_FUNCTION_CHARS), or one function word while
      another line of the same row has kana (the translation under a Japanese line); the rest (a kanji-only Japanese line
      such as ``初恋``, ``一期一会``) is listed unchecked.
    All candidates start unchecked when the encoding guess is uncertain (the text may be mojibake). ``count`` is the
    number of candidates that start checked; ``optional`` the number that start unchecked.
    """
    if converter() is None:
        return None
    japanese = any(has_kana(item) for row in rows for item in (row.get("lyrics") or []) if isinstance(item, str))
    lines = []
    found = []
    sure = False
    for row in rows:
        texts = [item for item in (row.get("lyrics") or []) if isinstance(item, str)]
        row_kana = any(has_kana(item) for item in texts)
        for sub, text in enumerate(row.get("lyrics") or []):
            if not isinstance(text, str):
                continue
            after = convert_line(text)
            if after is None:
                continue
            certain = looks_simplified(text, japanese)
            sure = sure or certain
            lines.append({"entry": row["entry"], "sub": sub, "before": text, "after": after, "_certain": certain, "_row_kana": row_kana})
    for index, text in headers:
        pair = convert_header(text)
        if pair is not None:
            certain = looks_simplified(pair[0], japanese)
            sure = sure or certain
            found.append({"line": index, "tag": text[1:3].lower(), "before": pair[0], "after": pair[1], "_certain": certain, "_row_kana": False})
    if not sure:
        return {"count": 0, "optional": 0, "japanese": japanese, "uncertain": bool(uncertain), "lines": [], "headers": []}
    for items in (lines, found):
        for item in items:
            certain = item.pop("_certain")
            row_kana = item.pop("_row_kana")
            if certain:
                checked = True
            elif has_traditional_only(item["before"]):
                checked = False
            elif not japanese:
                checked = True
            else:
                words = _chinese_function_words(item["before"])
                checked = _surely_not_japanese(item["before"]) or words >= 2 or (row_kana and words >= 1)
            item["default_checked"] = bool(checked and not uncertain)
            item["marks"] = _marks_for(item["before"])
    checked_count = sum(1 for item in lines + found if item["default_checked"])
    return {"count": checked_count, "optional": len(lines) + len(found) - checked_count,
            "japanese": japanese, "uncertain": bool(uncertain), "lines": lines, "headers": found}
