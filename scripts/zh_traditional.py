"""Simplified -> Traditional (Taiwan) detection for the review page.

OpenCC ``s2tw`` dictionaries only (never ``s2twp``: it rewrites vocabulary). A line with any kana is Japanese and is
never converted. Without OpenCC every function answers "nothing to convert", so the review never depends on it.

The conversion is stock OpenCC ``s2tw`` (opencc-python-reimplemented) run on our own tables: split at the separators,
then recursively "take the longest phrase of STPhrases, the leftmost one among equally long ones, and convert the two
sides the same way", the remaining characters through STCharacters, then TWVariants. With the four rule tables below
emptied the output is character for character the stock output (the test suite checks every STPhrases word).

Everything that differs from stock is one of five kinds of rule, each with a reason and a positive and a negative
example that the tests run (docs/ZH_S2TW_ROUNDTRIP.md lists them):

* ``ENTRIES``: a word added to or replaced in the dictionary; it takes part in the segmentation like any word, with no priority;
* ``DROPPED_PHRASES``: a stock phrase that is not used;
* ``CONTEXT_RULES``: a regular expression over the simplified text that fixes a span and its Traditional form where only the
  surrounding text can tell (a fixed span is a boundary: no phrase crosses it);
* ``VARIANT_PREFERENCE``: a character-level variant choice applied to the characters the conversion produced;
* ``WHOLE_LINES``: a whole line (white space at either end aside) written the way the user's lyrics write it; the same words
  inside a longer line, after a comma or with one more character in front go the stock way.

A PREFERENCES word never takes a character away from the stock reading: a word that stock also has (计划) only replaces the value, and
where a stock phrase crosses its border (预计|划分) it keeps the stock value; a word stock does not have (点赞) is left out of the
segmentation there (点|赞同).
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
    # 一直发X / 一头发X: stock's phrase 直发 / 头发 (hair) swallows the 发 of the verb; each word below is the whole string, so 梳着一直发 (一|直发) stays stock
    Entry("一直发现着", "一直發現著", "stock's 直发 (直髮) swallows the 发 of 一直|发现; the word ends in 着 so 一直发现在 (hair + now) stays stock", ("梳着一直发", "梳著一直髮")),
    Entry("一直发现著", "一直發現著", "the same word written with 著, which is what a Traditional-to-Simplified round trip leaves", ("梳着一直发", "梳著一直髮")),
    Entry("短短几", "短短幾", "stock's phrase 短几 (a low table) swallows 短短|几秒", ("一张短几", "一張短几")),
    Entry("一起干杯", "一起乾杯", "stock's phrase 一起干 (一起幹) swallows 一起|干杯", ("一起干活", "一起幹活")),
    Entry("没干透吧", "沒乾透吧", "stock's phrase 没干 (沒幹) swallows the 干 of 没|干透 (not dry yet); the word ends in 吧 so 没干透露 stays stock", ("他没干坏事", "他沒幹壞事")),
    Entry("这杯干了", "這杯乾了", "this cup / drink is finished (stock 幹了); the word is the whole string, a 这杯 elsewhere stays stock", ("他把这件事干了吧", "他把這件事幹了吧"), ("把这杯干了吧", "把這杯乾了吧")),
    Entry("若是只有", "若是只有", "若是 + 只有 = 'if only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("若是只有", "若是只有")),
    Entry("总是只有", "總是只有", "总是 + 只有 = 'always only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("总是只有", "總是只有")),
    Entry("总是只在", "總是只在", "总是 + 只在 = 'always only in'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("总是只在", "總是只在")),
    Entry("还是只想", "還是只想", "还是 + 只想 = 'still only want'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("还是只想", "還是只想")),
    Entry("没有只要", "沒有只要", "没有 + 只要 = 'without just asking'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("没有只要", "沒有只要")),
    Entry("别只是", "別只是", "别 + 只是 = 'don't just'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb", ("院子里有只会说话的八哥", "院子裡有隻會說話的八哥"), ("别只是", "別只是")),
    Entry("卷卷而上", "捲捲而上", "curling upward (捲捲而上); stock leaves 卷卷", ("问卷卷成了筒状", "問卷捲成了筒狀")),
    Entry("湿润的风后", "濕潤的風後", "wet air after the wind (風後); 风后 alone stays the name 風后 (stock)", ("黄帝的大臣风后", "黃帝的大臣風后")),
    Entry("闪烁的星海里", "閃爍的星海裡", "in the sea of stars (stock keeps 海里, the nautical mile); the word starts with 闪烁的 so 十二星海里 is not touched", ("多少海里才到岸", "多少海里才到岸")),
    # 被发 + a verb: stock reads 被发 as 被髮 (with loose hair: 被发乱首, 被发行吟; stock is right there and untouched); each modern verb below is the whole string
    Entry("被发送", "被發送", "stock reads 被发 as 被髮 (loose hair); here it is 被|发送", ("被发乱首", "被髮亂首")),
    Entry("被发生", "被發生", "stock reads 被发 as 被髮 (loose hair); here it is 被|发生", ("被发乱首", "被髮亂首")),
    Entry("被发布", "被發布", "stock reads 被发 as 被髮 (loose hair); here it is 被|发布", ("被发乱首", "被髮亂首")),
    Entry("被发表", "被發表", "stock reads 被发 as 被髮 (loose hair); here it is 被|发表", ("被发乱首", "被髮亂首")),
    Entry("了如列眉", "瞭如列眉", "idiom: stock only has 了如指掌 and the dropped 了如 phrase leaves 了如列眉 unconverted (0 / 1 in the Taiwan corpus)", ("我明白了如何做", "我明白了如何做")),
    Entry("唯有云雀的声音", "唯有雲雀的聲音", "唯有|云雀 (a lark); stock's classical phrase 有云 ('it is said') would hide it; the word is the whole phrase so 有云雀跃 stays stock", ("古人有云雀跃不已", "古人有云雀躍不已")),
    Entry("不是只有一次", "不是只有一次", "不是 + 只有 + 一次 = 'not only once' (stock reads 是只 as 是隻)", ("那不是只有斑点的狗吗", "那不是隻有斑點的狗嗎")),
    Entry("这杯酒我干了", "這杯酒我乾了", "'I drain this wine': stock 幹了; the word is the whole phrase", ("卖酒我干了十年", "賣酒我幹了十年")),
    Entry("如果喉咙干了", "如果喉嚨乾了", "if the throat is dry (stock 喉嚨幹了); the word is the whole phrase, so 喉咙干了坏事 stays stock", ("捂着喉咙干了坏事", "捂著喉嚨幹了壞事")),
    Entry("四库禁毁书丛刋", "四庫禁燬書叢刋", "a book series title written with 燬 (the variant preference 毀 does not apply)", ("烧毁", "燒毀")),
)
PREFERENCES: tuple = (
    Entry("鱼类在游动", "魚類在游動", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics", ("旅游动态", "旅遊動態")),
    Entry("正游动在", "正游動在", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics", ("我们正游动物园", "我們正遊動物園")),
    Entry("正游动其中", "正游動其中", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics", ("我们正游动物园", "我們正遊動物園")),
    Entry("交织游动吗", "交織游動嗎", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics", ("我们交织游动物园", "我們交織遊動物園")),
    Entry("游动着", "游動著", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics", ("旅游动态", "旅遊動態")),
    Entry("游动著", "游動著", "both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics (著 is what the Traditional-to-Simplified round trip leaves)", ("旅游动态", "旅遊動態")),
    Entry("游过群星之间", "游過群星之間", "both are standard; 游過 (swam) in the lyrics, stock 遊過; a phrase from the lyrics", ("旅游过程", "旅遊過程")),
    Entry("糊里糊涂", "糊裡糊塗", "both are standard (the dictionary headword itself is 糊里糊塗); the lyrics write 糊裡糊塗", ("糊里", "糊裡")),
    Entry("克制", "克制", "both are standard (剋制 for 相剋 senses); the lyrics write 克制 (restrain), stock gives 剋制", ("克星", "剋星")),
    Entry("女生外向", "女生外向", "both are standard (外嚮 / 外向); the lyrics write 外向, stock turns 向 into 嚮 here", ("心向往之", "心嚮往之")),
    Entry("计划", "計畫", "both are standard; 计划 -> 計畫 4 / 計劃 0 in the lyrics", ("合计划拨", "合計劃撥")),
    Entry("了解", "了解", "both are standard; 了解 4 / 瞭解 1", ("了如指掌", "瞭如指掌")),
    Entry("点赞", "點讚", "both are standard; 點讚 1 / 點贊 0", ("赞美", "讚美")),
    Entry("胡同", "胡同", "both are standard; 胡同 1 / 衚衕 0", ("胡子", "鬍子")),
    Entry("死胡同", "死胡同", "both are standard; 死胡同 1 / 死衚衕 0", ("胡子", "鬍子")),
    Entry("尽可能", "盡可能", "both are standard; 盡可能 1 / 儘可能 0", ("尽管", "儘管")),
    Entry("背包", "背包", "both are standard; 背包 29 / 揹包 1", ("背叛", "背叛")),
    Entry("背负", "背負", "both are standard; 背負 / 揹負", ("背叛", "背叛")),
    Entry("回归", "回歸", "both are standard; 回歸 1 / 迴歸 0 in the lyrics", ("线性回归", "線性迴歸")),
    Entry("回归分析", "迴歸分析", "carve-out of 回归: statistics writes 迴歸", ("香港回归", "香港回歸")),
    Entry("线性回归", "線性迴歸", "carve-out of 回归: statistics writes 迴歸", ("香港回归", "香港回歸")),
    Entry("逻辑回归", "邏輯迴歸", "carve-out of 回归: statistics writes 迴歸", ("香港回归", "香港回歸")),
    Entry("回归模型", "迴歸模型", "carve-out of 回归: statistics writes 迴歸", ("香港回归", "香港回歸")),
    Entry("回归方程", "迴歸方程", "carve-out of 回归: statistics writes 迴歸", ("香港回归", "香港回歸")),
    Entry("熏染", "燻染", "both are standard; 燻染 1 / 薰染 0", ("熏陶", "薰陶")),
    Entry("周而复始", "周而復始", "both are standard; 周而復始 3 / 週而復始 0", ("周末", "週末")),
    Entry("注定", "注定", "both are standard; 注定 25 / 註定 0 in the lyrics (註冊 註解 備註 are 註 in Taiwan and stay stock)", ("注册", "註冊")),
    Entry("命中注定", "命中注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
    Entry("命运注定", "命運注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
    Entry("五百姻缘天注定", "五百姻緣天注定", "both are standard; 注定 25 / 註定 0 in the lyrics", ("注册", "註冊")),
)
# --------------------------------------------------------------------------------------------------------------------
# a2. WHOLE_LINES: only a whole line equal to the key (white space at either end aside) is replaced. ``negative`` is a longer line
# holding the same words, which stays stock.
# --------------------------------------------------------------------------------------------------------------------
WHOLE_LINES: tuple = (
    Entry("干了就没事了！", "乾了就沒事了！", "a lyric line that stands alone: 'drink it and all is well' (stock 幹了); a literal word fails 这份作业，干了就没事了", ("这份作业，干了就没事了！", "這份作業，幹了就沒事了！")),
    Entry("干了就没事了", "乾了就沒事了", "the same lyric line without its closing mark", ("这份作业，干了就没事了", "這份作業，幹了就沒事了")),
    Entry("海里有各种鱼类在游动", "海裡有各種魚類在游動", "a lyric line that stands alone: 'fish of every kind swim in the sea' (stock keeps 海里, the nautical mile); a longer line with the same words is stock", ("方圆十海里有各种鱼类在游动", "方圓十海里有各種魚類在游動")),
)
# --------------------------------------------------------------------------------------------------------------------
# b. DROPPED_PHRASES: stock phrases that hide a better match across the same characters.
# --------------------------------------------------------------------------------------------------------------------
DROPPED_PHRASES = frozenset({"了如", "如干", "心向", "并包", "并产", "先尝"})   # kept only where the Taiwan dictionary has no sentence that needs the stock phrase (G4: 0 victims); every other stock phrase that hides a word is covered by a longer ENTRIES word instead
DROPPED_NOTES: dict = {
    "了如": (("我明白了如何做", "我明白了如何做"), ("了如指掌", "瞭如指掌")),   # 了如 is only meant for 了如指掌, which stock has as a whole phrase
    "如干": (("倒不如干脆", "倒不如乾脆"), ("若干", "若干")),     # 如干 swallows 不如|干脆
    "心向": (("内心向往", "內心嚮往"), ("心向党", "心向黨")),     # 心向 swallows 内心|向往
    "并包": (("并包含", "並包含"), ("合并", "合併")),             # 并包 (幷包) swallows 并|包含
    "并产": (("并产生", "並產生"), ("合并", "合併")),             # 并产 (併產) swallows 并|产生
    "先尝": (("先尝试", "先嘗試"), ("尝试", "嘗試")),             # 先尝 (先嚐) swallows 先|尝试
}
# c. CONTEXT_RULES: only a pattern whose left AND right side are closed lists of literal words (no gaps, no "not followed by"
# guards, no one-character left context that could end another word). Everything else is a literal ENTRIES word or stock.
# --------------------------------------------------------------------------------------------------------------------
_CONTEXT_END = "，。！？!?,.;；、… 的在是说有来去也就都很和与们了"
CONTEXT_RULES: tuple = (
    # --- 余: the surname before a title
    Context("surname-yu", r"(?:^|(?<=姓|叫|找|问|请))(?P<fix>余)(?=(?:先生|小姐|老师|医生|教授|女士|同学|太太|博士|经理|师傅)(?:$|[%s]))" % _CONTEXT_END, "余",
            "the surname 余 at the start of a clause (or right after 姓 叫 找 问 请) and right before a title that ends the word stays 余; stock writes 餘",
            (("余先生在家", "余先生在家"), ("我找余老师", "我找余老師")), (("剩余先生产", "剩餘先生產"), ("多余老师的话", "多餘老師的話"))),
)
# --------------------------------------------------------------------------------------------------------------------
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
    """``(phrases, characters, variants, traditional_only, longest, split, context, entry_keys, stock_value, whole, literal)`` or None when the OpenCC dictionaries cannot be read.

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
            stock_value = {item.key: base[0][item.key] for item in PREFERENCES if item.key in base[0]}
            whole = {item.key: item.value for item in WHOLE_LINES}
            literal = frozenset(item.key for item in PREFERENCES if item.key not in base[0])
            _dicts = (phrases, base[1], base[2], base[3], max(map(len, phrases)), base[4], context, frozenset(entries()), stock_value, whole, literal)
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


def _segment(text, phrases, longest, skip=frozenset()):
    """Phrase spans ``(start, end)`` of ``text`` the way stock OpenCC finds them: the longest phrase in the text, the leftmost
    among those of the same length, then the same inside the left part and inside the right part. ``skip`` is a set of words to leave out."""
    spans = []
    pending = [(0, len(text))]
    while pending:
        low, high = pending.pop()
        for size in range(min(longest, high - low), 1, -1):
            for start in range(low, high - size + 1):
                if text[start:start + size] in phrases and text[start:start + size] not in skip:
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
    phrases, characters, _variants_table, _only, longest, _split, context, entry_keys, stock_value, _whole, literal = tables
    tokens = []

    def contested(text, start, end, skip):
        """True when, with the word itself left out, a stock phrase crosses the border of the word at ``start``..``end``."""
        for low, high in _segment(text, phrases, longest, skip | {text[start:end]}):
            if low < start < high or low < end < high:
                return True
        return False

    def plain(low, high):
        text = piece[low:high]
        skip = frozenset()
        while True:
            found = _segment(text, phrases, longest, skip)
            lost = {text[start:end] for start, end in found if text[start:end] in literal and text[start:end] not in skip and contested(text, start, end, skip)}
            if not lost:
                break
            skip = skip | lost
        spans = {start: end for start, end in found}
        position = 0
        while position < len(text):
            end = spans.get(position)
            if end is not None:
                source = text[position:end]
                if source in stock_value and contested(text, position, end, skip):
                    tokens.append((source, stock_value[source], False))
                else:
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
    key = text.strip()
    if key in tables[9]:
        lead = text[:len(text) - len(text.lstrip())]
        marks = [{"at": len(lead) + offset, "from": char, "options": list(many[char])} for offset, char in enumerate(key) if char in many]
        return lead + tables[9][key] + text[len(lead) + len(key):], marks
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


# Chinese function words that Japanese text does not use. A line with a kanji-only Japanese word (初恋 一期一会 他国 目的 不安 存在 有名 回来)
# must not be checked because of 的 他 不 有 在 我 是 来 没 着 ..., which Japanese words contain; so only characters that are simplified-only
# forms of a different Japanese character (这 们 吗 说 对 为 过 时 个 给 让 还 谁 啥 您 咱, Japanese: 這 們 說 対 為 過 時 個 給 譲 還), or particles
# Japanese never writes in kanji (呢 哪 么), count here. Two different ones, or one next to a kana line of the same row, check the line.
CHINESE_FUNCTION_CHARS = frozenset("这们吗呢还让说对为过时个么给谁哪啥您咱你她它吧啊很")
# Two-character Chinese words, a closed list, that Japanese text does not contain even though each character may be a Japanese kanji
# (cp932 encodes 不 会 没 有 我 的 是 ...): 不会 不是 没有 你的 她的 我不 你不 我是 你是 我有 我在 你在. Left out on purpose
# because Japanese has the same word: 不要 不能 所以 有名 存在 目的 他国 回来 初恋 一期一会 (a kanji-only Japanese line must stay unchecked);
# and because Japanese has the two characters side by side inside a longer kanji word (the 〜的 adjectives 排他的 利他的 自我的 自主的, 京都是 首都是):
# 他的 我的 都是 are not in the list. Every word of the list is checked against that kind of substring (test_zh_fix13a.py).
CHINESE_FUNCTION_WORDS = frozenset(("不会", "不是", "没有", "你的", "她的", "我不", "你不", "我是", "你是", "我有", "我在", "你在"))


def _surely_not_japanese(text):
    """True when ``text`` has a Han character cp932 cannot encode (a Japanese line never has one)."""
    return any(not _encodes(char, "cp932") for char in _HAN_RE.findall(text))


def _chinese_function_words(text):
    """How many different Chinese-only items ``text`` has: characters of CHINESE_FUNCTION_CHARS and words of CHINESE_FUNCTION_WORDS."""
    found = {char for char in text if char in CHINESE_FUNCTION_CHARS}
    found.update(word for word in CHINESE_FUNCTION_WORDS if word in text)
    return len(found)


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
