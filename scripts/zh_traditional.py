"""Simplified -> Traditional (Taiwan) detection for the review page.

OpenCC ``s2tw`` dictionaries only (never ``s2twp``: it rewrites vocabulary). A line with any kana is Japanese and is
never converted. Without OpenCC every function answers "nothing to convert", so the review never depends on it.

The conversion reads OpenCC's own dictionaries (STPhrases, STCharacters, TWVariants) but segments the text itself: the
longest phrase matched from the left and the longest phrase matched from the right are both tried and the one that
covers more characters with multi-character phrases wins (see ``_segment``). The stock left-to-right match alone
turns ``一起干杯`` into ``一起幹杯`` (the phrase ``一起干`` hides ``干杯``).
"""
from __future__ import annotations

import functools
import os
import re

# Hiragana, katakana (not the middle dot U+30FB or the long-vowel mark U+30FC: Chinese lyrics use both), katakana extensions and halfwidth katakana.
_KANA_RE = re.compile(r"[぀-ヺヽ-ヿㇰ-ㇿｦ-ﾟ]")
# ID tags whose text is prose: [ti:] [ar:] [al:] [au:] [by:] [re:] [ve:]. Only the text between the colon and the closing bracket is converted.
HEADER_TEXT_RE = re.compile(r"^(\[(?:ti|ar|al|au|by|re|ve):)(.*?)(\][ 	]*)$", re.I | re.S)
# CJK ideographs: extension A, unified, compatibility.
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TAIWAN_CHURCH_CHAR = "祢"   # U+7962, the standard pronoun in Taiwan church lyrics; Big5 has no code for it
_converter = None
_loaded = False

# Corrections to the OpenCC dictionaries, found by a round trip over the user's own lyrics (Traditional -> OpenCC t2s ->
# this converter, compared character by character; numbers in docs/ZH_S2TW_ROUNDTRIP.md). Every choice follows the
# user's own wording: where both forms are standard Traditional, the one used more often in the lyrics wins.
# A phrase listed here replaces the OpenCC entry of the same key (simplified phrase -> Traditional text) and is not
# touched by VARIANT_PREFERENCE.
PHRASE_OVERRIDES: dict[str, str] = {
    # a different word, OpenCC maps it to the wrong character (dry / not 幹; wind 后 is 後; mile 海里 is 海裡 in lyrics; ...)
    "干了": "乾了", "干了什么": "幹了什麼", "干了啥": "幹了啥", "风后": "風後", "海里": "海裡", "糊里糊涂": "糊裡糊塗", "卷卷": "捲捲",
    # both forms are standard; the user writes the left one more often (count in the lyrics: left / right)
    "计划": "計畫",      # 4 / 0
    "了解": "了解",      # 4 / 瞭解 1
    "点赞": "點讚",      # 1 / 0
    "胡同": "胡同",      # 1 / 衚衕 0
    "尽可能": "盡可能",  # 1 / 儘可能 0
    "向往": "嚮往",      # 4 / 0 (the character table below prefers 向 for every other word)
    "游动": "游動",      # 7 / 遊動 0
    "背包": "背包", "背负": "背負",   # 29 / 揹 1
    "回归": "回歸",      # 1 / 迴歸 0
    "熏染": "燻染",      # 1 / 薰染 0
    "书签": "書籤",      # 1 / 書簽 0
    "尝到": "嚐到",      # 1 / 嘗到 0
    "周而复始": "周而復始",   # 3 / 週而復始 0
    "死胡同": "死胡同",  # 1 / 死衚衕 0
    "游过": "游過", "游移": "游移",   # swim / drift, not 遊
    "这只": "這隻", "那只": "那隻",   # measure word; 这只是 / 这只有 still match 只是 / 只有 from the right
    "松开": "鬆開",      # loosen, not the pine 松 (24 / 0)
}
# OpenCC phrases that hide a better match across the same characters (``一起干`` is a phrase for ``一起干活``; in
# ``一起干杯`` it would swallow the 干 that belongs to ``干杯``). They are not used.
DROPPED_PHRASES = frozenset({"一起干", "了如"})   # 了如 is only meant for 了如指掌, which OpenCC has as a whole phrase
# Characters where OpenCC writes one standard Traditional form and the user's lyrics use the other form more often
# (lyrics count of the form written here / of the form OpenCC writes). Applied only to characters the conversion
# produced, never to Traditional text that was already in the line.
VARIANT_PREFERENCE: dict[str, str] = {
    "臺": "台",   # 17 / 3
    "溼": "濕",   # 11 / 0
    "註": "注",   # 25 / 0
    "汙": "污",   # 3 / 2
    "燬": "毀",   # 8 / 0
    "絃": "弦",   # 4 / 0
    "悽": "淒",   # 2 / 0
    "歎": "嘆",   # 12 / 0
    "唸": "念",   # 50 / 0
    "嚮": "向",   # 124 / 4
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


def _tables():
    """``(phrases, characters, variants, traditional_only)`` or None when the OpenCC dictionaries cannot be read."""
    global _dicts
    if _dicts is None:
        _dicts = False
        directory = _dictionary_dir()
        if directory is not None:
            try:
                phrases = {key: value[0] for key, value in _read_dictionary(directory, "STPhrases.txt").items()}
                characters = _read_dictionary(directory, "STCharacters.txt")
                variants = {key: value[0] for key, value in _read_dictionary(directory, "TWVariants.txt").items()}
                reverse = _read_dictionary(directory, "TSCharacters.txt")
            except OSError:
                reverse = None
            if reverse is not None:
                for key in DROPPED_PHRASES:
                    phrases.pop(key, None)
                phrases.update(PHRASE_OVERRIDES)
                traditional_only = frozenset(key for key, value in reverse.items() if value[0] != key)
                _dicts = (phrases, characters, variants, traditional_only, max(map(len, phrases)))
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


def _forward(text, phrases, longest):
    """Phrase spans ``(start, end)`` of the longest match from the left."""
    spans, position = [], 0
    while position < len(text):
        for size in range(min(longest, len(text) - position), 1, -1):
            if text[position:position + size] in phrases:
                spans.append((position, position + size))
                position += size
                break
        else:
            position += 1
    return spans


def _backward(text, phrases, longest):
    """Phrase spans of the longest match from the right, in text order."""
    spans, position = [], len(text)
    while position > 0:
        for size in range(min(longest, position), 1, -1):
            if text[position - size:position] in phrases:
                spans.append((position - size, position))
                position -= size
                break
        else:
            position -= 1
    return spans[::-1]


def _segment(text, phrases, longest):
    """Phrase spans for ``text``: the left match and the right match are compared and the better one wins.

    Score, in order: the characters covered by a PHRASE_OVERRIDES phrase (curated from the user's lyrics), then the
    characters covered by any multi-character phrase. On a tie the right match is used: the head of a Chinese word is
    at its end and matching from the right is the more accurate direction (``一直发现`` is 一直+发现, not the phrase
    ``直发``; ``总是只有`` is 只有, not 是只)."""
    left = _forward(text, phrases, longest)
    right = _backward(text, phrases, longest)
    if left == right:
        return left

    def score(spans):
        return (sum(end - start for start, end in spans if text[start:end] in PHRASE_OVERRIDES),
                sum(end - start for start, end in spans))

    return left if score(left) > score(right) else right


@functools.lru_cache(maxsize=1)
def one_to_many():
    """``{simplified char: [Traditional candidates]}`` for every character STCharacters maps to more than one form."""
    tables = _tables()
    if tables is None:
        return {}
    return {key: value for key, value in tables[1].items() if len(value) > 1}


def convert_marked(text):
    """``(converted text, marks)``; ``marks`` is ``[{"at": index in the converted text, "from": char, "options": [...]}]``
    for every character of ``text`` that STCharacters maps to several Traditional forms. None without OpenCC."""
    tables = _tables()
    if tables is None:
        return None
    phrases, characters, variants, _only, longest = tables
    spans = {start: end for start, end in _segment(text, phrases, longest)}
    out, marks, position = [], [], 0
    many = one_to_many()
    while position < len(text):
        end = spans.get(position)
        if end is not None:
            source = text[position:end]
            target = phrases[source]
        else:
            end = position + 1
            source = text[position]
            target = characters.get(source, [source])[0]
        target = _variants(target, variants)
        if VARIANT_PREFERENCE and len(target) == len(source) and source not in PHRASE_OVERRIDES:
            target = "".join(VARIANT_PREFERENCE.get(new, new) if new != old else new for new, old in zip(target, source))
        base = sum(len(piece) for piece in out)
        for offset, char in enumerate(source):
            if char in many and len(target) == len(source):
                marks.append({"at": base + offset, "from": char, "options": list(many[char])})
        out.append(target)
        position = end
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
    that s2tw rewrites. A Traditional character s2tw only turns into a Taiwan variant (範, 臺, 裡) is in Big5, and a
    Hong Kong character (哋 嘅 喺) is missing from Big5 but has no s2tw rewrite. In a song with kana the character
    must also be missing from cp932, which keeps Japanese kanji such as 気 国 独."""
    cc = converter()
    if cc is None:
        return False
    for char in _HAN_RE.findall(text):
        if char == _TAIWAN_CHURCH_CHAR or _encodes(char, "cp950"):
            continue
        if japanese and _encodes(char, "cp932"):
            continue
        if cc.convert(char) != char:
            return True
    return False


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
    """``{count, japanese, uncertain, lines, headers}`` for the payload rows, or None when OpenCC is missing.

    Candidates are single lines: ``lines`` holds ``{entry, sub, before, after, marks, default_checked}`` where ``sub``
    is the 0-based position of the line inside its entry (a translation line under a lyric line is its own candidate).
    ``headers`` are ``(line_index, text)`` of the ID tag lines.

    The song gate is unchanged: nothing is listed unless some line has a character that is certainly simplified (see
    ``looks_simplified``). Past the gate, line by line (a line with kana is never a candidate):
    * a certainly simplified character: listed, checked;
    * in a song without kana, no such character, s2tw would still change the line: checked when the line has no
      Traditional-only character (``后面``, ``一起干杯``), unchecked when it has one (``命運若早已注定``: 注 -> 註 only
      swaps a variant);
    * in a song with kana only the lines with a certainly simplified character are listed (kanji-only Japanese lines
      are not).
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
        for sub, text in enumerate(row.get("lyrics") or []):
            if not isinstance(text, str):
                continue
            after = convert_line(text)
            if after is None:
                continue
            certain = looks_simplified(text, japanese)
            sure = sure or certain
            lines.append({"entry": row["entry"], "sub": sub, "before": text, "after": after, "_certain": certain})
    for index, text in headers:
        pair = convert_header(text)
        if pair is not None:
            certain = looks_simplified(pair[0], japanese)
            sure = sure or certain
            found.append({"line": index, "tag": text[1:3].lower(), "before": pair[0], "after": pair[1], "_certain": certain})
    if not sure:
        return {"count": 0, "optional": 0, "japanese": japanese, "uncertain": bool(uncertain), "lines": [], "headers": []}
    for items in (lines, found):
        for item in items:
            certain = item.pop("_certain")
            if japanese and not certain:
                continue
            checked = certain or not has_traditional_only(item["before"])
            item["default_checked"] = bool(checked and not uncertain)
            item["marks"] = _marks_for(item["before"])
    lines = [item for item in lines if "default_checked" in item]
    found = [item for item in found if "default_checked" in item]
    checked_count = sum(1 for item in lines + found if item["default_checked"])
    return {"count": checked_count, "optional": len(lines) + len(found) - checked_count,
            "japanese": japanese, "uncertain": bool(uncertain), "lines": lines, "headers": found}
