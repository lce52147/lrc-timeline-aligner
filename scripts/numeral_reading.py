#!/usr/bin/env python3
"""Spoken readings of Arabic numerals for alignment text only.

Lyrics keep the numerals as written ("1秒前"); the forced aligners cannot place
digits, so the text handed to them has the digits replaced by kanji numerals,
which every Japanese reader (UniDic, pykakasi) turns into the sung reading.
"""

from __future__ import annotations

import re
import unicodedata

_KANJI_DIGITS = "〇一二三四五六七八九"
_SMALL_UNITS = ("", "十", "百", "千")
_JAPANESE_RE = re.compile(r"[぀-ヿ㐀-鿿]")
_DIGIT_RUN_RE = re.compile(r"(?<![0-9.])[0-9]+(?![0-9]|\.[0-9])")


def _group_reading(value: int) -> str:
    out = []
    for position in (3, 2, 1, 0):
        digit = (value // 10**position) % 10
        if not digit:
            continue
        if position and digit == 1:
            out.append(_SMALL_UNITS[position])
        else:
            out.append(_KANJI_DIGITS[digit] + _SMALL_UNITS[position])
    return "".join(out)


def number_to_kanji(digits: str) -> str:
    """One run of ASCII digits as kanji; long or zero-padded runs digit by digit."""
    if len(digits) > 4 or (len(digits) > 1 and digits[0] == "0"):
        return "".join(_KANJI_DIGITS[int(char)] for char in digits)
    value = int(digits)
    return _group_reading(value) if value else _KANJI_DIGITS[0]


def numerals_for_alignment(text: str) -> str:
    """Replace Arabic numerals with kanji numerals when the line is Japanese.

    Lines without kana/kanji (English lyrics) are returned unchanged, so a
    Latin-script aligner is never fed a Japanese reading of "3".
    """
    normalized = unicodedata.normalize("NFKC", text)
    if not _JAPANESE_RE.search(normalized) or not re.search(r"[0-9]", normalized):
        return text
    return _DIGIT_RUN_RE.sub(lambda match: number_to_kanji(match.group(0)), normalized)
