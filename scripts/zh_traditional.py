"""Simplified -> Traditional (Taiwan) detection for the review page.

OpenCC ``s2tw`` only (never ``s2twp``: it rewrites vocabulary). A line with any kana is Japanese and is never
converted. Without OpenCC every function answers "nothing to convert", so the review never depends on it.
"""
from __future__ import annotations

import re

# Hiragana, katakana (not the middle dot U+30FB or the long-vowel mark U+30FC: Chinese lyrics use both), katakana extensions and halfwidth katakana.
_KANA_RE = re.compile(r"[぀-ヺヽ-ヿㇰ-ㇿｦ-ﾟ]")
# ID tags whose text is prose: [ti:] [ar:] [al:] [au:] [by:] [re:] [ve:]. Only the text between the colon and the closing bracket is converted.
HEADER_TEXT_RE = re.compile(r"^(\[(?:ti|ar|al|au|by|re|ve):)(.*?)(\][ 	]*)$", re.I | re.S)
_converter = None
_loaded = False


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


def convert_line(text):
    """The s2tw text of a non-kana line when it differs from ``text``; None otherwise."""
    if not text or has_kana(text):
        return None
    cc = converter()
    if cc is None:
        return None
    after = cc.convert(text)
    return after if after != text else None


def convert_header(text):
    """``(before, after)`` of the text inside an ID tag line (``[ti:...]``) when it converts; None otherwise (same kana rule as a lyric line)."""
    match = HEADER_TEXT_RE.match(text.rstrip("\r"))
    if not match:
        return None
    after = convert_line(match.group(2))
    return (match.group(2), after) if after is not None else None


def song_candidates(rows, headers=(), uncertain=False):
    """``{count, japanese, uncertain, lines, headers}`` for the payload rows, or None when OpenCC is missing.

    ``headers`` are ``(line_index, text)`` of the ID tag lines. A song with no kana line at all is Chinese: its
    candidates (lyric and header lines alike) start checked. In a song with kana the candidates may be Japanese kanji
    lines, so they start unchecked; so do all of them when the encoding guess is uncertain (the text may be mojibake).
    """
    if converter() is None:
        return None
    lines = []
    japanese = False
    for row in rows:
        lyrics = [item for item in (row.get("lyrics") or []) if isinstance(item, str)]
        japanese = japanese or any(has_kana(item) for item in lyrics)
        converted = [convert_line(item) for item in lyrics]
        if any(after is not None for after in converted):
            after = [new if new is not None else old for old, new in zip(lyrics, converted)]
            lines.append({"entry": row["entry"], "before": "\n".join(lyrics), "after": "\n".join(after)})
    found = []
    for index, text in headers:
        pair = convert_header(text)
        if pair is not None:
            found.append({"line": index, "tag": text[1:3].lower(), "before": pair[0], "after": pair[1]})
    for item in lines + found:
        item["default_checked"] = not japanese and not uncertain
    return {"count": len(lines) + len(found), "japanese": japanese, "uncertain": bool(uncertain), "lines": lines, "headers": found}
