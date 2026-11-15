"""Simplified -> Traditional (Taiwan) detection for the review page.

OpenCC ``s2tw`` only (never ``s2twp``: it rewrites vocabulary). A line with any kana is Japanese and is never
converted. Without OpenCC every function answers "nothing to convert", so the review never depends on it.
"""
from __future__ import annotations

import re

_KANA_RE = re.compile("[぀-ヿ]")
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


def song_candidates(rows):
    """``{count, japanese, lines}`` for the payload rows, or None when OpenCC is missing.

    A song with no kana line at all is Chinese: its candidates start checked. In a song with kana the candidates
    may be Japanese kanji lines, so they start unchecked.
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
    for item in lines:
        item["default_checked"] = not japanese
    return {"count": len(lines), "japanese": japanese, "lines": lines}
