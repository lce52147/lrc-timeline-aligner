"""Clean lyric candidates and compare them character by character.

Pure functions, no I/O: used by candidate_review (similarity between
candidates) and autopick (match against the ASR transcript).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿ]")
_CJK_RE = re.compile(r"[㐀-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_CREDIT_RE = re.compile(
    r"^\s*(作詞|作词|作曲|編曲|编曲|歌手|歌|詞|词|曲|唄|vocals?|lyrics?|lyricist|music|composer|"
    r"arranger?|words|written\s+by|translated?|翻譯|翻译|譯|译|製作|制作|原唱|演唱|"
    r"出品|專輯|专辑|album|artist|title|singer|cv)\s*[:：/／]",
    re.IGNORECASE,
)
_HEADER_WORD_RE = re.compile(r"歌詞|歌词|lyrics|https?:|www\.|@\w|translat|romaji|ローマ字|羅馬|罗马", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"^[\[\(（【<〈《].{0,24}[\]\)）】>〉》]$",
)
_SECTION_WORD_RE = re.compile(
    r"verse|chorus|bridge|intro|outro|hook|pre-?chorus|instrumental|サビ|[A-Dァ-ヶ]メロ|間奏|前奏|後奏|"
    r"间奏|前奏|尾奏|x\s*\d|×\s*\d|\d\s*[x×]|repeat",
    re.IGNORECASE,
)


def _line_kind(line: str) -> str:
    """Script class of one line: kana (Japanese), latin, cjk (kanji/hanzi only) or other."""
    if _KANA_RE.search(line):
        return "kana"
    latin = len(_LATIN_RE.findall(line))
    cjk = len(_CJK_RE.findall(line))
    if latin and latin >= cjk:
        return "latin"
    if cjk:
        return "cjk"
    return "other"


def detect_language(lines: list[str]) -> str:
    """'ja' | 'en' | 'zh' from the script mix of the lines (kana lines >= 25% means Japanese)."""
    counts = Counter(_line_kind(unicodedata.normalize("NFKC", line)) for line in lines if line.strip())
    total = counts["kana"] + counts["latin"] + counts["cjk"]
    if not total:
        return "ja"
    if counts["kana"] >= 0.25 * total:
        return "ja"
    return "en" if counts["latin"] >= counts["cjk"] else "zh"


def clean_lyric_lines(lines: list[str], lang: str | None = None) -> list[str]:
    """Keep only the original-language lyric lines.

    Drops empty lines, credits/headers, section markers, and the lines that are
    in another script than the song (Chinese translation between Japanese
    lines, romaji). lang: 'ja' | 'en' | 'zh'; detected from the lines if None.
    """
    lang = lang or detect_language(lines)
    kept: list[str] = []
    for raw in lines:
        line = unicodedata.normalize("NFKC", raw).strip()
        if not line:
            continue
        if _CREDIT_RE.match(line) or _HEADER_WORD_RE.search(line):
            continue
        if _SECTION_RE.match(line) and _SECTION_WORD_RE.search(line):
            continue
        kind = _line_kind(line)
        if lang == "ja" and kind != "kana":
            continue
        if lang == "en" and kind != "latin":
            continue
        if lang == "zh" and kind != "cjk":
            continue
        kept.append(line)
    return kept


def fold(text: str) -> str:
    """Comparison form: NFKC, case-folded, katakana->hiragana, letters/digits only."""
    text = unicodedata.normalize("NFKC", text).casefold()
    out: list[str] = []
    for ch in text:
        if "ァ" <= ch <= "ヶ":
            ch = chr(ord(ch) - 0x60)
        if ch in "ー〜~":
            continue
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def _bigrams(text: str) -> Counter:
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


def char_similarity(a: str, b: str) -> float:
    """Dice coefficient over the distinct character bigrams of two folded texts.

    Order-insensitive and blind to how often a chorus is repeated, so a
    candidate that writes the chorus once still agrees with one that writes it
    three times.
    """
    sa, sb = set(_bigrams(a)), set(_bigrams(b))
    if not sa or not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


def asr_match(candidate: str, transcript: str) -> dict[str, float]:
    """How well a folded candidate matches a folded ASR transcript of the audio.

    precision: share of the candidate's bigrams that were really sung (multiset);
    coverage: share of the distinct bigrams in the transcript the candidate explains
    (set-based, so a candidate that writes a repeated chorus once is not punished);
    score: their harmonic mean. A same-title wrong song or a snippet scores low.
    """
    cc, ca = _bigrams(candidate), _bigrams(transcript)
    if not cc or not ca:
        return {"precision": 0.0, "coverage": 0.0, "score": 0.0}
    precision = sum((cc & ca).values()) / sum(cc.values())
    coverage = len(set(ca) & set(cc)) / len(set(ca))
    score = 0.0 if precision + coverage == 0 else 2 * precision * coverage / (precision + coverage)
    return {"precision": round(precision, 4), "coverage": round(coverage, 4), "score": round(score, 4)}
