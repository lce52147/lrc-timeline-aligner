"""Shared Japanese reading + Hepburn romaji for the LRC tools core.

One place for kana readings and romaji, used by the report pipeline, the
aligner (compact form) and the review page (spaced form).  Reading comes from
fugashi + unidic (morphological analysis, so 解って -> wakatte, 君 -> kimi).
When fugashi/unidic is unavailable it falls back to pykakasi plus safe
post-fixes (small っ geminates the next consonant, particles は/へ/を).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

READING_SYSTEM_MORPH = "unidic-hepburn-v1"
READING_SYSTEM_FALLBACK = "pykakasi-hepburn-postfix-v1"

_KANA = "[ぁ-ゟァ-ヿㇰ-ㇿｦ-ﾟ]"
_KANA_RE = re.compile(_KANA)
_CJK_RE = re.compile("[㐀-鿿]")

_BASE = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヰ": "i", "ヱ": "e", "ヲ": "wo", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ヮ": "wa", "ヵ": "ka", "ヶ": "ke",
}
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_SMALL_V = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_YOON_STEM = {
    "キ": "ky", "ギ": "gy", "シ": "sh", "ジ": "j", "チ": "ch", "ヂ": "j", "ニ": "ny",
    "ヒ": "hy", "ビ": "by", "ピ": "py", "ミ": "my", "リ": "ry", "ヴ": "v",
}
# Small vowel after a consonant kana (loan words): ファ fa, ティ ti, ウィ wi ...
_LOAN = {
    ("フ", "ァ"): "fa", ("フ", "ィ"): "fi", ("フ", "ェ"): "fe", ("フ", "ォ"): "fo",
    ("ウ", "ィ"): "wi", ("ウ", "ェ"): "we", ("ウ", "ォ"): "wo",
    ("テ", "ィ"): "ti", ("デ", "ィ"): "di", ("ト", "ゥ"): "tu", ("ド", "ゥ"): "du",
    ("シ", "ェ"): "she", ("ジ", "ェ"): "je", ("チ", "ェ"): "che", ("ツ", "ァ"): "tsa", ("ツ", "ィ"): "tsi",
    ("ツ", "ェ"): "tse", ("ツ", "ォ"): "tso", ("ヴ", "ァ"): "va", ("ヴ", "ィ"): "vi", ("ヴ", "ェ"): "ve",
    ("ヴ", "ォ"): "vo", ("イ", "ェ"): "ye",
}
_PARTICLE_ROMA = {"は": "wa", "へ": "e", "を": "wo"}


def _to_katakana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096 or 0x309D <= code <= 0x309E:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def kana_to_romaji(kana: str) -> str:
    """Hepburn romaji of a kana string.  っ doubles the next consonant, ー repeats the vowel."""
    text = unicodedata.normalize("NFKC", _to_katakana(kana))
    out: list[str] = []
    geminate = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        piece = None
        used = 1
        if ch == "ッ":
            geminate = True
            i += 1
            continue
        if ch == "ー":
            for back in reversed(out):
                vowel = next((c for c in reversed(back) if c in "aeiou"), "")
                if vowel:
                    piece = vowel
                    break
            if piece is None:
                i += 1
                continue
        elif (ch, nxt) in _LOAN:
            piece, used = _LOAN[(ch, nxt)], 2
        elif nxt in _SMALL_Y and ch in _YOON_STEM:
            piece, used = _YOON_STEM[ch] + _SMALL_Y[nxt], 2
        elif nxt in _SMALL_Y and ch in ("イ",):
            piece, used = "y" + _SMALL_Y[nxt], 2
        elif nxt in _SMALL_V and ch in _YOON_STEM:
            piece, used = _YOON_STEM[ch] + _SMALL_V[nxt], 2
        elif ch in _BASE:
            piece = _BASE[ch]
        elif ch in _SMALL_Y:
            piece = "y" + _SMALL_Y[ch]
        elif ch.isascii() and ch.isalnum():
            piece = ch.lower()
        else:
            geminate = False
            i += 1
            continue
        if geminate:
            if piece and piece[0] not in "aeiouny":
                piece = ("t" if piece.startswith("ch") else piece[0]) + piece
            elif piece and piece[0] in "aeiou":
                piece = "t" + piece  # stray っ before a vowel: glottal stop, written t
            geminate = False
        out.append(piece)
        i += used
    return "".join(out)


@lru_cache(maxsize=1)
def _tagger():
    try:
        from fugashi import Tagger

        return Tagger()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _kakasi():
    try:
        import pykakasi

        return pykakasi.kakasi()
    except Exception:
        return None


def analyzer_available() -> bool:
    return _tagger() is not None


def reading_system() -> str:
    return READING_SYSTEM_MORPH if analyzer_available() else READING_SYSTEM_FALLBACK


def _feat(token, name):
    value = getattr(token.feature, name, None)
    return value if isinstance(value, str) and value != "*" else ""


def _morph_words(text: str):
    """Yield (kana, is_latin) words with conjugation chains joined."""
    words: list[list] = []  # [kana, pos1, kind]
    prev_pos = ""
    prev_pos2 = ""
    tokens = list(_tagger()(text))
    for index, token in enumerate(tokens):
        surface = token.surface
        following = tokens[index + 1].surface if index + 1 < len(tokens) else ""
        kana = _feat(token, "kana")
        pos1 = _feat(token, "pos1")
        pos2 = _feat(token, "pos2")
        lemma = _feat(token, "lemma")
        if surface in _PARTICLE_ROMA and pos1 == "助詞":
            kana = {"は": "ワ", "へ": "エ", "を": "ヲ"}[surface]
        elif kana in ("コンニチハ", "コンバンハ"):
            kana = kana[:-1] + "ワ"
        elif surface == "私" and kana == "ワタクシ":
            kana = "ワタシ"
        elif surface == "明日" and kana == "アス":
            kana = "アシタ"
        elif surface == "何" and kana == "ナン" and following[:1] in ("が", "も", "を", "か", "に", "よ", "て", "は"):
            kana = "ナニ"
        elif not kana:
            if _KANA_RE.search(surface) or _CJK_RE.search(surface):
                kana = _fallback_kana(surface)
            else:
                kana = surface
        join = False
        if words:
            if pos1 == "助動詞" and prev_pos in ("動詞", "形容詞", "助動詞") and not lemma.startswith(("だ", "です")):
                join = True
            elif pos1 == "助詞" and pos2 == "接続助詞" and surface in ("て", "で", "ちゃ", "ば") and prev_pos in ("動詞", "形容詞", "助動詞"):
                join = True
            elif pos1 == "接尾辞" and prev_pos in ("動詞", "形容詞", "助動詞", "接尾辞"):
                join = True
            elif pos1 == "動詞" and pos2 == "非自立可能" and prev_pos in ("動詞", "助詞", "助動詞") and words[-1][0].endswith(("テ", "デ", "ッ")):
                join = True
            elif words[-1][0].endswith("ッ"):
                join = True
            elif pos1 == "接尾辞" and prev_pos in ("名詞", "代名詞", "接頭辞"):
                join = True
            elif prev_pos == "接頭辞" or (pos1 == "名詞" and (pos2 == "助数詞" or (pos2 == "数詞" and prev_pos2 == "数詞"))):
                join = True
            elif words[-1][0] == "ン" and pos1 == "助動詞":
                join = True
            elif surface == "の" and pos2 == "準体助詞" and words[-1][0] == "ナ":
                join = True
            elif surface in ("に", "で") and words[-1][0] == "ナノ":
                join = True
        if join:
            words[-1][0] += kana
            words[-1][1] = pos1 or prev_pos
        else:
            words.append([kana, pos1])
        prev_pos = pos1
        prev_pos2 = pos2
    return [w[0] for w in words]


def _fallback_kana(text: str) -> str:
    conv = _kakasi()
    if conv is None:
        return ""
    try:
        return "".join(str(item.get("hira", "")) for item in conv.convert(text) if isinstance(item, dict))
    except Exception:
        return ""


def _fallback_words(text: str):
    conv = _kakasi()
    if conv is None:
        return []
    words: list[str] = []
    for item in conv.convert(text):
        if not isinstance(item, dict):
            continue
        orig = str(item.get("orig", ""))
        hira = str(item.get("hira", "")) or orig
        if orig in _PARTICLE_ROMA:
            hira = {"は": "わ", "へ": "え", "を": "ヲ"}[orig]
        if words and words[-1].endswith(("っ", "ッ")):
            words[-1] += hira
        else:
            words.append(hira)
    return words


def romanize_words(text: str, phonetic: bool = False) -> list[str]:
    """Spaced Hepburn words for a lyric line ('' list when nothing readable)."""
    text = unicodedata.normalize("NFKC", text or "")
    if not text.strip():
        return []
    if _tagger() is not None:
        try:
            words = _morph_words(text)
        except Exception:
            words = _fallback_words(text)
    else:
        words = _fallback_words(text)
    out = []
    for word in words:
        if phonetic:
            word = word.replace("ヲ", "オ")
        roma = re.sub(r"[^a-z0-9]+", "", kana_to_romaji(word))
        if roma:
            out.append(roma)
    return out


def romanize_line(text: str) -> str:
    """Display romaji: lowercase Hepburn, words separated by single spaces."""
    return " ".join(romanize_words(text))


def romanize_compact(text: str) -> str:
    """Alignment romaji: the same reading with the spaces removed."""
    return "".join(romanize_words(text, phonetic=True))


def has_japanese(text: str) -> bool:
    return bool(_KANA_RE.search(text or "") or _CJK_RE.search(text or ""))
