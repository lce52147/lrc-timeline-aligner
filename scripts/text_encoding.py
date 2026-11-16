"""Encoding detection for user-provided lyric / LRC files: UTF-8 (BOM optional), UTF-16 with BOM, Shift-JIS, Big5, GB18030.

Shared by the review page (review_server) and the alignment input reader (lyrics_io). A file that no codec reads cleanly, or whose
guess is a coin flip between legacy codecs, is reported as uncertain so the caller can refuse it instead of writing mojibake.
"""
from __future__ import annotations

class LrcDecodeError(ValueError):
    """The LRC bytes are not text in any encoding this tool reads (UTF-8, UTF-16 with BOM, Shift-JIS, GB18030, Big5)."""


def _legacy_score(text):
    """Characters outside ASCII, CJK, kana, fullwidth forms and common punctuation: a wrong legacy codec produces many."""
    return sum(1 for ch in text if not (ch < "\x80" or "\u3000" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                                        or "\uff00" <= ch <= "\uffef" or "\u2000" <= ch <= "\u206f"))


def _region_shares(raw):
    """(GB2312 level-1 share, Big5 frequent-char share) over the double-byte pairs in ``raw``, and the pair count."""
    gb = big5 = pairs = 0
    i, n = 0, len(raw)
    while i < n - 1:
        lead, trail = raw[i], raw[i + 1]
        if lead < 0x81:
            i += 1
            continue
        pairs += 1
        gb += 0xB0 <= lead <= 0xD7 and 0xA1 <= trail <= 0xFE
        big5 += 0xA4 <= lead <= 0xC6 and (0x40 <= trail <= 0x7E or 0xA1 <= trail <= 0xFE)
        i += 2
    if not pairs:
        return 0.0, 0.0, 0
    return gb / pairs, big5 / pairs, pairs


def _fullwidth_kana(text):
    return sum(1 for ch in text if "ぁ" <= ch <= "ゖ" or "ァ" <= ch <= "ヺ" or ch == "ー")


def _halfwidth_kana(text):
    return sum(1 for ch in text if "｡" <= ch <= "ﾟ")


_UNCERTAIN_MARGIN = 0.25
_UNCERTAIN_MIN_PAIRS = 4


def _decode_lrc_checked(raw):
    """(text, encoding, uncertain): ``uncertain`` when Big5 and GB18030 both fit and the byte regions barely tell them apart."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), "utf-16", False
        except UnicodeDecodeError:
            raise LrcDecodeError("utf-16 BOM but the bytes are not valid UTF-16") from None
    try:
        return raw.decode("utf-8-sig"), "utf-8", False
    except UnicodeDecodeError:
        pass
    found = []
    for codec in ("cp950", "gb18030"):
        try:
            found.append((_legacy_score(raw.decode(codec)), len(found), codec, raw.decode(codec)))
        except UnicodeDecodeError:
            continue
    if not found:
        raise LrcDecodeError("not UTF-8, UTF-16, Shift-JIS, Big5 or GB18030")
    best = min(found)
    uncertain = False
    if len(found) == 2 and found[0][0] == found[1][0]:
        gb_share, big5_share, pairs = _region_shares(raw)
        if gb_share > big5_share:
            best = found[1]
        uncertain = pairs < _UNCERTAIN_MIN_PAIRS or abs(gb_share - big5_share) < _UNCERTAIN_MARGIN
    _score, _order, codec, text = best
    try:
        jis = raw.decode("cp932")
    except UnicodeDecodeError:
        jis = None
    if jis is not None and _fullwidth_kana(jis) >= 2 and not _halfwidth_kana(jis) and not _fullwidth_kana(text):
        return jis, "cp932", False
    if jis is not None and not _halfwidth_kana(jis) and _fullwidth_kana(jis) < 2 and any(
            "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in jis):
        uncertain = True   # a Shift-JIS file with almost no kana also reads cleanly as cp932: the guess may be mojibake
    return text, codec, uncertain


def _decode_lrc(raw):
    """(text, encoding) for LRC bytes: UTF-8 (BOM optional), UTF-16 with BOM, then Big5 / GB18030 (the better fit)."""
    text, codec, _uncertain = _decode_lrc_checked(raw)
    return text, codec


def lrc_encoding_uncertain(raw):
    try:
        return _decode_lrc_checked(raw)[2]
    except LrcDecodeError:
        return False


class LyricEncodingError(ValueError):
    """A user-provided lyric file whose encoding is unreadable or too uncertain to trust; the message is Chinese and tells the user what to do."""


def read_lyric_text(path):
    """(text, encoding) of a user-provided .lrc / .txt: certain encodings are decoded, anything else is refused with ``LyricEncodingError``.

    UTF-8 (BOM optional), UTF-16 with BOM, Shift-JIS, Big5 and GB18030 are read; a file no codec reads cleanly, or whose legacy codec is a
    coin flip (``uncertain``), is not guessed at: decoding it wrongly would be aligned and written as a mojibake LRC.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        text, codec, uncertain = _decode_lrc_checked(raw)
    except LrcDecodeError:
        raise LyricEncodingError(
            f"無法判斷歌詞檔 {path} 的文字編碼（不是 UTF-8、UTF-16、Shift-JIS、Big5 或 GB18030）。"
            "請用記事本或編輯器另存成 UTF-8 後再試。"
        ) from None
    if uncertain:
        raise LyricEncodingError(
            f"歌詞檔 {path} 的編碼無法確定，可能是 Big5（繁體）、GB18030／GBK（簡體）或 Shift-JIS（日文）。"
            "為避免產生亂碼，已拒絕讀取；請用記事本或編輯器另存成 UTF-8 後再試。"
        )
    return text, codec
