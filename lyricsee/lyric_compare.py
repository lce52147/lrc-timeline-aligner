"""Notation-aware comparison of two transcriptions of the same lyrics.

Two published copies of one song rarely match byte for byte: one site writes
二人, another ふたり; one keeps ラララ fills, another drops them; punctuation,
full/half-width forms and katakana/hiragana vary.  None of those is a wrong
lyric.  黒い過去 against 暗い過去 is.  This module aligns two lyric texts
line by line and splits their differences into

* ``notation`` differences (kana vs kanji spelling, katakana vs hiragana,
  numerals, romanisation, vocal fills such as la la la / ah / ラララ, extra or
  missing parenthesised backing vocals, punctuation, width, old-form kanji)
* ``real`` differences (a different word, a missing or extra lyric line)

Only real differences count against cross-source agreement.  Everything here
is pure text processing with no network access.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import NamedTuple

# Katakana ァ..ヶ (U+30A1..U+30F6) sit exactly 0x60 above hiragana ぁ..ゖ.
_KATAKANA_TO_HIRAGANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}
# Small vowel kana are a stylistic elongation (あぁ / あ, ねぇ / ねえ).
_SMALL_VOWELS = str.maketrans("ぁぃぅぇぉゕゖ", "あいうえおかけ")
# Old-form / variant kanji that sites use interchangeably.
_VARIANT_KANJI = str.maketrans({
    "繫": "繋", "塡": "填", "繡": "繍", "吞": "呑", "噓": "嘘", "靜": "静",
    "效": "効", "澁": "渋", "敎": "教", "淚": "涙", "說": "説", "讀": "読",
    "國": "国", "惡": "悪", "應": "応", "會": "会", "來": "来", "變": "変",
    "體": "体", "眞": "真", "圖": "図", "聲": "声", "靑": "青", "櫻": "桜",
    "樂": "楽", "齒": "歯", "讓": "譲", "戀": "恋", "爲": "為", "爭": "争",
    "黑": "黒", "德": "徳", "搖": "揺", "遙": "遥", "瀨": "瀬", "淨": "浄",
    "劍": "剣", "晝": "昼", "晚": "晩", "燈": "灯", "畫": "画", "賣": "売",
    "濱": "浜", "燒": "焼", "穗": "穂", "莊": "荘", "臺": "台", "廳": "庁",
    "顏": "顔", "飮": "飲", "鬪": "闘", "鷄": "鶏", "歎": "嘆", "槇": "槙",
    "嶽": "岳", "區": "区", "單": "単", "嚴": "厳", "圓": "円", "壞": "壊",
    "將": "将", "專": "専", "從": "従", "徵": "徴", "惠": "恵", "收": "収",
    "曆": "暦", "歷": "歴", "殘": "残", "氣": "気", "溫": "温", "滯": "滞",
    "濕": "湿", "盜": "盗", "稻": "稲", "粹": "粋", "絲": "糸", "繩": "縄",
    "續": "続", "總": "総", "緖": "緒", "藏": "蔵", "處": "処", "衞": "衛",
    "譽": "誉", "譯": "訳", "豐": "豊", "轉": "転", "醉": "酔", "錢": "銭",
    "關": "関", "隱": "隠", "雜": "雑", "驛": "駅", "髮": "髪", "龜": "亀",
    "卽": "即", "勳": "勲", "彈": "弾", "戰": "戦", "拜": "拝", "揭": "掲",
    "擧": "挙", "曉": "暁", "檢": "検", "歡": "歓", "滿": "満", "爐": "炉",
    "犧": "犠", "獨": "独", "當": "当", "禪": "禅", "經": "経", "繪": "絵",
    "聽": "聴", "蟲": "虫", "覽": "覧", "觸": "触", "讚": "讃", "辯": "弁",
    "鄰": "隣", "釋": "釈", "鐵": "鉄", "鑛": "鉱", "陷": "陥", "險": "険",
    "靈": "霊", "顯": "顕", "驅": "駆", "麥": "麦", "默": "黙", "齋": "斎",
    "齡": "齢", "龍": "竜", "弒": "弑", "慾": "欲", "舘": "館", "迄": "迄",
})
_PAREN_RE = re.compile(r"[（(][^（()）]*[)）]")
_LATIN_DIACRITIC_RE = re.compile(r"[À-ɏ]")
_NON_WORD_RE = re.compile(r"[^\w]+|_")
_KANA_RE = re.compile(r"[ぁ-ゟー]")
_KANJI_RE = re.compile(r"[㐀-鿿々〇]")
_LATIN_RE = re.compile(r"[a-z]")
_DIGIT_RE = re.compile(r"[0-9]")
_KANA_ONLY_RE = re.compile(r"^[ぁ-ゟー]+$")
_LATIN_ONLY_RE = re.compile(r"^[a-z]+$")
_DIGIT_ONLY_RE = re.compile(r"^[0-9]+$")
_KANJI_NUMERAL_RE = re.compile(r"^[〇零一二三四五六七八九十百千万億兆]+$")
_NUMBER_WORD_RE = re.compile(
    r"^(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"ひと|ふた|なな|ここの|とお|いち|さん|ろく|しち|はち|きゅう|じゅう|れい)+$"
)
# Vocal fills.  Matched against canonical text (no spaces, hiragana folded),
# so ラララ, la la la, ah ah, woo, yeah yeah and あぁ all collapse here.
_FILLER_RE = re.compile(
    r"^(?:la|na|ah|aah|oh|ooh|oo|woo|wow|wo|yeah|yea|yeh|uh|hey|hoo|hmm|mm|ha|da|dum|du|lu|pa|ra|"
    r"a|o|u|e|i|"
    r"はあ|へい|ほう|ふう|わお|うぉ|おう|いえ|"
    r"ら|な|あ|お|う|い|え|ん|わ|る|ぱ|だ|ー|ゔ|ど|ゃ|ゅ|ょ|っ)+$"
)
_FRAGMENT = 24


def _strip_latin_diacritics(match: re.Match[str]) -> str:
    decomposed = unicodedata.normalize("NFD", match.group())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _expand_iteration_marks(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "々" and out:
            out.append(out[-1])
        else:
            out.append(ch)
    return "".join(out)


def canonical_text(text: str, *, drop_parens: bool = True) -> str:
    """Fold spelling that does not change the lyric.

    NFKC width folding, case folding, katakana to hiragana, small vowel kana to
    full vowels, iteration marks, variant kanji, Latin diacritics and all
    punctuation / whitespace.  With ``drop_parens`` parenthesised backing
    vocals are removed as well.
    """
    folded = unicodedata.normalize("NFKC", text)
    if drop_parens:
        folded = _PAREN_RE.sub("", folded)
    folded = folded.casefold().translate(_KATAKANA_TO_HIRAGANA).translate(_SMALL_VOWELS)
    folded = _expand_iteration_marks(folded).translate(_VARIANT_KANJI)
    folded = _LATIN_DIACRITIC_RE.sub(_strip_latin_diacritics, folded)
    return _NON_WORD_RE.sub("", folded)


def is_filler(canonical: str, *, min_length: int = 2) -> bool:
    """True for empty text or pure vocal fills (ラララ, la la la, ah, woo).

    Spans shorter than ``min_length`` are not treated as fills so that a
    dropped particle (は, に) stays a real difference; whole lines use 1.
    """
    if not canonical:
        return True
    return len(canonical) >= min_length and bool(_FILLER_RE.match(canonical))


def _is_subsequence(needle: str, haystack: str) -> bool:
    position = 0
    for ch in needle:
        position = haystack.find(ch, position)
        if position < 0:
            return False
        position += 1
    return True


def _reading_plausible(kana: str, mixed: str) -> bool:
    """Could ``kana`` be ``mixed`` (kanji/digits plus kana) read out in kana?

    二人 / ふたり, 行く / ゆく and 1 / ひと pass; 此処に / ここで fails because
    the kana already present in the kanji spelling (に) must survive in the
    kana spelling.  Without a reading dictionary this checks shape only.
    """
    kanji_count = len(_KANJI_RE.findall(mixed)) + len(_DIGIT_RE.findall(mixed))
    kana_part = "".join(_KANA_RE.findall(mixed))
    if kanji_count == 0:
        return False
    if not _is_subsequence(kana_part, kana):
        return False
    return kanji_count + len(kana_part) <= len(kana) <= 5 * kanji_count + len(kana_part)


def classify_segment(a_segment: str, b_segment: str) -> str:
    """Classify one differing span of canonical text.

    Returns ``"real"`` or a ``"notation:<kind>"`` label.
    """
    if is_filler(a_segment) and is_filler(b_segment):
        return "notation:filler"
    if not a_segment or not b_segment:
        return "real"
    a_kana = bool(_KANA_ONLY_RE.match(a_segment))
    b_kana = bool(_KANA_ONLY_RE.match(b_segment))
    a_latin = bool(_LATIN_ONLY_RE.match(a_segment))
    b_latin = bool(_LATIN_ONLY_RE.match(b_segment))
    if a_kana and not b_kana and not _LATIN_RE.search(b_segment):
        return "notation:kana_kanji" if _reading_plausible(a_segment, b_segment) else "real"
    if b_kana and not a_kana and not _LATIN_RE.search(a_segment):
        return "notation:kana_kanji" if _reading_plausible(b_segment, a_segment) else "real"
    if (a_kana and b_latin) or (b_kana and a_latin):
        return "notation:transliteration"

    def numeral_like(segment: str) -> bool:
        return bool(
            _DIGIT_ONLY_RE.match(segment)
            or _KANJI_NUMERAL_RE.match(segment)
            or _NUMBER_WORD_RE.match(segment)
        )

    if numeral_like(a_segment) and numeral_like(b_segment):
        return "notation:numeral"
    return "real"


class LineDiff(NamedTuple):
    kind: str            # word | missing_in_a | missing_in_b | notation:<kind>
    a_line: int | None   # 1-based line numbers in each text (None when absent)
    b_line: int | None
    a_text: str          # short fragments, never the whole lyric
    b_text: str


class LyricDiff(NamedTuple):
    similarity: float          # SequenceMatcher ratio of canonical full texts
    a_lines: int               # non-filler lines in each text
    b_lines: int
    matched_lines: int         # equal or notation-equal line pairs
    real_diffs: list[LineDiff]
    notation_diffs: list[LineDiff]
    pairs: list[tuple[int, int]] = []   # 1-based (a_line, b_line) of matched pairs

    @property
    def word_diffs(self) -> int:
        return sum(1 for diff in self.real_diffs if diff.kind == "word")

    @property
    def coverage_a(self) -> float:
        return self.matched_lines / self.a_lines if self.a_lines else 0.0

    @property
    def coverage_b(self) -> float:
        return self.matched_lines / self.b_lines if self.b_lines else 0.0

    def agrees(self, *, min_coverage: float = 0.75) -> bool:
        """Same lyric: no word-level real difference and most lines matched."""
        return (
            self.matched_lines > 0
            and self.word_diffs == 0
            and self.coverage_a >= min_coverage
            and self.coverage_b >= min_coverage
        )

    def summary(self) -> dict:
        return {
            "similarity": round(self.similarity, 3),
            "lines": [self.a_lines, self.b_lines],
            "matched_lines": self.matched_lines,
            "coverage": [round(self.coverage_a, 3), round(self.coverage_b, 3)],
            "real_diff_lines": len(self.real_diffs),
            "word_diff_lines": self.word_diffs,
            "notation_diff_lines": len(self.notation_diffs),
            "agrees": self.agrees(),
        }


def _fragment(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _FRAGMENT else text[: _FRAGMENT - 1] + "…"


class _Line(NamedTuple):
    number: int
    text: str
    core: str   # canonical without parentheses
    full: str   # canonical with parentheses kept


def _prepare(lines: list[str]) -> list[_Line]:
    prepared: list[_Line] = []
    for number, text in enumerate(lines, start=1):
        core = canonical_text(text, drop_parens=True)
        full = canonical_text(text, drop_parens=False)
        if not full:
            continue
        if not core:
            core = full  # a line that is only "(ah ah)" keeps its content
        prepared.append(_Line(number, text, core, full))
    return prepared


def _span_labels(a_text: str, b_text: str) -> list[tuple[str, str, str]]:
    """(label, a_span, b_span) for every differing span of two canonical strings."""
    if a_text == b_text:
        return []
    matcher = difflib.SequenceMatcher(None, a_text, b_text, autojunk=False)
    labels = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a_span, b_span = a_text[i1:i2], b_text[j1:j2]
        labels.append((classify_segment(a_span, b_span), a_span, b_span))
    return labels


def _compare_pair(a: _Line, b: _Line) -> list[tuple[str, str, str]]:
    """Labels of all differing spans between two lines.

    One site writes backing vocals in parentheses, another inline, a third
    drops them.  The pair is therefore compared with parentheses kept and
    dropped on either side, and the reading with the fewest real differences
    wins; needing to drop them is itself a notation difference.
    """
    if a.full == b.full:
        return []
    best: list[tuple[str, str, str]] | None = None
    best_key = None
    for a_text, b_text, dropped in (
        (a.full, b.full, False),
        (a.core, b.core, True),
        (a.core, b.full, True),
        (a.full, b.core, True),
    ):
        labels = _span_labels(a_text, b_text)
        if dropped and a.core == a.full and b.core == b.full:
            continue
        key = (sum(1 for label, _, _ in labels if label == "real"), len(labels))
        if best_key is None or key < best_key:
            best, best_key = labels, key
            if dropped and not any(label == "real" for label, _, _ in labels):
                best = labels + [("notation:parenthesised", "", "")]
            if key[0] == 0 and key[1] == 0:
                break
    return best or []


def compare_lyrics(a_lines: list[str], b_lines: list[str]) -> LyricDiff:
    """Align two lyric texts and classify every difference."""
    a_prepared = _prepare(list(a_lines))
    b_prepared = _prepare(list(b_lines))
    a_keys = [line.core for line in a_prepared]
    b_keys = [line.core for line in b_prepared]
    similarity = difflib.SequenceMatcher(
        None,
        "".join(line.full for line in a_prepared),
        "".join(line.full for line in b_prepared),
        autojunk=False,
    ).ratio()

    real: list[LineDiff] = []
    notation: list[LineDiff] = []
    pairs: list[tuple[int, int]] = []
    matched = 0

    def record_pair(a: _Line, b: _Line) -> None:
        nonlocal matched
        labels = _compare_pair(a, b)
        real_spans = [(a_span, b_span) for label, a_span, b_span in labels if label == "real"]
        if real_spans:
            a_span, b_span = real_spans[0]
            real.append(LineDiff("word", a.number, b.number, a_span[:_FRAGMENT], b_span[:_FRAGMENT]))
            return
        pairs.append((a.number, b.number))
        if not is_filler(a.core, min_length=1) or not is_filler(b.core, min_length=1):
            matched += 1
        if labels:
            kinds = sorted({label.split(":", 1)[1] for label, _, _ in labels})
            notation.append(
                LineDiff("notation:" + "+".join(kinds), a.number, b.number, _fragment(a.text), _fragment(b.text))
            )

    def record_missing(line: _Line, side: str) -> None:
        if is_filler(line.core, min_length=1):
            notation.append(
                LineDiff(
                    "notation:filler",
                    line.number if side == "b" else None,
                    line.number if side == "a" else None,
                    _fragment(line.text) if side == "b" else "",
                    _fragment(line.text) if side == "a" else "",
                )
            )
            return
        if side == "b":   # present in a, missing in b
            real.append(LineDiff("missing_in_b", line.number, None, _fragment(line.text), ""))
        else:
            real.append(LineDiff("missing_in_a", None, line.number, "", _fragment(line.text)))

    matcher = difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                record_pair(a_prepared[i1 + offset], b_prepared[j1 + offset])
            continue
        a_block = a_prepared[i1:i2]
        b_block = b_prepared[j1:j2]
        if tag == "replace":
            # Pair the lines inside the block so a single differing word stays
            # a word difference instead of one missing line plus one extra
            # line: a notation-only pair wins outright, otherwise the most
            # similar line (at least half its characters shared).
            unused_b = list(range(len(b_block)))
            for a_line in a_block:
                best_index, best_key = None, None
                for index in unused_b:
                    other = b_block[index]
                    labels = _compare_pair(a_line, other)
                    real_count = sum(1 for label, _, _ in labels if label == "real")
                    ratio = difflib.SequenceMatcher(None, a_line.core, other.core, autojunk=False).ratio()
                    if real_count and ratio < 0.5:
                        continue
                    key = (real_count > 0, -ratio)
                    if best_key is None or key < best_key:
                        best_index, best_key = index, key
                if best_index is not None:
                    unused_b.remove(best_index)
                    record_pair(a_line, b_block[best_index])
                else:
                    record_missing(a_line, "b")
            for index in unused_b:
                record_missing(b_block[index], "a")
            continue
        for a_line in a_block:
            record_missing(a_line, "b")
        for b_line in b_block:
            record_missing(b_line, "a")

    a_count = sum(1 for line in a_prepared if not is_filler(line.core, min_length=1))
    b_count = sum(1 for line in b_prepared if not is_filler(line.core, min_length=1))
    return LyricDiff(similarity, a_count, b_count, matched, real, notation, pairs)


def diff_lines_for_report(diff: LyricDiff, *, limit: int = 20) -> list[dict]:
    """Real differences as JSON-friendly rows (line numbers and fragments)."""
    rows = []
    for item in diff.real_diffs[:limit]:
        rows.append({
            "kind": item.kind,
            "line": item.a_line,
            "other_line": item.b_line,
            "text": item.a_text,
            "other_text": item.b_text,
        })
    return rows
