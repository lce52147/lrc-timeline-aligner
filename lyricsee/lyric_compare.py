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
    matched_lines: int         # non-filler lines of a without any real difference
    real_diffs: list[LineDiff]
    notation_diffs: list[LineDiff]
    pairs: list[tuple[int, int]] = []   # 1-based (a_line, b_line) of lines that map 1:1
    matched_b_lines: int = 0            # non-filler lines of b without any real difference

    @property
    def word_diffs(self) -> int:
        return sum(1 for diff in self.real_diffs if diff.kind == "word")

    @property
    def coverage_a(self) -> float:
        return self.matched_lines / self.a_lines if self.a_lines else 0.0

    @property
    def coverage_b(self) -> float:
        return self.matched_b_lines / self.b_lines if self.b_lines else 0.0

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
    canon: str   # canonical text, parentheses content kept
    start: int   # offset of canon inside the joined text
    end: int


def _prepare(lines: list[str]) -> tuple[list[_Line], str, set[str]]:
    """Canonical lines with their offsets in the joined text, plus the
    canonical forms of every parenthesised group (backing vocals)."""
    prepared: list[_Line] = []
    parens: set[str] = set()
    offset = 0
    for number, text in enumerate(lines, start=1):
        canon = canonical_text(text, drop_parens=False)
        if not canon:
            continue
        prepared.append(_Line(number, text, canon, offset, offset + len(canon)))
        offset += len(canon)
        for group in _PAREN_RE.findall(unicodedata.normalize("NFKC", text)):
            inner = canonical_text(group, drop_parens=False)
            if inner:
                parens.add(inner)
    return prepared, "".join(line.canon for line in prepared), parens


def _lines_in_span(lines: list[_Line], start: int, end: int) -> list[tuple[_Line, bool]]:
    """Lines overlapping [start, end) with a flag for 'entirely inside'."""
    if start == end:
        # an insertion point: attribute it to the line that contains it
        for line in lines:
            if line.start <= start < line.end:
                return [(line, False)]
        return [(lines[-1], False)] if lines else []
    result = []
    for line in lines:
        if line.end <= start or line.start >= end:
            continue
        result.append((line, line.start >= start and line.end <= end))
    return result


def compare_lyrics(a_lines: list[str], b_lines: list[str]) -> LyricDiff:
    """Align two lyric texts character by character and classify every
    difference, mapping it back to line numbers.

    Alignment is done on the joined canonical texts, so two copies that only
    break lines differently (one long line against two short ones) still
    agree.  A span that one side lacks counts as a missing line when it
    covers a whole line of the other side, as a vocal fill when it is one,
    and as a parenthesised backing vocal when the other side has it in
    brackets.
    """
    a_prepared, a_text, a_parens = _prepare(list(a_lines))
    b_prepared, b_text, b_parens = _prepare(list(b_lines))
    matcher = difflib.SequenceMatcher(None, a_text, b_text, autojunk=False)
    similarity = matcher.ratio() if a_text and b_text else 0.0

    real: list[LineDiff] = []
    notation: dict[tuple[int | None, int | None], set[str]] = {}
    real_a: set[int] = set()
    real_b: set[int] = set()
    seen_real: set[tuple] = set()

    def add_real(kind: str, a_line, b_line, a_frag: str, b_frag: str) -> None:
        key = (kind, a_line, b_line)
        if key in seen_real:
            return
        seen_real.add(key)
        real.append(LineDiff(kind, a_line, b_line, a_frag[:_FRAGMENT], b_frag[:_FRAGMENT]))
        if a_line is not None:
            real_a.add(a_line)
        if b_line is not None:
            real_b.add(b_line)

    def add_notation(kind: str, a_line, b_line) -> None:
        notation.setdefault((a_line, b_line), set()).add(kind.split(":", 1)[1])

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a_seg, b_seg = a_text[i1:i2], b_text[j1:j2]
        label = classify_segment(a_seg, b_seg)
        if label == "real" and (
            (not b_seg and a_seg in a_parens)
            or (not a_seg and b_seg in b_parens)
            or (a_seg in a_parens and is_filler(b_seg))
            or (b_seg in b_parens and is_filler(a_seg))
        ):
            label = "notation:parenthesised"
        a_hits = _lines_in_span(a_prepared, i1, i2)
        b_hits = _lines_in_span(b_prepared, j1, j2)
        a_first = a_hits[0][0].number if a_hits else None
        b_first = b_hits[0][0].number if b_hits else None
        if label != "real":
            add_notation(label, a_first, b_first)
            continue
        if not b_seg:
            # text present in a, absent in b
            for line, whole in a_hits:
                if whole:
                    if is_filler(line.canon, min_length=1):
                        add_notation("notation:filler", line.number, None)
                    elif line.canon in a_parens:
                        add_notation("notation:parenthesised", line.number, None)
                    else:
                        add_real("missing_in_b", line.number, None, _fragment(line.text), "")
                else:
                    part = a_text[max(i1, line.start):min(i2, line.end)]
                    if is_filler(part) or part in a_parens:
                        add_notation("notation:filler" if is_filler(part) else "notation:parenthesised", line.number, b_first)
                    else:
                        add_real("word", line.number, b_first, part, "")
            continue
        if not a_seg:
            for line, whole in b_hits:
                if whole:
                    if is_filler(line.canon, min_length=1):
                        add_notation("notation:filler", None, line.number)
                    elif line.canon in b_parens:
                        add_notation("notation:parenthesised", None, line.number)
                    else:
                        add_real("missing_in_a", None, line.number, "", _fragment(line.text))
                else:
                    part = b_text[max(j1, line.start):min(j2, line.end)]
                    if is_filler(part) or part in b_parens:
                        add_notation("notation:filler" if is_filler(part) else "notation:parenthesised", a_first, line.number)
                    else:
                        add_real("word", a_first, line.number, "", part)
            continue
        # replaced text: a word difference on every line it touches
        for line, _whole in a_hits:
            part = a_text[max(i1, line.start):min(i2, line.end)]
            b_line = None
            for other, _ in b_hits:
                b_line = other.number
                break
            add_real("word", line.number, b_line, part, b_seg)
        for line, _whole in b_hits:
            if line.number not in real_b:
                part = b_text[max(j1, line.start):min(j2, line.end)]
                add_real("word", a_first, line.number, a_seg, part)

    real_keys = {(d.a_line, d.b_line) for d in real}
    by_number_a = {line.number: line for line in a_prepared}
    by_number_b = {line.number: line for line in b_prepared}
    notation_diffs = [
        LineDiff(
            "notation:" + "+".join(sorted(kinds)),
            a_line,
            b_line,
            _fragment(by_number_a[a_line].text) if a_line in by_number_a else "",
            _fragment(by_number_b[b_line].text) if b_line in by_number_b else "",
        )
        for (a_line, b_line), kinds in notation.items()
        if (a_line, b_line) not in real_keys
    ]

    a_nonfiller = [line for line in a_prepared if not is_filler(line.canon, min_length=1)]
    b_nonfiller = [line for line in b_prepared if not is_filler(line.canon, min_length=1)]
    matched_a = sum(1 for line in a_nonfiller if line.number not in real_a)
    matched_b = sum(1 for line in b_nonfiller if line.number not in real_b)

    # 1:1 line pairs: an a line without real differences whose span maps onto
    # exactly one b line (notation differences inside the line are fine).
    # start_pos maps an a offset to b *after* any insertion at that point,
    # end_pos to b *before* it, so an a line keeps its span when b adds lines.
    start_pos = [0] * (len(a_text) + 1)
    end_pos = [0] * (len(a_text) + 1)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            end_pos[i1] = j1
            start_pos[i1] = j2
            continue
        for k in range(i1, i2):
            mapped = j1 + (k - i1) if tag == "equal" else j1
            start_pos[k] = end_pos[k] = mapped
        start_pos[i2] = end_pos[i2] = j2
    b_by_span = {(line.start, line.end): line for line in b_prepared}
    pairs: list[tuple[int, int]] = []
    for line in a_prepared:
        if line.number in real_a:
            continue
        other = b_by_span.get((start_pos[line.start], end_pos[line.end]))
        if other is not None and other.number not in real_b:
            pairs.append((line.number, other.number))

    return LyricDiff(
        similarity, len(a_nonfiller), len(b_nonfiller), matched_a, real, notation_diffs, pairs, matched_b
    )


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
