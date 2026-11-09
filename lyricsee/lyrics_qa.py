"""Quality gate for an 'original  translation' lyric: translator notes, neighbour bleed, order, variants.

Pure functions, no I/O. ``check`` returns Issue rows; the pipeline refuses to write a ``note_leak`` line and
re-translates a ``bleed`` line on its own, and every remaining issue turns the song into ``needs_review``.

Kinds: note_leak, bleed, order, jp_variant, untranslated, ratio.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from candidate_text import char_similarity, detect_language, fold

KINDS = ("note_leak", "bleed", "order", "jp_variant", "untranslated", "ratio")

_KANA = re.compile(r"[぀-ゟ゠-ヿㇰ-ㇿ]")
_HAN = re.compile(r"[㐀-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
_TRACK_ROW_RE = re.compile(r"^\s*\d{1,2}\s*[.．、]\s*[^\d\s]")
_NOTE_MARKERS = (
    "我認為", "我认为", "譯註", "译注", "譯者", "译者", "翻譯", "翻译", "此外", "原因是", "神來之筆", "神来之笔",
    "個人", "个人", "意象", "文學性", "文学性", "整張專輯", "整张专辑", "直譯", "直译", "意譯", "意译", "註解", "注解",
    "補充", "补充", "聽起來", "考量", "選用", "选用",
)
_BRACKET_PAIR_RE = re.compile(r"【[^】]*】")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
STRONG_BLEED_CHARS = 6
_INTERJECTION_RE = re.compile(r"^[\s\W]*(?:(?:la|na|oh|ooh|woah|whoa|yeah|ah|hey|uh|mm|hmm|da|ha|yo|ba|doo|wow)[\s\W]*)+$",
                              re.IGNORECASE)

# Chinese forms of kanji that Japanese writes differently (cp932 cannot encode them or they are the wrong form).
_VARIANT_TO_JP = {"繫": "繋", "淚": "涙", "說": "説", "歎": "嘆", "裏": "裏", "迴": "廻", "眞": "真", "兇": "凶",
                  "衝": "衝", "着": "着", "僞": "偽", "纔": "才", "痺": "痺"}


@dataclass(frozen=True)
class Issue:
    kind: str
    line: int          # 1-based line number in the lyric
    reason: str
    strong: bool = False   # a bleed that is safe to repair without a human (identical, or >= STRONG_BLEED_CHARS shared)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "line": self.line, "reason": self.reason, "strong": self.strong}


def _kana_share(text: str) -> float:
    kana, han = len(_KANA.findall(text)), len(_HAN.findall(text))
    return kana / (kana + han) if kana + han else 0.0


def is_track_row(text: str) -> bool:
    """'2.春を待つ': a track-list row, not a lyric line."""
    return bool(_TRACK_ROW_RE.match(text or ""))


def is_note_line(text: str, song_lang: str = "ja") -> bool:
    """Chinese commentary in an original column (a translator's note), as opposed to a lyric line."""
    text = (text or "").strip()
    if song_lang != "ja" or len(text) < 20 or not _HAN.search(text):
        return False
    if _kana_share(text) >= 0.2 and not _has_marker(text):
        return False        # a Japanese line (kana carries the grammar)
    brackets = len(_BRACKET_PAIR_RE.findall(text))
    markers = sum(marker in text for marker in _NOTE_MARKERS)
    if markers >= 2 or (markers >= 1 and len(text) >= 30):
        return True
    return len(text) >= 80 and _kana_share(text) < 0.12 and ("，" in text or "。" in text or brackets >= 2)


def _has_marker(text: str) -> bool:
    return any(marker in text for marker in _NOTE_MARKERS)


def is_translation_note(text: str) -> bool:
    """Commentary in the translation column: a long line carrying note markers or bracketed quotations."""
    text = (text or "").strip()
    if len(text) < 40:
        return False
    markers = sum(marker in text for marker in _NOTE_MARKERS)
    return markers >= 2 or len(_BRACKET_PAIR_RE.findall(text)) >= 2 or (markers >= 1 and len(text) >= 60)


def _longest_common_substring(a: str, b: str) -> int:
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            value = prev[j - 1] + 1 if ca == cb else 0
            cur.append(value)
            if value > best:
                best = value
        prev = cur
    return best


def _plain(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def _is_bleed_pair(orig_a: str, trans_a: str, orig_b: str, trans_b: str) -> str:
    """Reason when translation b repeats translation a although the originals differ; '' if fine."""
    pa, pb = _plain(trans_a), _plain(trans_b)
    if not pa or not pb:
        return ""
    fa, fb = fold(orig_a), fold(orig_b)
    if fa == fb or (fa and fb and char_similarity(fa, fb) >= 0.6):
        return ""
    if pa == pb and len(pa) >= 4:
        return "identical translation for a different original"
    common = _longest_common_substring(pa, pb)
    shorter = min(len(pa), len(pb))
    if common < 3 or not shorter:
        return ""
    if common >= 6 or (common >= 4 and common >= 0.6 * shorter) or (shorter >= 4 and common >= 0.75 * shorter):
        if _longest_common_substring(fa, fb) >= max(3, int(0.5 * common)):
            return ""       # the originals share that material too: a legitimate repeat
        return f"shares {common} chars with the neighbouring translation"
    return ""


def _find_bleed(lines: Sequence[tuple[str, str]]) -> list[Issue]:
    issues: list[Issue] = []
    for i in range(1, len(lines)):
        orig_a, trans_a = lines[i - 1]
        orig_b, trans_b = lines[i]
        if not trans_a or not trans_b:
            continue
        reason = _is_bleed_pair(orig_a, trans_a, orig_b, trans_b)
        if reason:
            strong = reason.startswith("identical") or int(re.search(r"\d+", reason).group()) >= STRONG_BLEED_CHARS
            issues.append(Issue("bleed", i + 1, reason, strong))     # the later line is the one that repeats the earlier
    return issues


def _match_reference(line: str, reference_fold: list[str]) -> list[int]:
    key = fold(line)
    if not key:
        return []
    exact = [i for i, other in enumerate(reference_fold) if other == key]
    if exact:
        return exact
    return [i for i, other in enumerate(reference_fold) if other and char_similarity(key, other) >= 0.8]


def _find_order(originals: Sequence[str], reference: Sequence[str]) -> list[Issue]:
    reference_fold = [fold(line) for line in reference]
    if not any(reference_fold):
        return []
    matched = sum(bool(_match_reference(line, reference_fold)) for line in originals if fold(line))
    if matched < 0.5 * max(1, sum(bool(fold(line)) for line in originals)):
        return []   # a reference that explains under half of the lines is another transcription, not an order oracle
    issues: list[Issue] = []
    last = -1
    seen: set[str] = set()
    last_matched_line = 0
    for n, line in enumerate(originals, 1):
        key = fold(line)
        candidates = _match_reference(line, reference_fold)
        if not candidates:
            seen.add(key)
            continue
        forward = [i for i in candidates if i >= last]
        if forward:
            last = min(forward)
            last_matched_line = n
        elif key in seen:
            pass                 # a chorus the reference writes once and the song repeats
        elif max(candidates) < last - 2:
            issues.append(Issue("order", n, "line sits before text already placed (moved block)"))
        seen.add(key)
    for n in range(last_matched_line + 1, len(originals) + 1):
        if fold(originals[n - 1]):
            issues.append(Issue("order", n, "line after the end of the reference (appended)"))
    return issues


def _find_variants(originals: Sequence[str], reference: Sequence[str] | None) -> list[Issue]:
    issues: list[Issue] = []
    reference_fold = [fold(line) for line in reference or []]
    for n, line in enumerate(originals, 1):
        if not _KANA.search(line):
            continue
        bad = []
        for ch in line:
            if not _HAN.search(ch):
                continue
            try:
                ch.encode("cp932")
            except UnicodeEncodeError:
                bad.append(ch)
        if not bad:
            continue
        reference_form = ""
        if reference_fold:
            for index in _match_reference(line, reference_fold):
                reference_form = reference[index]
                break
        if reference_form:
            if all(ch not in reference_form for ch in bad):
                issues.append(Issue("jp_variant", n, f"{len(bad)} Chinese-variant kanji; the reference has the Japanese form"))
        else:
            issues.append(Issue("jp_variant", n, f"{len(bad)} kanji outside cp932 (Chinese variant form)"))
    return issues


def _is_interjection(text: str) -> bool:
    return bool(_INTERJECTION_RE.match(text or "")) or len(_plain(text)) <= 2


def _find_untranslated(lines: Sequence[tuple[str, str]], song_lang: str) -> list[Issue]:
    issues: list[Issue] = []
    if lines and sum(bool((t or "").strip()) for _, t in lines) <= 0.1 * len(lines):
        return [Issue("untranslated", 1, "the song has no translation column")]
    for n, (original, translation) in enumerate(lines, 1):
        if not original.strip() or _is_interjection(original):
            continue
        translation = (translation or "").strip()
        if not translation:
            issues.append(Issue("untranslated", n, "empty translation"))
        elif _plain(translation) == _plain(original) and (_KANA.search(original) or len(_LATIN.findall(original)) >= 4):
            issues.append(Issue("untranslated", n, "translation equals the original"))
        else:
            kana = len(_KANA.findall(translation))
            letters = len(_LATIN.findall(translation))
            han = len(_HAN.findall(translation))
            if kana >= 3 and kana >= 0.3 * (kana + han):
                issues.append(Issue("untranslated", n, "kana left in the translation"))
            elif letters >= 8 and letters >= 3 * max(han, 1) and not _is_interjection(translation):
                issues.append(Issue("untranslated", n, "Latin text left in the translation"))
    return issues


def _find_ratio(lines: Sequence[tuple[str, str]]) -> list[Issue]:
    ratios = []
    for original, translation in lines:
        o, t = len(_plain(original)), len(_plain(translation))
        if o >= 4 and t:
            ratios.append(t / o)
    if len(ratios) < 8:
        return []
    median = statistics.median(ratios)
    if median <= 0:
        return []
    issues: list[Issue] = []
    for n, (original, translation) in enumerate(lines, 1):
        o, t = len(_plain(original)), len(_plain(translation))
        if o < 6 or not t:
            continue
        ratio = t / o
        if ratio > 3.0 * median + 0.5:
            issues.append(Issue("ratio", n, "translation far longer than the song's own ratio"))
        elif ratio < median / 4 and o >= 10:
            issues.append(Issue("ratio", n, "translation far shorter than the song's own ratio"))
    return issues


def check(lines: Sequence[tuple[str, str]], *, reference_original: Sequence[str] | None = None,
          song_lang: str | None = None, translated: bool = True) -> list[Issue]:
    """Issues of one lyric given as (original, translation) rows, sorted by line number."""
    lines = [(o or "", t or "") for o, t in lines]
    originals = [o for o, _ in lines]
    song_lang = song_lang or detect_language(originals)
    issues: list[Issue] = []
    for n, (original, translation) in enumerate(lines, 1):
        if is_track_row(original):
            issues.append(Issue("note_leak", n, "track-list row"))
        elif is_note_line(original, song_lang):
            issues.append(Issue("note_leak", n, "Chinese commentary in the original column"))
        elif is_translation_note(translation):
            issues.append(Issue("note_leak", n, "commentary in the translation column"))
    leaked = {issue.line for issue in issues}
    issues += [issue for issue in _find_bleed(lines) if issue.line not in leaked]
    if reference_original:
        issues += _find_order(originals, reference_original)
    issues += _find_variants(originals, reference_original)
    if translated:   # an original-only lyric has no translation column to judge
        issues += [issue for issue in _find_untranslated(lines, song_lang) if issue.line not in leaked]
        issues += [issue for issue in _find_ratio(lines) if issue.line not in leaked]
    return sorted(issues, key=lambda issue: (issue.line, issue.kind))


def parse_lyrics_text(text: str) -> list[tuple[str, str]]:
    """'original  translation' lines (two ASCII spaces) -> (original, translation) rows."""
    rows = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        original, _, translation = raw.partition("  ")
        rows.append((original.strip(), translation.strip()))
    return rows


def summarize(issues: Iterable[Issue]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for issue in issues:
        result.setdefault(issue.kind, []).append(issue.line)
    return result


def find_bleed(lines: Sequence[tuple[str, str]]) -> list[Issue]:
    return _find_bleed([(o or "", t or "") for o, t in lines])


def gate(pairs: Sequence, *, reference_original: Sequence[str] | None = None, song_lang: str | None = None,
         retranslate=None, repair_weak: bool = True, translated: bool = True) -> tuple[list, list[Issue], list[Issue]]:
    """The write-time gate. ``pairs``: objects with ``original`` / ``translation`` (LyricPair).

    Returns (kept pairs, issues left on the kept pairs, dropped note rows). Note / track-list rows are removed
    and never written. A bleed line is re-translated alone through ``retranslate(original) -> str`` (no
    neighbours in the prompt) and re-checked; a line that still fails keeps its text and stays in the issues.
    ``repair_weak`` False repairs only strong bleeds (use it when the translation is a human one).
    """
    pairs = list(pairs)
    rows = [(p.original, p.translation or "") for p in pairs]
    first = check(rows, reference_original=reference_original, song_lang=song_lang, translated=translated)
    drop = {issue.line for issue in first if issue.kind == "note_leak"}
    dropped = [issue for issue in first if issue.kind == "note_leak"]
    kept = [p for n, p in enumerate(pairs, 1) if n not in drop]
    if retranslate is not None and kept:
        kept_rows = [(p.original, p.translation or "") for p in kept]
        for issue in find_bleed(kept_rows):
            if not (issue.strong or repair_weak):
                continue
            try:
                new = (retranslate(kept[issue.line - 1].original) or "").strip()
            except Exception:
                continue        # the line keeps its text and stays flagged
            if new:
                kept[issue.line - 1] = type(kept[issue.line - 1])(kept[issue.line - 1].original, new)
    issues = check([(p.original, p.translation or "") for p in kept], reference_original=reference_original,
                   song_lang=song_lang, translated=translated)
    return kept, issues, dropped
