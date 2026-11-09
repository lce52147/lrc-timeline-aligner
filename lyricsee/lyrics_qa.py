"""Quality gate for an 'original  translation' lyric: translator notes, neighbour bleed, order, variants.

Pure functions, no I/O. ``check`` returns Issue rows; the pipeline refuses to write a ``note_leak`` line and
re-translates a ``bleed`` line on its own, and every remaining issue turns the song into ``needs_review``.

Kinds: note_leak, note_in_original_keep_translation, bleed, order, jp_variant, untranslated, ratio.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from candidate_text import char_similarity, detect_language, fold

KINDS = ("note_leak", "note_in_original_keep_translation", "bleed", "order", "jp_variant", "untranslated", "ratio", "script_garbage")

_KANA = re.compile(r"[぀-ゟ゠-ヿㇰ-ㇿ]")
_HAN = re.compile(r"[㐀-鿿]")
_LATIN = re.compile(r"[A-Za-z]")
_TRACK_ROW_RE = re.compile(r"^\s*\d{1,2}\s*[.．、]\s*[^\d\s]")
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d{1,2}\s*[.．、]\s*")
# Phrases that only a commentator writes. One of them in a long line is a note.
_NOTE_MARKERS = (
    "我認為", "我认为", "譯註", "译注", "譯者", "译者", "直譯", "直译", "意譯", "意译", "神來之筆", "神来之笔",
    "文學性", "文学性", "整張專輯", "整张专辑",
)
# Commentary frames: a translator talking about the translation or the word.
_NOTE_FRAMES = (
    "翻成", "翻譯成", "翻译成", "去查", "查了", "不採用", "不采用", "不刻意", "決定維持", "决定维持",
    "這是描述", "这是描述", "是描述", "這個詞", "这个词", "這個字", "这个字", "此處", "此处",
)
# Ordinary words of lyrics and translations: they count only next to a marker or a frame, never alone.
_WEAK_MARKERS = ("個人", "个人", "意象", "註解", "注解", "補充", "补充", "聽起來", "考量", "選用", "选用", "此外", "原因是")
# Function words of Chinese: a long kanji run that carries two of them is Chinese prose, not Japanese.
_CHINESE_FUNCTION = frozenset("的是也了在這这個个為为對对與与們们說说會会從从過过讓让還还沒没把被就都很")
_HAN_RUN_RE = re.compile(r"[㐀-鿿]{8,}")
_DICTIONARY_FRAME_RE = re.compile(
    r"(?:ことをいいます|のことです|を指します|を意味|という意味|の意味|と表し|漢字で|を表す|状態のこと|ことを指す|「[^」]*―[^」]*」)")
_BRACKET_PAIR_RE = re.compile(r"【[^】]*】")
_PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
STRONG_BLEED_CHARS = 6
_INTERJECTION_RE = re.compile(r"^[\s\W]*(?:(?:la|na|oh|ooh|woah|whoa|yeah|ah|hey|uh|mm|hmm|da|ha|yo|ba|doo|wow)[\s\W]*)+$",
                              re.IGNORECASE)

# Chinese forms of kanji that Japanese writes differently (cp932 cannot encode them or they are the wrong form).
# Only forms a Japanese site does not use: standard JIS X 0213 forms (頰 剝 塡 𠮟 …) are not in the table.
_VARIANT_TO_JP = {
    "繫": "繋", "淚": "涙", "戀": "恋", "說": "説", "眞": "真", "僞": "偽", "迴": "廻", "纔": "才", "與": "与",
    "們": "達", "爲": "為", "麼": "么", "嗎": "か", "讓": "譲", "從": "従",
    "對": "対", "會": "会", "點": "点", "來": "来", "聲": "声", "讀": "読", "寫": "写", "關": "関", "當": "当",
    "學": "学", "歸": "帰", "傳": "伝", "體": "体", "觀": "観", "歷": "歴", "戲": "戯", "響": "響", "憶": "憶",
}
_VARIANT_TO_JP = {k: v for k, v in _VARIANT_TO_JP.items() if k != v}


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
    """Shape only: '2.春を待つ' looks like a numbered row. Whether it IS a track-list row depends on where it sits
    (see ``track_row_indices``): a numbered line inside the song body is a lyric."""
    return bool(_TRACK_ROW_RE.match(text or ""))


def _strip_number(text: str) -> str:
    return _NUMBER_PREFIX_RE.sub("", text or "", count=1)


def track_row_indices(texts: Sequence[str], reference: Sequence[str] | None = None, *, is_lyric=None,
                      head: bool = False) -> set[int]:
    """Indices of the track-list rows among ``texts``.

    A numbered row is a track-list row only when it sits in a run of at least 2 consecutive numbered rows, the run
    is after the last lyric line (the tail; ``head`` also accepts a run before the first lyric line), and the row has
    no match in ``reference``. ``is_lyric(text)`` says which non-numbered lines count as lyric lines (default: any
    non-empty line); other lines neither break a run nor count as lyric. A numbered line inside the body is kept.
    """
    texts = [t or "" for t in texts]
    numbered = [is_track_row(t) for t in texts]
    is_lyric = is_lyric or (lambda t: bool(t.strip()))
    relevant = [i for i, t in enumerate(texts) if numbered[i] or is_lyric(t)]
    runs: list[list[int]] = []
    current: list[int] = []
    for i in relevant:
        if numbered[i]:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    reference_fold = [fold(line) for line in reference or []]
    result: set[int] = set()
    for run in runs:
        if len(run) < 2:
            continue
        lyric_after = any(not numbered[i] for i in relevant if i > run[-1])
        lyric_before = any(not numbered[i] for i in relevant if i < run[0])
        if lyric_after and not (head and not lyric_before):
            continue
        for i in run:
            if reference_fold and _match_reference(_strip_number(texts[i]), reference_fold):
                continue
            result.add(i)
    return result


def _marker_counts(text: str) -> tuple[int, int, int]:
    """(commentator phrases, commentary frames, ordinary words) found in the text."""
    return (sum(m in text for m in _NOTE_MARKERS), sum(f in text for f in _NOTE_FRAMES),
            sum(w in text for w in _WEAK_MARKERS))


def has_commentary_frame(text: str) -> bool:
    strong, frames, _ = _marker_counts(text or "")
    return bool(strong or frames)


# Function characters that Japanese also writes (過去 在 会 為 也 是 的 …): they never count alone.
_JA_COMMON_FUNCTION = frozenset("的是也了在這过過會会為为對对與与從从都就被把")
_CHINESE_ONLY_FUNCTION = _CHINESE_FUNCTION - _JA_COMMON_FUNCTION | frozenset("這")


def _is_chinese_prose(text: str) -> bool:
    """A run of 8+ kanji with 2+ Chinese-only function words, or 3+ function words in all ('同時也是日本古搖籃曲的句子').

    A kanji chain of Japanese ('過去現在未来永劫') carries 過/在/会/為, which are not enough."""
    for run in _HAN_RUN_RE.finditer(text):
        chars = run.group()
        if sum(ch in _CHINESE_ONLY_FUNCTION for ch in chars) >= 2 or sum(ch in _CHINESE_FUNCTION for ch in chars) >= 3:
            return True
    return False


def is_note_line(text: str, song_lang: str = "ja") -> bool:
    """A translator's note in the original column of a Japanese song, as opposed to a lyric line.

    Shapes: Chinese prose (a long kanji run with Chinese function words), a Japanese dictionary explanation
    (full stop plus a definition frame), or commentary phrases in a line that is not kana-rich.
    """
    text = (text or "").strip()
    if song_lang != "ja" or len(text) < 20 or not _HAN.search(text):
        return False
    if _is_chinese_prose(text):
        return True
    share = _kana_share(text)
    if _KANA.search(text) and share >= 0.2:
        # a Japanese line (kana carries the grammar), unless it is a dictionary explanation
        return len(text) >= 25 and "。" in text and bool(_DICTIONARY_FRAME_RE.search(text))
    strong, frames, weak = _marker_counts(text)
    score = 2 * (strong + frames) + weak
    if score >= 3 or (strong + frames >= 1 and len(text) >= 30):
        return True
    return len(text) >= 80 and share < 0.12 and ("，" in text or "。" in text)


def is_translation_note(text: str) -> bool:
    """Commentary in the translation column: a numbered or long line that talks about the translation."""
    text = (text or "").strip()
    if len(text) < 20:
        return False
    strong, frames, weak = _marker_counts(text)
    talk = strong + frames
    if is_track_row(text) and len(text) >= 30 and talk >= 1:
        return True       # '2. レガシー看到大家都翻譯成遺產…'
    if len(text) < 40:
        return False
    if talk >= 2 or (talk >= 1 and (len(text) >= 60 or weak >= 1)):
        return True
    return talk >= 1 and len(_BRACKET_PAIR_RE.findall(text)) >= 2


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
    fa, fb = fold(_PAREN_RE.sub("", orig_a)), fold(_PAREN_RE.sub("", orig_b))   # ruby in brackets is not text
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


def _median_ratio(lines: Sequence[tuple[str, str]]) -> float:
    ratios = [len(_plain(t)) / len(_plain(o)) for o, t in lines if len(_plain(o)) >= 4 and _plain(t)]
    return statistics.median(ratios) if ratios else 1.0


def _find_bleed(lines: Sequence[tuple[str, str]]) -> list[Issue]:
    issues: list[Issue] = []
    median = _median_ratio(lines)
    for i in range(1, len(lines)):
        orig_a, trans_a = lines[i - 1]
        orig_b, trans_b = lines[i]
        if not trans_a or not trans_b:
            continue
        reason = _is_bleed_pair(orig_a, trans_a, orig_b, trans_b)
        if reason:
            strong = reason.startswith("identical") or int(re.search(r"\d+", reason).group()) >= STRONG_BLEED_CHARS
            longer = len(_plain(trans_b)) > 1.3 * median * max(1, len(_plain(orig_b)))
            if strong or longer:      # a weak overlap counts only when the line carries more text than its own original
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
        last_matched_line = n       # a line the reference explains is never "appended", repeated chorus included
        forward = [i for i in candidates if i >= last]
        if forward:
            last = min(forward)
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
        bad = [ch for ch in line if ch in _VARIANT_TO_JP]
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
            issues.append(Issue("jp_variant", n, f"{len(bad)} Chinese-variant kanji ({''.join(dict.fromkeys(bad))} for {''.join(_VARIANT_TO_JP[c] for c in dict.fromkeys(bad))})"))
    return issues


def _is_interjection(text: str) -> bool:
    return bool(_INTERJECTION_RE.match(text or "")) or len(_plain(text)) <= 2


def _is_pure_latin(text: str) -> bool:
    return len(_LATIN.findall(text)) >= 4 and not _KANA.search(text) and not _HAN.search(text)


def _find_untranslated(lines: Sequence[tuple[str, str]], song_lang: str) -> list[Issue]:
    issues: list[Issue] = []
    if lines and sum(bool((t or "").strip()) for _, t in lines) <= 0.1 * len(lines):
        return [Issue("untranslated", 1, "the song has no translation column")]
    # An English hook / scat line the song keeps in English on another row is on purpose, not untranslated.
    kept_latin = [n for n, (o, t) in enumerate(lines, 1)
                  if _is_pure_latin(o) and _is_pure_latin(t or "") and not _HAN.search(t or "")]
    for n, (original, translation) in enumerate(lines, 1):
        if not original.strip() or _is_interjection(original):
            continue
        translation = (translation or "").strip()
        pure_latin = _is_pure_latin(original)
        elsewhere = any(other != n for other in kept_latin)
        suffix = " (pure Latin line: report only)" if pure_latin else ""
        if not translation:
            issues.append(Issue("untranslated", n, "empty translation"))
        elif _plain(translation) == _plain(original) and (_KANA.search(original) or len(_LATIN.findall(original)) >= 4):
            if not (pure_latin and elsewhere):
                issues.append(Issue("untranslated", n, "translation equals the original" + suffix))
        else:
            for chunk in _PAREN_RE.findall(translation):
                if chunk in original:
                    translation = translation.replace(chunk, "")     # a chant kept in brackets on purpose
            kana = len(_KANA.findall(translation))
            letters = len(_LATIN.findall(translation))
            han = len(_HAN.findall(translation))
            if kana >= 3 and kana >= 0.3 * (kana + han):
                issues.append(Issue("untranslated", n, "kana left in the translation"))
            elif letters >= 8 and han == 0 and not _is_interjection(translation):
                if not (pure_latin and elsewhere):
                    issues.append(Issue("untranslated", n, "Latin text left in the translation" + suffix))
    return issues


_FOREIGN_SCRIPT = re.compile("[\u0370-\u03ff\u0400-\u052f\u0530-\u058f\u0590-\u06ff\u0900-\u0dff\u0e00-\u0fff\u1100-\u11ff\uac00-\ud7af\u3130-\u318f]")
_MIXED_TOKEN = re.compile("[A-Za-z]+[\u0370-\u052f\uac00-\ud7af]|[\u0370-\u052f\uac00-\ud7af]+[A-Za-z]")


def _find_script_garbage(lines: Sequence[tuple[str, str]]) -> list[Issue]:
    """A translation holding Cyrillic, Hangul or another foreign script (the original has none of it), or one
    token that mixes Latin with such letters ('тюmin'): a model slip, never a translation."""
    issues: list[Issue] = []
    for n, (original, translation) in enumerate(lines, 1):
        foreign = [ch for ch in _FOREIGN_SCRIPT.findall(translation or "") if ch not in (original or "")]
        if foreign:
            issues.append(Issue("script_garbage", n, f"foreign script in the translation ({''.join(dict.fromkeys(foreign))[:6]})", True))
        elif _MIXED_TOKEN.search(translation or ""):
            issues.append(Issue("script_garbage", n, "mixed-script token in the translation", True))
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


def _is_real_translation(original: str, translation: str) -> bool:
    """The translation column of a row whose original is a note still holds a lyric translation."""
    translation = (translation or "").strip()
    po, pt = _plain(original), _plain(translation)
    if not pt or po in pt or (pt in po and len(pt) >= 0.6 * len(po)) or char_similarity(fold(original), fold(translation)) >= 0.8:
        return False       # empty, or a copy of the note (a lyric the note merely quotes is still a lyric)
    return not (is_translation_note(translation) or has_commentary_frame(translation) or is_note_line(translation))


def check(lines: Sequence[tuple[str, str]], *, reference_original: Sequence[str] | None = None,
          song_lang: str | None = None, translated: bool = True) -> list[Issue]:
    """Issues of one lyric given as (original, translation) rows, sorted by line number."""
    lines = [(o or "", t or "") for o, t in lines]
    originals = [o for o, _ in lines]
    song_lang = song_lang or detect_language(originals)
    issues: list[Issue] = []
    track_rows = track_row_indices(originals, reference_original)
    for n, (original, translation) in enumerate(lines, 1):
        if n - 1 in track_rows:
            issues.append(Issue("note_leak", n, "track-list row"))
        elif is_note_line(original, song_lang):
            if _is_real_translation(original, translation):
                issues.append(Issue("note_in_original_keep_translation", n,
                                    "commentary in the original column but the translation column is a lyric: "
                                    "a human restores the original"))
            else:
                issues.append(Issue("note_leak", n, "Chinese commentary in the original column"))
        elif is_translation_note(translation):
            issues.append(Issue("note_leak", n, "commentary in the translation column"))
    leaked = {issue.line for issue in issues}
    issues += [issue for issue in _find_bleed(lines) if issue.line not in leaked]
    if reference_original:
        issues += _find_order(originals, reference_original)
    issues += [issue for issue in _find_variants(originals, reference_original) if issue.line not in leaked]
    if translated:   # an original-only lyric has no translation column to judge
        issues += [issue for issue in _find_untranslated(lines, song_lang) if issue.line not in leaked]
        issues += [issue for issue in _find_ratio(lines) if issue.line not in leaked]
        issues += [issue for issue in _find_script_garbage(lines) if issue.line not in leaked]
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


def _clearly_better(original: str, old: str, new: str, rows: list, at: int) -> bool:
    """The re-translation of a bleed line replaces the old text only when it passes the checks on its own row and is
    not a fragment: at least half of the length the song's own ratio expects, no Latin the original lacks."""
    if not new or _plain(new) == _plain(old):
        return False
    trial = list(rows)
    trial[at] = (original, new)
    for i in check(trial, translated=True):
        if i.line == at + 1 and i.kind in ("bleed", "untranslated", "script_garbage"):
            return False
        if i.line == at + 2 and i.kind == "bleed":      # a new bleed into the next row (reported on its later line)
            return False
    expected = _median_ratio(rows) * len(_plain(original))
    if len(_plain(new)) < max(2, 0.5 * expected):
        return False
    if len(_LATIN.findall(new)) >= 4 and not _LATIN.search(original):
        return False
    return True


def gate(pairs: Sequence, *, reference_original: Sequence[str] | None = None, song_lang: str | None = None,
         retranslate=None, repair_weak: bool = False, translated: bool = True) -> tuple[list, list[Issue], list[Issue]]:
    """The write-time gate. ``pairs``: objects with ``original`` / ``translation`` (LyricPair).

    Returns (kept pairs, issues left on the kept pairs, dropped note rows). Note / track-list rows are removed
    and never written. A STRONG bleed (or a foreign-script slip) is re-translated alone through
    ``retranslate(original, context=(before, after)) -> str``; a retranslator that does not accept context gets only
    the line. The new text replaces the old one only when it passes the checks and is not a fragment
    (``_clearly_better``); otherwise the old text stays and the line stays in the issues. Weak bleeds are report-only
    (``repair_weak`` is kept for old callers and has no effect). A row whose original is a note but whose translation
    is a lyric is flagged, never dropped.
    """
    pairs = list(pairs)
    rows = [(p.original, p.translation or "") for p in pairs]
    first = check(rows, reference_original=reference_original, song_lang=song_lang, translated=translated)
    drop = {issue.line for issue in first if issue.kind == "note_leak"}
    dropped = [issue for issue in first if issue.kind == "note_leak"]
    kept = [p for n, p in enumerate(pairs, 1) if n not in drop]
    if retranslate is not None and kept:
        kept_rows = [(p.original, p.translation or "") for p in kept]
        repairs = [i for i in find_bleed(kept_rows) if i.strong]
        repairs += [i for i in _find_script_garbage(kept_rows)]
        for issue in repairs:
            try:
                at = issue.line - 1
                if getattr(retranslate, "accepts_context", False):
                    context = ([p.original for p in kept[max(0, at - 2):at]], [p.original for p in kept[at + 1:at + 3]])
                    new = (retranslate(kept[at].original, context=context) or "").strip()
                else:
                    new = (retranslate(kept[at].original) or "").strip()
            except Exception:
                continue        # the line keeps its text and stays flagged
            current = [(p.original, p.translation or "") for p in kept]
            if _clearly_better(kept[at].original, kept[at].translation or "", new, current, at):
                kept[at] = type(kept[at])(kept[at].original, new)
    issues = check([(p.original, p.translation or "") for p in kept], reference_original=reference_original,
                   song_lang=song_lang, translated=translated)
    return kept, issues, dropped
