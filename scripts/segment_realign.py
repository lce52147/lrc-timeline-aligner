"""LRC 2.0 segment-level realignment: lyric sections, vocal passages and their mapping.

Pure Python, no audio decoding: callers pass the vocal-activity segments the aligner
already computes (``auto_lrc.analyze_audio(...).segments``), the normalized line texts
(or a precomputed line-similarity matrix) and optional weak priors. The result is one
time window per lyric line, inside which a later bounded CTC pass can look for the line
without the whole-song path drift that misplaces repeated choruses, the first line after a
long break and whole blocks of lines. Nothing here reads a checked LRC, writes an LRC or
claims timing authority: windows and proposed times are hypotheses for Central.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Sequence

# Lyric structure: two lines are "the same" when their romaji bigram Dice similarity is at
# least this; a repeated block needs at least this many lines.
REPEAT_LINE_SIMILARITY = 0.75
REPEAT_MIN_LINES = 2
MAX_CHORUS_FAMILIES = 2

# Audio structure: vocal-activity segments closer than PASSAGE_GAP_S belong to one passage
# (a verse or chorus); shorter passages are hums, ad libs or noise.
PASSAGE_GAP_S = 1.5
PASSAGE_MIN_S = 2.0
MAX_PASSAGES_PER_SECTION = 3

# Mapping costs (one lyric section onto a run of passages).
W_DURATION = 1.6        # |log(voiced time / expected sung time)|; above SKIP_PASSAGE_COST for a 60% excess
W_PROGRESS = 2.0        # |progress through the lyrics - progress through the singing|
W_PRIOR = 0.08          # per second of (capped) disagreement with the current CTC times
PRIOR_CAP_S = 7.0
W_CHORUS_SIM = 1.5      # 1 - timbre similarity between two occurrences of one chorus
W_CHORUS_DURATION = 0.8 # |log(duration ratio)| between two occurrences of one chorus
SKIP_PASSAGE_COST = 0.35  # an instrumental hum, doubling or ad-lib passage with no lyrics; cramming a
                          # section into too little voiced time costs more through W_DURATION
WINDOW_MARGIN_S = 0.35


@dataclass(frozen=True)
class LyricSection:
    start_row: int   # inclusive, 0-based
    end_row: int     # exclusive
    kind: str        # "verse" or "chorus"
    family: int | None = None      # chorus family index (occurrences of one chorus share it)
    occurrence: int | None = None  # 0-based occurrence index within the family

    @property
    def rows(self) -> range:
        return range(self.start_row, self.end_row)


@dataclass(frozen=True)
class Passage:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SectionPlacement:
    section: LyricSection
    first_passage: int
    last_passage: int   # inclusive
    start: float
    end: float


@dataclass(frozen=True)
class LineWindow:
    row: int
    start: float
    end: float
    proposed: float
    section_index: int


@dataclass(frozen=True)
class SectionMap:
    status: str
    reason: str
    sections: tuple[LyricSection, ...]
    passages: tuple[Passage, ...]
    placements: tuple[SectionPlacement, ...]
    windows: tuple[LineWindow, ...]
    cost: float
    diagnostics: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------- lyric structure


def dice_bigram(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) < 2 or len(b) < 2:
        return float(a == b)
    x = Counter(a[i:i + 2] for i in range(len(a) - 1))
    y = Counter(b[i:i + 2] for i in range(len(b) - 1))
    return 2.0 * sum((x & y).values()) / (sum(x.values()) + sum(y.values()))


def line_similarity(canon: Sequence[str]) -> list[list[float]]:
    n = len(canon)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0 if canon[i] else 0.0
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = dice_bigram(canon[i], canon[j])
    return sim


def repeated_blocks(
    sim: Sequence[Sequence[float]],
    *,
    min_lines: int = REPEAT_MIN_LINES,
    threshold: float = REPEAT_LINE_SIMILARITY,
) -> list[tuple[int, int, int]]:
    """Maximal diagonal runs (i, j, length) with sim[i+k][j+k] >= threshold, i < j."""
    n = len(sim)
    hit = [[sim[i][j] >= threshold for j in range(n)] for i in range(n)]
    runs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not hit[i][j] or (i and hit[i - 1][j - 1]):
                continue
            length = 0
            while i + length < j and j + length < n and hit[i + length][j + length]:
                length += 1
            if length >= min_lines:
                runs.append((i, j, length))
    return runs


def chorus_families(
    sim: Sequence[Sequence[float]],
    *,
    min_lines: int = REPEAT_MIN_LINES,
    threshold: float = REPEAT_LINE_SIMILARITY,
    max_families: int = MAX_CHORUS_FAMILIES,
) -> list[tuple[int, tuple[int, ...]]]:
    """Repeated blocks grouped into families: (length, sorted occurrence start rows).

    Longer blocks win; a family's occurrences never overlap each other or an accepted
    family. Nested repeats (a chorus whose second half also repeats elsewhere) fall under
    the longer block.
    """
    runs = repeated_blocks(sim, min_lines=min_lines, threshold=threshold)
    by_length: dict[int, dict[int, set[int]]] = {}
    for i, j, length in runs:
        groups = by_length.setdefault(length, {})
        # union-find over start rows for this block length
        a, b = groups.setdefault(i, {i}), groups.setdefault(j, {j})
        if a is not b:
            merged = a | b
            for row in merged:
                groups[row] = merged
    families: list[tuple[int, tuple[int, ...]]] = []
    taken: set[int] = set()
    for length in sorted(by_length, reverse=True):
        seen: set[int] = set()
        for start, members in sorted(by_length[length].items()):
            if start in seen:
                continue
            seen |= members
            occurrences = tuple(sorted(members))
            rows = {r for occ in occurrences for r in range(occ, occ + length)}
            if len(occurrences) < 2 or rows & taken:
                continue
            families.append((length, occurrences))
            taken |= rows
            if len(families) >= max_families:
                return families
    return families


def lyric_sections(
    canon: Sequence[str] | None = None,
    *,
    sim: Sequence[Sequence[float]] | None = None,
    min_lines: int = REPEAT_MIN_LINES,
    threshold: float = REPEAT_LINE_SIMILARITY,
) -> tuple[LyricSection, ...]:
    """Partition the lines into chorus occurrences and the verse runs between them."""
    if sim is None:
        if canon is None:
            raise ValueError("canon lines or a similarity matrix is required")
        sim = line_similarity(canon)
    n = len(sim)
    if n == 0:
        return ()
    marks: dict[int, tuple[int, int, int]] = {}  # start row -> (family, occurrence, length)
    for family, (length, occurrences) in enumerate(chorus_families(sim, min_lines=min_lines, threshold=threshold)):
        for occurrence, start in enumerate(occurrences):
            marks[start] = (family, occurrence, length)
    sections: list[LyricSection] = []
    row = 0
    verse_start: int | None = None
    while row < n:
        if row in marks:
            if verse_start is not None:
                sections.append(LyricSection(verse_start, row, "verse"))
                verse_start = None
            family, occurrence, length = marks[row]
            sections.append(LyricSection(row, row + length, "chorus", family, occurrence))
            row += length
            continue
        if verse_start is None:
            verse_start = row
        row += 1
    if verse_start is not None:
        sections.append(LyricSection(verse_start, n, "verse"))
    return tuple(sections)


# ---------------------------------------------------------------- audio structure


def vocal_passages(
    segments: Sequence[tuple[float, float]],
    *,
    duration: float,
    gap_s: float = PASSAGE_GAP_S,
    min_s: float = PASSAGE_MIN_S,
) -> tuple[Passage, ...]:
    """Merge vocal-activity segments across short gaps into verse/chorus-sized passages."""
    ordered = sorted((max(0.0, float(a)), min(duration, float(b))) for a, b in segments)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        if merged and start - merged[-1][1] <= gap_s:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return tuple(Passage(s, e) for s, e in merged if e - s >= min_s)


# ---------------------------------------------------------------- mapping


def _log_ratio(actual: float, expected: float) -> float:
    return abs(math.log(max(actual, 1e-3) / max(expected, 1e-3)))


def map_sections(
    sections: Sequence[LyricSection],
    passages: Sequence[Passage],
    *,
    duration: float,
    line_chars: Sequence[int],
    priors: Sequence[float | None] | None = None,
    pair_similarity: Sequence[Sequence[float]] | None = None,
    max_passages_per_section: int = MAX_PASSAGES_PER_SECTION,
    _expected_voiced: float | None = None,
) -> SectionMap:
    """Monotonic DP: each section takes 1..k consecutive passages, passages may be skipped.

    Two occurrences of one chorus must sound alike (``pair_similarity`` between their first
    passages, when given) and last about as long; that is what tells the mapper which of
    several similar-looking passages the second chorus is, where the whole-song CTC path
    only sees a long stretch of vowels. Expected section lengths come from the voiced time
    of the passages that carry lyrics, so a first pass that skips passages is repeated once
    with the skipped time removed from the budget.
    """
    sections = tuple(sections)
    passages = tuple(passages)
    n_lines = len(line_chars)
    if not sections or not passages or n_lines == 0:
        return SectionMap("BLOCKED", "no-sections-or-passages", sections, passages, (), (), math.inf)
    if sections[-1].end_row != n_lines or sections[0].start_row != 0:
        return SectionMap("BLOCKED", "sections-do-not-cover-lines", sections, passages, (), (), math.inf)
    chars = [max(1, int(c)) for c in line_chars]
    total_chars = sum(chars)
    total_voiced = sum(p.duration for p in passages)
    expected_voiced = _expected_voiced if _expected_voiced else total_voiced
    voice_before = [0.0]
    for p in passages:
        voice_before.append(voice_before[-1] + p.duration)
    char_before = [0]
    for c in chars:
        char_before.append(char_before[-1] + c)
    S, M = len(sections), len(passages)
    families = sorted({s.family for s in sections if s.family is not None})
    family_slot = {f: k for k, f in enumerate(families)}
    prior_list = list(priors) if priors is not None else [None] * n_lines

    def run_bounds(j0: int, j1: int) -> tuple[float, float, float]:
        voiced = sum(passages[j].duration for j in range(j0, j1 + 1))
        return passages[j0].start, passages[j1].end, voiced

    def interpolate(section: LyricSection, j0: int, j1: int) -> list[float]:
        """Spread the section's lines over its passages by character share."""
        rows = list(section.rows)
        if not rows:
            return []
        section_chars = sum(chars[r] for r in rows)
        voiced = sum(passages[j].duration for j in range(j0, j1 + 1))
        times: list[float] = []
        consumed = 0
        for r in rows:
            share = consumed / section_chars  # position of the line start, 0..1 of voiced time
            target = share * voiced
            # walk the passages
            acc = 0.0
            t = passages[j1].end
            for j in range(j0, j1 + 1):
                if target <= acc + passages[j].duration:
                    t = passages[j].start + (target - acc)
                    break
                acc += passages[j].duration
            times.append(t)
            consumed += chars[r]
        return times

    def placement_cost(i: int, j0: int, j1: int, previous_same_family: tuple[int, int] | None) -> float:
        section = sections[i]
        start, end, voiced = run_bounds(j0, j1)
        section_chars = char_before[section.end_row] - char_before[section.start_row]
        expected = expected_voiced * section_chars / total_chars
        cost = W_DURATION * _log_ratio(voiced, expected)
        progress_lyric = (char_before[section.start_row] + section_chars / 2) / total_chars
        progress_voice = (voice_before[j0] + voiced / 2) / max(total_voiced, 1e-3)
        cost += W_PROGRESS * abs(progress_lyric - progress_voice)
        if any(prior_list[r] is not None for r in section.rows):
            times = interpolate(section, j0, j1)
            deltas = [min(abs(t - float(prior_list[r])), PRIOR_CAP_S) for t, r in zip(times, section.rows) if prior_list[r] is not None]
            cost += W_PRIOR * sum(deltas) / max(1, len(deltas))
        if previous_same_family is not None:
            p0, p1 = previous_same_family
            _s, _e, prev_voiced = run_bounds(p0, p1)
            cost += W_CHORUS_DURATION * _log_ratio(voiced, prev_voiced)
            if pair_similarity is not None and len(pair_similarity) == M:
                try:
                    similarity = float(pair_similarity[p0][j0])
                except (IndexError, TypeError, ValueError):
                    similarity = 0.0
                cost += W_CHORUS_SIM * max(0.0, 1.0 - max(0.0, similarity))
        return cost

    INF = math.inf
    empty_state = tuple([None] * len(families))

    @lru_cache(maxsize=None)
    def best(i: int, j: int, placed: tuple) -> tuple[float, tuple]:
        """Cheapest way to place sections i.. using passages j.. ; returns (cost, decisions)."""
        if i == S:
            return SKIP_PASSAGE_COST * (M - j), ()
        if j == M:
            return INF, ()
        options: list[tuple[float, tuple]] = []
        # skip passage j (no lyrics in it)
        cost, rest = best(i, j + 1, placed)
        options.append((SKIP_PASSAGE_COST + cost, rest))
        section = sections[i]
        for j1 in range(j, min(M, j + max_passages_per_section)):
            previous = None
            new_placed = placed
            if section.family is not None:
                slot = family_slot[section.family]
                previous = placed[slot]
                new_placed = placed[:slot] + ((j, j1),) + placed[slot + 1:]
            here = placement_cost(i, j, j1, previous)
            cost, rest = best(i + 1, j1 + 1, new_placed)
            options.append((here + cost, ((i, j, j1),) + rest))
        return min(options, key=lambda item: item[0])

    total, decisions = best(0, 0, empty_state)
    if not math.isfinite(total):
        return SectionMap("BLOCKED", "no-monotonic-path", sections, passages, (), (), INF,
                          {"sections": S, "passages": M})
    placements: list[SectionPlacement] = []
    windows: list[LineWindow] = []
    for i, j0, j1 in decisions:
        section = sections[i]
        start, end, _voiced = run_bounds(j0, j1)
        placements.append(SectionPlacement(section, j0, j1, start, end))
        for r, t in zip(section.rows, interpolate(section, j0, j1)):
            windows.append(LineWindow(r, max(0.0, start - WINDOW_MARGIN_S), min(duration, end + WINDOW_MARGIN_S), round(t, 3), i))
    windows.sort(key=lambda w: w.row)
    used_voiced = sum(passages[j].duration for p in placements for j in range(p.first_passage, p.last_passage + 1))
    skipped = M - sum(p.last_passage - p.first_passage + 1 for p in placements)
    if skipped and _expected_voiced is None and used_voiced > 0:
        return map_sections(sections, passages, duration=duration, line_chars=line_chars, priors=priors,
                            pair_similarity=pair_similarity, max_passages_per_section=max_passages_per_section,
                            _expected_voiced=used_voiced)
    return SectionMap("READY", "ok", sections, passages, tuple(placements), tuple(windows), round(total, 4), {
        "sections": S, "passages": M, "chorus_families": len(families),
        "skipped_passages": skipped, "cost_per_section": round(total / S, 4),
        "expected_voiced_s": round(expected_voiced, 2), "refined": _expected_voiced is not None,
    })


def plan_segment_realign(
    canon: Sequence[str],
    segments: Sequence[tuple[float, float]],
    *,
    duration: float,
    line_chars: Sequence[int] | None = None,
    priors: Sequence[float | None] | None = None,
    sim: Sequence[Sequence[float]] | None = None,
    pair_similarity: Sequence[Sequence[float]] | None = None,
) -> SectionMap:
    """Lyric sections -> vocal passages -> one window per line."""
    chars = list(line_chars) if line_chars is not None else [max(1, len(c)) for c in canon]
    sections = lyric_sections(canon, sim=sim)
    passages = vocal_passages(segments, duration=duration)
    return map_sections(sections, passages, duration=duration, line_chars=chars, priors=priors, pair_similarity=pair_similarity)
