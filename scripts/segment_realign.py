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


# ---------------------------------------------------------------------------
# Row-level structural map (V2 P2).
#
# The whole-song CTC path is right for about 95 % of the lines; its failures are
# short runs of lines sung in another section, placed in an instrumental break,
# or collapsed on top of a repeated line.  Instead of mapping sections to coarse
# voiced passages (consecutive sections rarely have a 1.5 s gap between them),
# this map runs a Viterbi path over a 0.1 s grid for the rows themselves:
#   * rows with independent corroboration (``status`` TRUSTED / UNTRUSTED) are
#     observations with a strong weight and a narrow state range;
#   * review rows are weak observations that may move anywhere between their
#     nearest anchors;
#   * the time between two consecutive rows must fit the lyric structure: inside
#     a repeated section (chorus) it must equal the same gap in a sibling
#     occurrence (median absolute difference on the evaluation set 0.02 s, 95th
#     percentile 0.33 s); elsewhere the *voiced* time between the onsets must
#     match the line's length times the song's syllable rate (5th/95th
#     percentile of the ratio 0.57 / 1.41 on the evaluation set).
# Rows the path moves by more than a second are flagged; consecutive flagged
# rows form a realign run bounded by the neighbouring unflagged rows.  The run
# bounds are what a bounded CTC pass needs; the proposed time is only a hint.

import numpy as np

GRID_S = 0.1
VOICED_FLOOR_S = 0.3            # a line may start in a VAD gap (10 % of true onsets do)
OBS_WEIGHT = {"TRUSTED": 3.0, "UNTRUSTED": 2.0}   # cost per second of displacement
OBS_WEIGHT_REVIEW = 0.3
# A review row with a decent CTC score resists moving more than a near-zero one.
REVIEW_WEIGHT_TIERS = ((0.15, OBS_WEIGHT_REVIEW), (math.inf, 0.6))   # (ctc score below, weight)
OBS_CAP_S = 8.0
OBS_RANGE = {"TRUSTED": 2.0, "UNTRUSTED": 4.0}    # half range of the state grid around an anchor
REVIEW_RANGE_S = 60.0           # a review row with no anchor on one side still stays within this
HINT_BONUS = 0.4                # a time some backend already proposed is a little cheaper
HINT_TOL_S = 0.25
VAD_ABSENT_COST = 0.15
W_RATE = 1.4                    # per unit |ln(voiced gap / expected)| beyond the dead zone
RATE_DEAD_ZONE = 0.47           # ln(1.6): gaps within 0.62x..1.6x of the expectation cost nothing
W_SIBLING = 1.0 / 0.3           # per second of disagreement with the sibling chorus gap
MIN_GAP_RATIO = 0.3             # a line lasts at least this share of its expected time
MIN_GAP_S = 0.15
DEFAULT_RATE_S_PER_CHAR = 0.18  # romaji characters; evaluation-set medians 0.10-0.30
MIN_RATE_PAIRS = 5
MIN_ANCHORS = 3
FLAG_MOVE_REVIEW_S = 1.0
FLAG_MOVE_ANCHOR_S = 2.5
RUN_WINDOW_MAX_S = 45.0
RUN_WINDOW_SLACK_S = 6.0


@dataclass(frozen=True)
class RowObservation:
    """What the pipeline knows about one lyric row before Central runs."""

    time: float | None                  # current (whole-song CTC) onset in seconds
    chars: int                          # romaji length used for the rate model
    status: str = "REVIEW"              # TRUSTED / UNTRUSTED are anchors, anything else is weak
    hints: tuple[float, ...] = ()       # other onset proposals (hypotheses / candidates), seconds
    confidence: float = 0.0             # the row's own CTC score; sets a review row's weight


@dataclass(frozen=True)
class RowPlacement:
    row: int
    current: float | None
    proposed: float
    flagged: bool
    anchor: bool
    gap_model: str                      # how the gap to the next row was judged: sibling / rate / none


@dataclass(frozen=True)
class RealignRun:
    first_row: int
    last_row: int
    start: float                        # bounded by the neighbouring unflagged rows
    end: float
    proposed: tuple[float, ...]


@dataclass(frozen=True)
class StructuralMap:
    status: str
    reason: str
    rows: tuple[RowPlacement, ...]
    runs: tuple[RealignRun, ...]
    diagnostics: dict


def voiced_cumulative(segments: Sequence[tuple[float, float]], duration: float, grid: float = GRID_S) -> np.ndarray:
    """Cumulative voiced seconds at every grid point (length ``int(duration/grid)+1``)."""
    n = int(math.floor(max(0.0, duration) / grid)) + 1
    voiced = np.zeros(n, dtype=float)
    for start, end in segments:
        a = max(0, int(math.floor(float(start) / grid)))
        b = min(n, int(math.ceil(float(end) / grid)))
        if b > a:
            voiced[a:b] = 1.0
    return np.concatenate(([0.0], np.cumsum(voiced) * grid))[:n]


def sibling_gap_templates(sim: Sequence[Sequence[float]], obs: Sequence[RowObservation]) -> dict[int, float]:
    """Row index -> gap to the next row measured on a sibling occurrence of the same chorus.

    Only sibling pairs whose two rows are anchors contribute; the template is their median.
    """
    templates: dict[int, float] = {}
    for length, starts in chorus_families(sim):
        for offset in range(length - 1):
            gaps: list[tuple[int, float]] = []
            for start in starts:
                r0, r1 = start + offset, start + offset + 1
                if r1 >= len(obs):
                    continue
                a, b = obs[r0], obs[r1]
                if a.status in OBS_WEIGHT and b.status in OBS_WEIGHT and a.time is not None and b.time is not None and b.time > a.time:
                    gaps.append((start, b.time - a.time))
            for start in starts:
                others = sorted(g for s, g in gaps if s != start)
                if others:
                    templates[start + offset] = others[len(others) // 2]
    return templates


def song_syllable_rate(obs: Sequence[RowObservation], voiced: np.ndarray, grid: float = GRID_S) -> tuple[float, int]:
    """Median voiced seconds per romaji character over consecutive anchor pairs."""
    rates: list[float] = []
    for a, b in zip(obs, obs[1:]):
        if a.status not in OBS_WEIGHT or b.status not in OBS_WEIGHT or a.time is None or b.time is None:
            continue
        if b.time <= a.time:
            continue
        ia, ib = _grid_index(a.time, voiced, grid), _grid_index(b.time, voiced, grid)
        gap = float(voiced[ib] - voiced[ia]) + VOICED_FLOOR_S
        rates.append(gap / max(1, a.chars))
    if len(rates) < MIN_RATE_PAIRS:
        return DEFAULT_RATE_S_PER_CHAR, len(rates)
    rates.sort()
    return rates[len(rates) // 2], len(rates)


def review_weight(confidence: float) -> float:
    """Observation weight (cost per second of displacement) of a non-anchor row."""
    for ceiling, weight in REVIEW_WEIGHT_TIERS:
        if confidence < ceiling:
            return weight
    return REVIEW_WEIGHT_TIERS[-1][1]


def _grid_index(t: float, voiced: np.ndarray, grid: float) -> int:
    return min(len(voiced) - 1, max(0, int(round(t / grid))))


def structural_row_map(
    obs: Sequence[RowObservation],
    segments: Sequence[tuple[float, float]],
    *,
    duration: float,
    sim: Sequence[Sequence[float]] | None = None,
    grid: float = GRID_S,
) -> StructuralMap:
    """Viterbi path for every row; flags the rows the lyric structure moves."""
    obs = tuple(obs)
    n = len(obs)
    if n == 0 or duration <= grid:
        return StructuralMap("BLOCKED", "no-rows", (), (), {})
    anchors = [o.status in OBS_WEIGHT and o.time is not None for o in obs]
    if sum(anchors) < MIN_ANCHORS:
        return StructuralMap("BLOCKED", "too-few-anchors", (), (), {"anchors": sum(anchors), "rows": n})
    voiced = voiced_cumulative(segments, duration, grid)
    T = len(voiced)
    times = np.arange(T) * grid
    in_vad = np.zeros(T, dtype=bool)
    for start, end in segments:
        a = max(0, int(math.floor(float(start) / grid)))
        b = min(T, int(math.ceil(float(end) / grid)) + 1)
        in_vad[a:b] = True
    rate, rate_pairs = song_syllable_rate(obs, voiced, grid)
    templates = sibling_gap_templates(sim, obs) if sim is not None else {}

    # state range per row
    ranges: list[tuple[int, int]] = []
    for i, o in enumerate(obs):
        if anchors[i]:
            half = OBS_RANGE[o.status]
            lo, hi = o.time - half, o.time + half
        else:
            prev_anchor = next((obs[j].time for j in range(i - 1, -1, -1) if anchors[j]), None)
            next_anchor = next((obs[j].time for j in range(i + 1, n) if anchors[j]), None)
            lo = prev_anchor - OBS_RANGE["UNTRUSTED"] if prev_anchor is not None else (o.time - REVIEW_RANGE_S if o.time is not None else 0.0)
            hi = next_anchor + OBS_RANGE["UNTRUSTED"] if next_anchor is not None else (o.time + REVIEW_RANGE_S if o.time is not None else duration)
        k0 = max(0, min(T - 1, int(math.floor(lo / grid))))
        k1 = max(k0, min(T - 1, int(math.ceil(hi / grid))))
        ranges.append((k0, k1))

    def observation_cost(i: int, idx: np.ndarray) -> np.ndarray:
        o = obs[i]
        t = times[idx]
        cost = np.where(in_vad[idx], 0.0, VAD_ABSENT_COST)
        if o.time is not None:
            weight = OBS_WEIGHT.get(o.status) or review_weight(float(o.confidence or 0.0))
            cost = cost + weight * np.minimum(np.abs(t - o.time), OBS_CAP_S)
        if o.hints:
            hints = np.asarray(o.hints, dtype=float)
            near = np.min(np.abs(t[:, None] - hints[None, :]), axis=1) <= HINT_TOL_S
            cost = cost - HINT_BONUS * near
        return cost

    gap_models: list[str] = []
    idx_prev = np.arange(ranges[0][0], ranges[0][1] + 1)
    cost_prev = observation_cost(0, idx_prev)
    back: list[np.ndarray] = []
    for i in range(1, n):
        idx = np.arange(ranges[i][0], ranges[i][1] + 1)
        t0, t1 = times[idx_prev], times[idx]
        dt = t1[None, :] - t0[:, None]
        template = templates.get(i - 1)
        expected_rate = rate * max(1, obs[i - 1].chars)
        if template is not None:
            trans = W_SIBLING * np.abs(dt - template)
            min_gap = max(MIN_GAP_S, MIN_GAP_RATIO * template)
            gap_models.append("sibling")
        else:
            vg = voiced[idx][None, :] - voiced[idx_prev][:, None] + VOICED_FLOOR_S
            trans = W_RATE * np.maximum(np.abs(np.log(np.maximum(vg, 1e-3) / expected_rate)) - RATE_DEAD_ZONE, 0.0)
            min_gap = max(MIN_GAP_S, MIN_GAP_RATIO * expected_rate)
            gap_models.append("rate")
        trans = np.where(dt >= min_gap, trans, np.inf)
        total = cost_prev[:, None] + trans
        best_prev = np.argmin(total, axis=0)
        cost_here = total[best_prev, np.arange(len(idx))]
        if not np.isfinite(cost_here).any():
            return StructuralMap("BLOCKED", f"no-monotonic-path-at-row-{i}", (), (), {"anchors": sum(anchors), "rows": n})
        back.append(idx_prev[best_prev])
        cost_prev = cost_here + observation_cost(i, idx)
        idx_prev = idx
    gap_models.append("none")
    path_idx = [int(idx_prev[int(np.argmin(cost_prev))])]
    for i in range(n - 1, 0, -1):
        prev_idx = back[i - 1]
        local = path_idx[-1] - ranges[i][0]
        path_idx.append(int(prev_idx[local]))
    path_idx.reverse()
    path = [float(times[k]) for k in path_idx]

    rows: list[RowPlacement] = []
    for i, o in enumerate(obs):
        moved = abs(path[i] - o.time) if o.time is not None else 0.0
        threshold = FLAG_MOVE_ANCHOR_S if anchors[i] else FLAG_MOVE_REVIEW_S
        rows.append(RowPlacement(i, o.time, round(path[i], 3), bool(o.time is not None and moved > threshold), anchors[i], gap_models[i]))
    runs: list[RealignRun] = []
    i = 0
    while i < n:
        if not rows[i].flagged:
            i += 1
            continue
        j = i
        while j + 1 < n and rows[j + 1].flagged:
            j += 1
        start = path[i - 1] if i > 0 else 0.0
        end = path[j + 1] if j + 1 < n else duration
        if end - start > RUN_WINDOW_MAX_S:
            start = max(start, min(path[i:j + 1]) - RUN_WINDOW_SLACK_S)
            end = min(end, max(path[i:j + 1]) + RUN_WINDOW_SLACK_S)
        runs.append(RealignRun(i, j, round(start, 3), round(end, 3), tuple(round(p, 3) for p in path[i:j + 1])))
        i = j + 1
    return StructuralMap("READY", "ok", tuple(rows), tuple(runs), {
        "rows": n, "anchors": sum(anchors), "rate_s_per_char": round(rate, 4), "rate_pairs": rate_pairs,
        "sibling_pairs": len(templates), "flagged": sum(r.flagged for r in rows), "runs": len(runs),
        "cost": round(float(np.min(cost_prev)), 4),
    })
