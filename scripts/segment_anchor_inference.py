"""Derive *untrusted* repeated-chorus passage anchors from lyrics and vocal audio.

No LRC/BK timestamps, CTC clocks or reviewer consensus are used. This module
does NOT prove occurrence identity or grant Central timing authority. It will
refuse ambiguous repeated sections rather than pretend a score is ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import unicodedata
import re
import math

import numpy as np

from segment_structure import VocalPassage
from segment_recurrence import scan_vocal_recurrences


@dataclass(frozen=True)
class RepeatedSection:
    first_start: int       # one-based lyric row
    second_start: int
    count: int


@dataclass(frozen=True)
class PassagePairEvidence:
    first_passage: int
    second_passage: int
    similarity: float
    matched_time: float


@dataclass(frozen=True)
class InferredAnchors:
    status: str
    anchors: tuple[tuple[int, int, int], ...]
    repeated_section: RepeatedSection | None
    evidence: tuple[PassagePairEvidence, ...]
    reasons: tuple[str, ...] = ()


def _canonical(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text))).casefold().strip()


def find_unambiguous_repeated_section(
    lyric_texts: Sequence[str],
    *,
    min_lines: int = 3,
    min_distinct_lines: int = 2,
    max_lines: int = 12,
) -> tuple[RepeatedSection | None, str]:
    """Prefer one maximal 2-occurrence lexical block; fail on ties/triples.

    A run such as "la la la" repeated throughout a chorus is ambiguous, even
    when its local position happens to look monotonic.
    """
    lines = tuple(_canonical(x) for x in lyric_texts)
    n = len(lines)
    if n < min_lines * 2:
        return None, "BLOCKED_NO_REPEATED_SECTION"
    found: list[RepeatedSection] = []
    for i in range(n):
        if not lines[i]:
            continue
        for j in range(i + min_lines, n):
            if lines[i] != lines[j] or (i and j and lines[i-1] == lines[j-1]):
                continue
            length = 0
            while (i + length < j and j + length < n
                   and lines[i+length] and lines[i+length] == lines[j+length]):
                length += 1
            if (min_lines <= length <= max_lines and
                    len(set(lines[i:i+length])) >= min_distinct_lines):
                found.append(RepeatedSection(i+1, j+1, length))
    if not found:
        return None, "BLOCKED_NO_UNAMBIGUOUS_REPEATED_LYRICS"
    longest = max(x.count for x in found)
    best = [x for x in found if x.count == longest]
    # A true two-occurrence repeat has exactly one maximal start pair. Any
    # competing equality or >2 occurrences must be left untrusted.
    if len(best) != 1:
        return None, "BLOCKED_AMBIGUOUS_REPEATED_SECTIONS"
    target = best[0]
    run = lines[target.first_start-1:target.first_start-1+target.count]
    occurrences = [
        k+1 for k in range(n-target.count+1)
        if lines[k:k+target.count] == run
    ]
    if occurrences != [target.first_start, target.second_start]:
        return None, "BLOCKED_NOT_EXACTLY_TWO_OCCURRENCES"
    return target, "LEXICAL_REPEAT_UNVERIFIED"


def infer_repeated_passage_anchors(
    lyric_texts: Sequence[str],
    passages: Sequence[VocalPassage],
    features: np.ndarray,
    rms_db: np.ndarray,
    *,
    hop_seconds: float,
    min_similarity: float = .62,
    min_winner_margin: float = .08,
    max_progress_difference: float = .22,
    max_duration_fraction_diff: float = .38,
    onset_tolerance_s: float = .50,
) -> InferredAnchors:
    """Select one unique vocal passage pair for one unique repeated lyric run.

    An individual passage can contain multiple lines. The passage pair is only
    a *research candidate*, never independent line identity proof.
    """
    section, lexical_status = find_unambiguous_repeated_section(lyric_texts)
    if section is None:
        return InferredAnchors(lexical_status, (), None, (), (lexical_status,))
    n, m = len(lyric_texts), len(passages)
    if m < 2 or hop_seconds <= 0:
        return InferredAnchors("BLOCKED_NO_ACOUSTIC_PAIR", (), section, (), ("insufficient-vocal-passages",))
    if not np.isfinite(features).all() or not np.isfinite(rms_db).all():
        return InferredAnchors("BLOCKED_INVALID_ACOUSTICS", (), section, (), ("nonfinite-features",))
    candidates: list[PassagePairEvidence] = []
    # Before listening, constrain the *coarse* order with text proportion only.
    # No baseline CTC or BK timestamps can enter these bounds.
    for i, p in enumerate(passages):
        if p.duration < 4.0:
            continue
        position1 = (section.first_start - 1 + section.count / 2) / n
        if abs((i + .5) / m - position1) > max_progress_difference:
            continue
        later = []
        for j in range(i+1, m):
            q = passages[j]
            position2 = (section.second_start - 1 + section.count / 2) / n
            if abs((j + .5) / m - position2) > max_progress_difference:
                continue
            if q.duration < 4.0 or abs(q.duration-p.duration) > max(2., p.duration*max_duration_fraction_diff):
                continue
            later.append(j)
        if not later:
            continue
        scan = scan_vocal_recurrences(
            features, rms_db, hop_seconds=hop_seconds,
            template_start=p.start, template_end=p.end,
            min_similarity=min_similarity, min_spacing_seconds=4.,
        )
        if scan.status != "RESEARCH_UNVERIFIED_OCCURRENCE":
            continue
        for j in later:
            near = [
                point for point in scan.matches
                if abs(point.time-passages[j].start) <= onset_tolerance_s
            ]
            if len(near) == 1:
                candidates.append(PassagePairEvidence(i,j,near[0].similarity,near[0].time))
    candidates.sort(key=lambda x:(-x.similarity,x.first_passage,x.second_passage))
    if not candidates:
        return InferredAnchors("BLOCKED_NO_ACOUSTIC_PAIR", (), section, (), ("no-matched-vocal-passages",))
    if len(candidates)>1 and candidates[0].similarity-candidates[1].similarity < min_winner_margin:
        return InferredAnchors("BLOCKED_ACOUSTIC_AMBIGUITY", (), section, tuple(candidates), ("runner-up-too-similar",))
    best=candidates[0]
    a,b=section.first_start,section.second_start
    anchors=((a,a+section.count-1,best.first_passage),
             (b,b+section.count-1,best.second_passage))
    return InferredAnchors(
        "RESEARCH_UNVERIFIED_OCCURRENCE",anchors,section,tuple(candidates),
        ("lexical-and-acoustic-pair-do-not-prove-individual-line-ownership",),
    )

@dataclass(frozen=True)
class StructuralRepairActivation:
    status: str
    medians_s: tuple[float, ...]
    displaced_rows: tuple[int, ...]


def qualify_structural_repair(
    anchored_bundles: Sequence[tuple[int,int,int]],
    proposed_times: Sequence[float],
    global_ctc_priors: Sequence[float | None],
    *,
    min_median_offset_s: float = 3.0,
    min_displaced_rows_per_section: int = 2,
) -> StructuralRepairActivation:
    """Do NOT activate repeat repairs for a clean or small-offset chorus.

    DP proposed clocks are weak acoustic allocation; this check can suppress
    unnecessary GPU work, but cannot authenticate occurrence ownership.
    """
    if len(proposed_times)!=len(global_ctc_priors) or len(anchored_bundles)<2:
        return StructuralRepairActivation("BLOCKED_INVALID_REPAIR_SCOPE",(),())
    medians=[]
    displaced=[]
    for a,b,_ in anchored_bundles:
        if not (1<=a<=b<=len(proposed_times)):
            return StructuralRepairActivation("BLOCKED_INVALID_REPAIR_SCOPE",(),())
        deltas=[]
        local_displaced=[]
        for i in range(a,b+1):
            x=global_ctc_priors[i-1]
            if x is None or not math.isfinite(float(x)) or not math.isfinite(float(proposed_times[i-1])):
                continue
            delta=abs(float(proposed_times[i-1])-float(x))
            deltas.append(delta)
            if delta>=min_median_offset_s:local_displaced.append(i)
        if not deltas:
            return StructuralRepairActivation("BLOCKED_INSUFFICIENT_GLOBAL_CTC_PRIORS",(),())
        ordered=sorted(deltas)
        mid=len(ordered)//2
        median=(ordered[mid] if len(ordered)%2 else (ordered[mid-1]+ordered[mid])/2)
        medians.append(round(median,3))
        displaced.extend(local_displaced)
        if median<min_median_offset_s or len(local_displaced)<min_displaced_rows_per_section:
            return StructuralRepairActivation(
                "BLOCKED_NO_BILATERAL_STRUCTURAL_COLLAPSE",tuple(medians),tuple(displaced))
    return StructuralRepairActivation(
        "RESEARCH_TRIGGER_UNVERIFIED_OCCURRENCE",tuple(medians),tuple(displaced))
