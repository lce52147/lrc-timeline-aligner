"""LRC 1.3: independent vocal-passage allocation research core.

No BK timestamps or user annotations are accepted by this module.
A global CTC clock is a capped *weak* prior, never a hard window.
S1 only emits untrusted, passage-grouped hypotheses; an entire bundle must
validate before any caller may ask Central to evaluate it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Sequence


@dataclass(frozen=True)
class VocalPassage:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PassageBundle:
    start_row: int
    end_row: int
    passage_index: int
    proposed_times: tuple[float, ...]
    ctc_prior_displaced_rows: tuple[int, ...]
    status: str = "UNVERIFIED_NO_INDEPENDENT_OCCURRENCE_PROOF"


@dataclass(frozen=True)
class PassageAlignment:
    status: str
    passages: tuple[VocalPassage, ...]
    bundles: tuple[PassageBundle, ...]
    entry_passages: tuple[int, ...]
    entry_times: tuple[float, ...]
    cost: float
    reasons: tuple[str, ...]


def merge_vocal_passages(
    segments: Sequence[tuple[float, float]],
    *,
    duration: float,
    merge_gap_s: float = 0.65,
    min_duration_s: float = 0.55,
) -> tuple[VocalPassage, ...]:
    """Allow sub-second sung consonant gaps while retaining long silence."""
    if duration <= 0 or merge_gap_s < 0 or min_duration_s <= 0:
        raise ValueError("invalid vocal passage parameters")
    ordered = sorted((max(0.0, float(a)), min(duration, float(b))) for a, b in segments)
    merged: list[VocalPassage] = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        if merged and start - merged[-1].end <= merge_gap_s:
            merged[-1] = VocalPassage(merged[-1].start, max(end, merged[-1].end))
        else:
            merged.append(VocalPassage(start, end))
    return tuple(x for x in merged if x.duration >= min_duration_s)


def _row_times(passage: VocalPassage, lengths: Sequence[int]) -> tuple[float, ...]:
    """Place weak *pre-alignment* hypotheses inside an independently voiced passage."""
    # The time is deliberately not passed to production Central as onset proof.
    weights = [max(1, int(n)) for n in lengths]
    total = sum(weights)
    padding = min(0.14, passage.duration * 0.06)
    usable = max(0.0, passage.duration - padding * 2)
    running = 0
    out: list[float] = []
    for weight in weights:
        t = passage.start + padding + usable * (running / total)
        out.append(t)
        running += weight
    return tuple(out)


def _group_cost(
    passage: VocalPassage,
    lengths: Sequence[int],
    *,
    row_indices: range,
    cumulative_chars: Sequence[int],
    total_chars: int,
    total_voiced: float,
    preceding_voice: float,
    ctc_priors: Sequence[float | None],
    absolute_start: float,
) -> tuple[float, tuple[float, ...]]:
    proposed = _row_times(passage, lengths)
    count = len(lengths)
    if not count:
        return 0.35, ()
    # Estimate the amount of vocal time consumed by lyric characters, not
    # by line count; penalize stuffing long verses into a short passage.
    expected_span = max(0.10, total_voiced * sum(lengths) / max(total_chars, 1))
    duration_ratio = max(passage.duration / expected_span, 1e-4)
    allocation_cost = 1.1 * abs(math.log(duration_ratio))
    # Keep song-level relative progress weak compared with vocal segmentation.
    target_progress = (cumulative_chars[row_indices.start] + sum(lengths) / 2) / max(total_chars, 1)
    actual_progress = (preceding_voice + passage.duration / 2) / max(total_voiced, 0.01)
    progress_cost = 4.0 * abs(actual_progress - target_progress)
    # Exact CTC clocks can be catastrophically wrong at repeated lyrics.
    # Saturate the penalty; do not anchor or forbid a distant occurrence.
    ctc_cost = 0.0
    for i, t in zip(row_indices, proposed):
        prior = ctc_priors[i]
        if prior is not None and math.isfinite(prior):
            ctc_cost += 0.12 * min(abs(t - prior), 7.0)
    # Prevent pathological concentration, but permit rapid verses and rap.
    density_cost = 0.3 * max(0.0, (count / max(passage.duration, .2)) - 2.2)
    # Very short passages are unlikely to contain six full lyric lines.
    return allocation_cost + progress_cost + ctc_cost + density_cost, proposed


def assign_lyrics_to_vocal_passages(
    segments: Sequence[tuple[float, float]],
    *,
    duration: float,
    lyric_lengths: Sequence[int],
    ctc_priors: Sequence[float | None] | None = None,
    max_lines_per_passage: int = 12,
    anchored_bundles: Sequence[tuple[int, int, int]] = (),
) -> PassageAlignment:
    """Monotonic many-lyrics-to-one-passage DP, never reads BK.

    Passages may contain many lines; some passages may be skipped (hums,
    doubling, ad libs). Returned times are non-authoritative proposals.
    """
    vocal = merge_vocal_passages(segments, duration=duration)
    lens = tuple(max(1, int(n)) for n in lyric_lengths)
    n, m = len(lens), len(vocal)
    if n == 0 or m == 0:
        return PassageAlignment("BLOCKED_NO_SINGING_OR_LYRICS", vocal, (), (), (), float("inf"), ("no-passages-or-lines",))
    if max_lines_per_passage < 1:
        raise ValueError("max_lines_per_passage must be positive")
    priors = tuple(ctc_priors if ctc_priors is not None else [None] * n)
    if len(priors) != n:
        raise ValueError("one CTC prior per lyric required")
    # Repeat-section anchor proposals are never independent lyric proof by themselves.
    locked: dict[int, tuple[int, int]] = {}
    locked_rows: set[int] = set()
    previous_end, previous_passage = 0, -1
    for begin, end, passage_index in anchored_bundles:
        if not (1 <= begin <= end <= n and 0 <= passage_index < m):
            raise ValueError("invalid anchored bundle bounds")
        if begin <= previous_end or passage_index <= previous_passage:
            raise ValueError("anchored bundles must be sequential and nonoverlapping")
        if end - begin + 1 > max_lines_per_passage:
            raise ValueError("anchor exceeds passage capacity")
        locked[passage_index] = (begin, end)
        locked_rows.update(range(begin, end + 1))
        previous_end, previous_passage = end, passage_index
    prefix = [0]
    for x in lens:
        prefix.append(prefix[-1] + x)
    voiced_total = sum(x.duration for x in vocal)
    voice_before = [0.]
    for passage in vocal:
        voice_before.append(voice_before[-1] + passage.duration)
    # dp[j][i] = best way to allocate i lyrics using first j passages.
    inf = float("inf")
    dp = [[inf] * (n+1) for _ in range(m+1)]
    parent: list[list[tuple[int, int, tuple[float, ...]] | None]] = [[None]*(n+1) for _ in range(m+1)]
    dp[0][0] = 0.
    for j, passage in enumerate(vocal):
        for used in range(n+1):
            if not math.isfinite(dp[j][used]):
                continue
            # A silent/ad-lib/duplicate passage is legal, with a small penalty.
            for count in range(0, min(max_lines_per_passage,n-used)+1):
                if j in locked:
                    begin, end = locked[j]
                    if used + 1 != begin or count != end - begin + 1:
                        continue
                elif count and any(row in locked_rows for row in range(used+1,used+count+1)):
                    continue
                if count == 0:
                    score, times = 0.65, ()
                else:
                    row_indices = range(used, used+count)
                    score, times = _group_cost(
                        passage, lens[used:used+count], row_indices=row_indices,
                        cumulative_chars=prefix, total_chars=prefix[-1],
                        total_voiced=voiced_total, preceding_voice=voice_before[j],
                        ctc_priors=priors, absolute_start=passage.start,
                    )
                proposed = dp[j][used]+score
                nxt = used+count
                if proposed+1e-8 < dp[j+1][nxt]:
                    dp[j+1][nxt]=proposed
                    parent[j+1][nxt]=(used,j,times)
    if not math.isfinite(dp[m][n]):
        return PassageAlignment("BLOCKED_INSUFFICIENT_PASSAGE_CAPACITY",vocal,(),(),(),inf,("no-monotonic-path",))
    bundles=[]
    current=n
    for j in range(m,0,-1):
        previous = parent[j][current]
        if previous is None:
            raise RuntimeError("DP reconstruction failed")
        prev_count, passage_ix, times = previous
        if current>prev_count:
            displaced = tuple(i+1 for i,t in enumerate(times,prev_count)
                if priors[i] is not None and abs(priors[i]-t) > 1.)
            bundles.append(PassageBundle(prev_count+1,current,passage_ix,times,displaced))
        current=prev_count
    assert current==0
    bundles.reverse()
    proposal=[0.]*n
    assignment=[-1]*n
    for bundle in bundles:
        for i,t in enumerate(bundle.proposed_times,start=bundle.start_row-1):
            proposal[i]=t
            assignment[i]=bundle.passage_index
    status,issues=validate_passage_bundles(
        vocal,bundles,n,required_independent_ownership=False
    )
    return PassageAlignment(
       status if status!="STRUCTURALLY_VALID" else "UNVERIFIED_OCCURRENCE",
       vocal,tuple(bundles),tuple(assignment),tuple(proposal),
       dp[m][n],tuple(issues),
    )


def validate_passage_bundles(
    passages: Sequence[VocalPassage],
    bundles: Sequence[PassageBundle],
    row_count: int,
    *,
    required_independent_ownership: bool = True,
    proof_rows: frozenset[int] = frozenset(),
) -> tuple[str, tuple[str, ...]]:
    """Atomic structural guard; a missing proof rejects the ENTIRE bundle.

    No per-line salvage on failed passage ownership. Caller must not infer
    acoustic proof from vocal energy alone or from global CTC agreement.
    """
    errors: list[str] = []
    prev_line = 0
    prev_t = -float("inf")
    prev_passage = -1
    for group in bundles:
        if group.start_row != prev_line + 1:
            errors.append("row-gap-or-overlap")
        if group.end_row < group.start_row:
            errors.append("invalid-row-range")
        if not (0 <= group.passage_index < len(passages)):
            errors.append("invalid-passage-index")
            continue
        if group.passage_index <= prev_passage:
            errors.append("passage-order-not-strict")
        span = passages[group.passage_index]
        if len(group.proposed_times) != group.end_row - group.start_row + 1:
            errors.append("invalid-timing-count")
        for row,t in zip(range(group.start_row,group.end_row+1),group.proposed_times):
            if not math.isfinite(t) or not span.start<=t<=span.end:
                errors.append("outside-vocal-passage")
            if t <= prev_t:
                errors.append("nonmonotonic-time")
            if required_independent_ownership and row not in proof_rows:
                errors.append("independent-occurrence-proof-missing")
            prev_t = t
        prev_line = group.end_row
        prev_passage = group.passage_index
    if prev_line != row_count:
        errors.append("unassigned-lines")
    unique=tuple(dict.fromkeys(errors))
    return ("STRUCTURALLY_VALID" if not unique else "REJECTED_ATOMIC"),unique



def choose_atomic_central_passage(
    *,
    passage: VocalPassage,
    baseline_times: Sequence[float],
    local_ctc_times: Sequence[float],
    local_ctc_candidate_ids: Sequence[str],
    central_selected_ids: Sequence[str],
    independent_occurrence_witnesses: Sequence[bool],
    previous_bound: float | None = None,
    next_bound: float | None = None,
) -> tuple[tuple[float, ...], str]:
    """Atomic local-CTC transaction: either ALL rows survive Central, or none.

    This is a post-selection integrity gate, NOT a replacement for Central.
    Raw CTC confidence and mere vocal presence cannot authenticate occurrence.
    A full bundle requires independent witness per line, valid sequence and
    ownership boundaries, and Central selection of every proposed candidate.
    """
    original=tuple(float(x) for x in baseline_times)
    n=len(original)
    if not n or len(local_ctc_times)!=n or len(local_ctc_candidate_ids)!=n or len(central_selected_ids)!=n or len(independent_occurrence_witnesses)!=n:
        return original,"REJECTED_INVALID_BUNDLE_SHAPE"
    hypothesis=tuple(float(x) for x in local_ctc_times)
    if not all(math.isfinite(x) and passage.start<=x<=passage.end for x in hypothesis):
        return original,"REJECTED_OUTSIDE_VOCAL_PASSAGE"
    if not all(b>a for a,b in zip(hypothesis,hypothesis[1:])):
        return original,"REJECTED_NONMONOTONIC_PASSAGE"
    if previous_bound is not None and not hypothesis[0]>previous_bound:
        return original,"REJECTED_PREVIOUS_TAIL_OWNERSHIP"
    if next_bound is not None and not hypothesis[-1]<next_bound:
        return original,"REJECTED_NEXT_ANCHOR_BOUNDARY"
    if not all(independent_occurrence_witnesses):
        return original,"REJECTED_UNPROVED_OCCURRENCE"
    if len(set(local_ctc_candidate_ids))!=n or not all(local_ctc_candidate_ids):
        return original,"REJECTED_DUPLICATE_CANDIDATE_IDS"
    if tuple(local_ctc_candidate_ids)!=tuple(central_selected_ids):
        return original,"REJECTED_PARTIAL_CENTRAL_ACCEPTANCE"
    return hypothesis,"ACCEPTED_ATOMIC_BY_CENTRAL"