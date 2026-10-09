"""S1 research-only CTC boundary competition gate.

Low-confidence edge rows are permitted to change passage only when the
alternative improves independent acoustic CTC fit for the entire neighbor pair.
No BK, reviewer or known-right timestamps; no direct Central authority.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math


@dataclass(frozen=True)
class CtcBoundaryRow:
    row: int
    time: float
    score: float


@dataclass(frozen=True)
class CtcBoundaryChoice:
    status: str
    average_gain: float
    moved_gain: float
    reason: str


def select_ctc_boundary_variant(
    old: Sequence[CtcBoundaryRow],
    alternative: Sequence[CtcBoundaryRow],
    *,
    moved_row: int,
    min_mean_gain: float = .04,
    min_moved_score: float = .08,
    min_moved_gain: float = .10,
    max_unmoved_score_drop: float = .14,
) -> CtcBoundaryChoice:
    """Whole-boundary atomic validation; reject if even one invariant fails."""
    reject=lambda reason: CtcBoundaryChoice('REJECTED_BOUNDARY',0.,0.,reason)
    if not old or len(old)!=len(alternative):
        return reject('row-count-mismatch')
    keys=[x.row for x in old];alt_keys=[x.row for x in alternative]
    if keys!=list(range(keys[0],keys[0]+len(keys))) or keys!=alt_keys or moved_row not in keys:
        return reject('row-identity-or-order-mismatch')
    values=[x.time for x in alternative]
    if any(not math.isfinite(x.time) or not math.isfinite(x.score) or
           x.score<0 or x.score>1 for x in (*old,*alternative)):
        return reject('nonfinite-or-out-of-range')
    if any(b<=a for a,b in zip(values,values[1:])):
        return reject('nonmonotonic-timestamps')
    row_ix=keys.index(moved_row)
    old_moved=old[row_ix].score;new_moved=alternative[row_ix].score
    gain=sum(new.score-old_score.score for new,old_score in zip(alternative,old))/len(old)
    moved_gain=new_moved-old_moved
    if new_moved<min_moved_score or moved_gain<min_moved_gain:
        return CtcBoundaryChoice('REJECTED_BOUNDARY',round(gain,6),round(moved_gain,6),'moved-row-unproved')
    if any(old[i].score-alternative[i].score>max_unmoved_score_drop
           for i in range(len(old)) if i!=row_ix):
        return CtcBoundaryChoice('REJECTED_BOUNDARY',round(gain,6),round(moved_gain,6),'other-row-score-regression')
    if gain<min_mean_gain:
        return CtcBoundaryChoice('REJECTED_BOUNDARY',round(gain,6),round(moved_gain,6),'insufficient-net-ctc-evidence')
    return CtcBoundaryChoice(
        'RESEARCH_ACCEPTED_NOT_CENTRAL',round(gain,6),round(moved_gain,6),
        'evidence-strength-only-not-occurrence-proof',
    )
