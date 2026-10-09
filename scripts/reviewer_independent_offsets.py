"""S2 pairwise reviewer clock differences, independent of baseline and BK.

Only raw reviewer timestamps are accepted. The absolute time origin cannot be
identified from reviewer comparisons alone, so the returned zero-mean gauge is
a *relative* calibration. This does NOT prove the global audio onset offset.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping,Sequence
import math
import statistics


@dataclass(frozen=True)
class PairwiseOffsetResult:
    status: str
    offsets_s: dict[str,float]
    pair_sample_counts: dict[str,int]
    pair_residuals_s: dict[str,float]
    absolute_offset_identifiable: bool=False


def independent_pairwise_offsets(
    rows: Sequence[Mapping[str,object]],
    *,
    reviewers: Sequence[str]=("HUBP","WX","XLSR"),
    max_pair_delta_s: float=.20,
    min_samples_per_pair: int=6,
    max_pair_cycle_residual_s: float=.08,
) -> PairwiseOffsetResult:
    """Estimate a robust relative clock gauge with no backend timestamps.

    For each pair only near-coincident raw reviewer clocks participate.
    Disagreement outside this fixed window is *ignored*, never pulled toward
    baseline or BK. Graph non-connectivity or cycle inconsistency fails closed.
    """
    names=tuple(reviewers)
    blocked=lambda why: PairwiseOffsetResult(why,{},{},{})
    if not names or len(set(names))!=len(names):
        return blocked("BLOCKED_REVIEWER_IDENTITIES")
    if not (.01<max_pair_delta_s<2 and min_samples_per_pair>=2
            and 0<max_pair_cycle_residual_s<.5):
        return blocked("BLOCKED_CALIBRATION_PARAMETERS")
    edges:dict[tuple[str,str],float]={}
    counts:dict[str,int]={}
    for i,first in enumerate(names):
        for second in names[i+1:]:
            observations=[]
            for row in rows:
                if not isinstance(row,Mapping):
                    continue
                raw=row.get("reviewer_times")
                if not isinstance(raw,Mapping):continue
                one,two=raw.get(first),raw.get(second)
                if (not isinstance(one,(int,float)) or not isinstance(two,(int,float))
                        or not math.isfinite(float(one)) or not math.isfinite(float(two))):
                    continue
                delta=float(one)-float(two)
                if abs(delta)<=max_pair_delta_s:
                    observations.append(delta)
            key=f"{first}:{second}"
            counts[key]=len(observations)
            if len(observations)<min_samples_per_pair:continue
            edges[(first,second)]=float(statistics.median(observations))
    if not edges:
        return PairwiseOffsetResult("BLOCKED_NO_PAIRWISE_EVIDENCE",{},counts,{})
    relative={names[0]:0.}
    remaining=True
    while remaining:
        remaining=False
        for (first,second),delta in edges.items():
            if first in relative and second not in relative:
                relative[second]=relative[first]-delta
                remaining=True
            elif second in relative and first not in relative:
                relative[first]=relative[second]+delta
                remaining=True
    if len(relative)!=len(names):
        return PairwiseOffsetResult("BLOCKED_DISCONNECTED_REVIEWERS",{},counts,{})
    residuals={
        f"{first}:{second}":round(relative[first]-relative[second]-delta,6)
        for (first,second),delta in edges.items()
    }
    if any(abs(value)>max_pair_cycle_residual_s for value in residuals.values()):
        return PairwiseOffsetResult("BLOCKED_PAIRWISE_CYCLE_INCONSISTENT",{},counts,residuals)
    # Pairwise differences determine the offsets only up to a constant.
    # Zero-mean is an explicit gauge choice, NOT an absolute correction.
    mean=statistics.mean(relative.values())
    offsets={name:round(relative[name]-mean,6) for name in names}
    return PairwiseOffsetResult("RELATIVE_CLOCK_ONLY_UNVERIFIED_ABSOLUTE",
                               offsets,counts,residuals)
