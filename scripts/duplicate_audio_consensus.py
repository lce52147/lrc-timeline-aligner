"""S4 safe vocal evidence projection for lyrics repeated more than twice.

Only consecutive repeated occurrences in a stable, independent multi-lyric
displacement cluster may corroborate an already-selected line. No timing
change, no provenance authority, and no invented independent voice peaks.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import math


def consecutive_duplicate_vocal_support(
    *,
    lyric_indexes: Sequence[int],
    current_index: int,
    timestamps: Sequence[float],
    peaks: Mapping[int, Mapping[str, float] | None],
    stable_cluster_medians: Mapping[int, float],
    delta_tolerance_s: float = .10,
    peak_radius_s: float = .10,
) -> tuple[bool, dict[str, object] | None]:
    """Fail closed on discordant/ambiguous neighbors and unsupported peaks."""
    if (
        len(lyric_indexes)<2
        or len(set(lyric_indexes))!=len(lyric_indexes)
        or list(lyric_indexes)!=sorted(lyric_indexes)
        or current_index not in lyric_indexes
    ):
        return False,None
    neighbors=[]
    for first,second in zip(lyric_indexes,lyric_indexes[1:]):
        if current_index not in (first,second):continue
        if first<0 or second>=len(timestamps):return False,None
        expected=stable_cluster_medians.get(second-first)
        if expected is None or not math.isfinite(float(expected)):
            continue
        time_gap=float(timestamps[second])-float(timestamps[first])
        if not math.isfinite(time_gap):
            return False,None
        first_peak=peaks.get(first)
        second_peak=peaks.get(second)
        if not isinstance(first_peak,Mapping) or not isinstance(second_peak,Mapping):
            continue
        try:
            first_d=float(first_peak.get("delta_seconds",float("nan")))
            second_d=float(second_peak.get("delta_seconds",float("nan")))
        except (TypeError,ValueError):
            continue
        if not all(math.isfinite(q) for q in (first_d,second_d)):
            continue
        if not (
            abs(time_gap-float(expected))<=delta_tolerance_s
            and abs(first_d)<=peak_radius_s
            and abs(second_d)<=peak_radius_s
        ):
            continue
        neighbors.append({
            "entries":[first+1,second+1],
            "entry_delta":second-first,
            "observed_delta_seconds":round(time_gap,3),
            "median_delta_seconds":round(float(expected),3),
            "first_peak":dict(first_peak),
            "second_peak":dict(second_peak),
        })
    if not neighbors:
        return False,None
    # Two valid adjacent pairs may corroborate the middle occurrence. Neither
    # one is an independent phonetic/occurrence vote; it stays generic acoustic.
    if len(neighbors)>2:
        return False,None
    if len(neighbors)==1:
        return True,neighbors[0]
    # A middle lyric having two stable but conflicting section displacements
    # is not automatically a defect: both pair events are physically distinct.
    # Keep both provenance records rather than collapse to a fake global delta.
    return True,{
        "entries":[neighbors[0]["entries"][0],current_index+1,neighbors[1]["entries"][1]],
        "pair_details":neighbors,"consecutive":True,
    }
