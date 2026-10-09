"""Reviewer clock calibration from *explicit human-verified* time references.

This function is supervised and MUST NOT be used by independent auto mode or
as an unbiased score on its own training rows. The caller must hold out
disjoint lyric rows for validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping,Sequence
import math
import statistics


@dataclass(frozen=True)
class HumanCheckedOffsetCalibration:
    status: str
    offsets_s: dict[str,float]
    qualified_samples: dict[str,int]
    median_absolute_deviation_s: dict[str,float]
    source_type: str="human-reviewed-bk"
    independent_auto_eligible: bool=False


def calibrate_reviewers_against_human_gold(
    raw_rows: Sequence[Mapping[str,object]],
    human_gold_seconds: Sequence[float],
    *,
    calibration_row_indexes: Sequence[int],
    reviewers: Sequence[str]=("HUBP","WX","XLSR"),
    max_abs_delta_s: float=.35,
    max_mad_s: float=.06,
    min_samples: int=5,
) -> HumanCheckedOffsetCalibration:
    block=lambda code: HumanCheckedOffsetCalibration(code,{},{},{})
    if len(raw_rows)!=len(human_gold_seconds) or len(raw_rows)<2:
        return block("BLOCKED_WRONG_REFERENCE_SHAPE")
    if (not calibration_row_indexes
        or len(set(calibration_row_indexes))!=len(calibration_row_indexes)
        or any(type(x) is not int or x<0 or x>=len(raw_rows)
               for x in calibration_row_indexes)
        or not reviewers or len(set(reviewers))!=len(reviewers)):
        return block("BLOCKED_CALIBRATION_SPLIT")
    if not (.05<max_abs_delta_s<2 and 0<max_mad_s<.5 and min_samples>=2):
        return block("BLOCKED_PARAMETERS")
    offsets:dict[str,float]={}
    count:dict[str,int]={}
    dispersions:dict[str,float]={}
    for reviewer in reviewers:
        differences=[]
        for i in calibration_row_indexes:
            row=raw_rows[i]
            if not isinstance(row,Mapping):
                continue
            times=row.get("reviewer_times")
            if not isinstance(times,Mapping):continue
            raw=times.get(reviewer)
            gold=human_gold_seconds[i]
            if (not isinstance(raw,(float,int)) or not isinstance(gold,(float,int))
               or not math.isfinite(float(raw)) or not math.isfinite(float(gold))):
                continue
            delta=float(raw)-float(gold)
            if abs(delta)<=max_abs_delta_s:
                differences.append(delta)
        count[reviewer]=len(differences)
        if len(differences)<min_samples:continue
        center=float(statistics.median(differences))
        mad=float(statistics.median(abs(q-center) for q in differences))
        dispersions[reviewer]=round(mad,6)
        if mad>max_mad_s:continue
        offsets[reviewer]=round(center,6)
    status=(
        "HUMAN_BK_ASSISTED_CALIBRATED_NOT_INDEPENDENT_AUTO"
        if offsets else "BLOCKED_NO_COHERENT_HUMAN_BK_ANCHORS"
    )
    return HumanCheckedOffsetCalibration(status,offsets,count,dispersions)
