"""Reviewer absolute offset calibration from *independent vocal audio*, not Central.

The calibration inputs are reviewer timestamps and waveform-derived onsets only.
No baseline/CUR/FIN timestamp enters offset estimation. Missing or ambiguous
acoustic anchors cause fail-closed no-offset, never a fabricated zero offset.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import statistics
from typing import Sequence

@dataclass(frozen=True)
class OffsetEstimate:
    offset_seconds: float | None
    sample_count: int
    status: str
    dispersion_ms: float | None = None

def anchor_peaks_from_vocal_features(
    frame_times: Sequence[float],
    rms_db: Sequence[float],
    onset_strength: Sequence[float],
    *,
    refractory_seconds: float = .30,
    min_relative_strength: float = .68,
    min_active_db_margin: float = 18.,
) -> tuple[float, ...]:
    """Use isolated-vocal spectral onset peaks with an absolute loudness gate."""
    if not (len(frame_times)==len(rms_db)==len(onset_strength)):
        raise ValueError("vocal feature arrays must align")
    active=sorted(float(db) for db in rms_db if math.isfinite(float(db)) and float(db)>-60)
    if len(active)<12:return ()
    med=statistics.median(active)
    threshold=med-min_active_db_margin
    contenders=[]
    for i in range(1,len(frame_times)-1):
        s=float(onset_strength[i])
        db=float(rms_db[i])
        t=float(frame_times[i])
        if not (math.isfinite(t) and math.isfinite(s) and math.isfinite(db)):
            continue
        if db<threshold or s<min_relative_strength:
            continue
        if s < float(onset_strength[i-1]) or s <= float(onset_strength[i+1]):
            continue
        contenders.append((t,s))
    chosen=[]
    # Strongest first: when overlapping peaks compete, use a single witness.
    for t,s in sorted(contenders,key=lambda p:-p[1]):
        if all(abs(t-p)>refractory_seconds for p in chosen):
            chosen.append(t)
    return tuple(sorted(chosen))

def calibrate_reviewer_vocal_offset(
    reviewer_times: Sequence[float | None],
    onset_peaks: Sequence[float],
    *,
    min_samples: int = 8,
    max_nearby_seconds: float = .35,
    max_mad_seconds: float = .08,
) -> OffsetEstimate:
    """Offset = reviewer clock minus isolated vocal onset time.

    This deliberately remains conservative. A nearby arbitrary vocal onset
    does not establish lyric identity, so only low-dispersion repeated matches
    can estimate a global clock shift. Such an offset is calibration, not an
    occurrence or timing ownership proof.
    """
    if min_samples<2 or max_nearby_seconds<=0:
        raise ValueError("invalid offset calibration parameters")
    peaks=tuple(sorted(float(t) for t in onset_peaks if math.isfinite(float(t))))
    matches=[]
    used=set()
    for reviewer in reviewer_times:
        if reviewer is None or not math.isfinite(reviewer):
            continue
        nearby=[(abs(float(reviewer)-t),i,t) for i,t in enumerate(peaks)
                if i not in used and abs(float(reviewer)-t)<max_nearby_seconds]
        # Unique, unambiguous strongest local acoustic onset.
        nearby.sort()
        if not nearby:
            continue
        if len(nearby)>1 and abs(nearby[1][0]-nearby[0][0])<.08:
            continue
        distance,index,onset=nearby[0]
        used.add(index)
        matches.append(float(reviewer)-onset)
    if len(matches)<min_samples:
        return OffsetEstimate(None,len(matches),"INSUFFICIENT_INDEPENDENT_VOCAL_ANCHORS")
    median=float(statistics.median(matches))
    mad=float(statistics.median(abs(v-median) for v in matches))
    # MAD alone misses a two-peak population when a bare majority agrees:
    # 6 clean and 5 wrong-occurrence anchors would otherwise give MAD=0.
    consistent=sum(abs(v-median)<=max_mad_seconds for v in matches)
    if mad>max_mad_seconds or consistent/len(matches)<.8:
        return OffsetEstimate(None,len(matches),"AMBIGUOUS_VOCAL_CLOCK_DISPERSION",round(mad*1000,2))
    # If there is not a detectable difference, carry the measured tiny
    # non-zero offset; never replace an absent value with 0.
    return OffsetEstimate(round(median,6),len(matches),"CALIBRATED_VOCAL_ONSETS",round(mad*1000,2))
