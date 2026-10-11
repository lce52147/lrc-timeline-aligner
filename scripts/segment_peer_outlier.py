"""BK-blind evidence admission: contradictory solitary WhisperX fused onset.

Only on activated segment skeletons, and only when two separately
produced existing candidate sources (CTC/vocal and raw-ASR/vocal) agree
within the ordinary 0.5s onset association window. A distant WhisperX
fusion onset remains diagnostic but is not authoritative timing evidence.
No new clocks or trust signals are generated; Central is unchanged.
"""
AGREEMENT_WINDOW_SECONDS = .5
DISTANT_WINDOW_SECONDS = .5

def exclude_uncorroborated_whisperx_fusion(candidates):
    confirmed=[x for x in candidates if x.source=="ctc-vocal-onset-confirmed"
               and x.acoustic_evidence is not None
               and x.direct_onset_evidence is not None]
    raw=[x for x in candidates if x.source=="raw-vocal-independent-fusion"
         and x.acoustic_evidence is not None
         and x.direct_onset_evidence is not None]
    corroborated=[]
    for c in confirmed:
        for r in raw:
            if abs(c.written_time.seconds-r.written_time.seconds)<=AGREEMENT_WINDOW_SECONDS:
                corroborated.append((c,r))
    if not corroborated:return tuple(candidates),()
    excluded=[]
    kept=[]
    for x in candidates:
        if x.source!="whisperx-vocal-independent-fusion":
            kept.append(x)
            continue
        timestamp=x.written_time.seconds
        is_outlier=any(
            timestamp<min(c.written_time.seconds,r.written_time.seconds)-DISTANT_WINDOW_SECONDS
            or timestamp>max(c.written_time.seconds,r.written_time.seconds)+DISTANT_WINDOW_SECONDS
            for c,r in corroborated)
        if is_outlier:excluded.append(x)
        else:kept.append(x)
    return tuple(kept),tuple(excluded)
