"""Common, exact MMS forced-CTC score mapping for global and segment paths."""
def ctc_alignment_confidence(ctc_score:float, *, has_start:bool=True)->float:
    if not has_start:
        return 0.0
    return 0.90 if ctc_score >= 0.08 else 0.70 if ctc_score >= 0.03 else 0.35
