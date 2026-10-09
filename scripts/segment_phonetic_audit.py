"""S1 Japanese independent-ASR phonetic disagreement audit, never timing proof.

A raw recognizer's Japanese text can have alternate kanji/kana readings.
Romanization-based edit proximity is useful for triage only: it is NOT an
independent known-lyric occurrence binding, nor a reliable onset time. This
module never returns an eligible Central candidate or a timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math


@dataclass(frozen=True)
class PhoneticAudit:
    status: str
    target_phonetic: str
    matched_recognized_phonetic: str
    match_similarity: float
    competing_target_similarity: float
    independent_occurrence_proven: bool=False
    direct_onset_proven: bool=False


def _edit_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    row=list(range(len(right)+1))
    for a in left:
        nxt=[row[0]+1]
        for j,b in enumerate(right):
            nxt.append(min(nxt[-1]+1,row[j+1]+1,row[j]+(a!=b)))
        row=nxt
    return row[-1]


def _best_window(target: str, observed: str) -> tuple[float,str]:
    """Bounded contiguous romaji substring match, never nearest-time mapping."""
    if not target or not observed or len(target)>128 or len(observed)>2048:
        return 0.0,""
    low=max(1,int(math.floor(.65*len(target))))
    high=min(len(observed),int(math.ceil(1.35*len(target))))
    best=(0.0,"")
    for n in range(low,high+1):
        for i in range(len(observed)-n+1):
            part=observed[i:i+n]
            score=1-_edit_distance(target,part)/max(len(target),len(part))
            if score>best[0]:
                best=(round(score,5),part)
    return best


def audit_japanese_asr_phonetics(
    *,
    target_lyric: str,
    raw_asr_text: str,
    other_lyric_texts: Sequence[str],
    to_romaji: Callable[[str],str|None],
    min_target_chars: int=9,
    minimum_similarity: float=.90,
    unique_margin: float=.075,
) -> PhoneticAudit:
    def compact(text: str) -> str:
        # phonetic converter may be genuinely unavailable: never call the
        # Unicode spelling an independent phonetic model.
        return "".join(c for c in (to_romaji(text) or "").lower() if c.isascii() and c.isalpha())
    target=compact(target_lyric)
    raw=compact(raw_asr_text)
    unknown=lambda name,t="",o="",s=0.,c=0.:PhoneticAudit(name,t,o,s,c)
    if len(target)<min_target_chars or not raw:
        return unknown("BLOCKED_PHONETIC_PROFILE",target)
    observed_score,observed=_best_window(target,raw)
    competitor_score=max((
        _best_window(compact(value),raw)[0]
        for value in other_lyric_texts
        if compact(value) and compact(value)!=target
    ),default=0.0)
    # A homophonic alternative (same latinized text) cannot be disambiguated.
    phonetics=[compact(value) for value in other_lyric_texts]
    if target in phonetics:
        return unknown("BLOCKED_SAME_PHONETIC_LYRIC_EXISTS",target,observed,observed_score,competitor_score)
    if observed_score<minimum_similarity:
        return unknown("BLOCKED_INDEPENDENT_ASR_PHONETIC_MISMATCH",target,observed,observed_score,competitor_score)
    if observed_score-competitor_score<unique_margin:
        return unknown("BLOCKED_PHONETIC_NEAREST_LYRIC_AMBIGUOUS",target,observed,observed_score,competitor_score)
    # High lexical similarity is still not a first-char audio alignment or
    # independent occurrence binding; this may only generate review targets.
    return unknown("REVIEW_ONLY_PHONETICALLY_SIMILAR",target,observed,observed_score,competitor_score)
