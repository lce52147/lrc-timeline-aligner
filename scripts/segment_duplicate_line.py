"""Research-only repeated single-lyric acoustic occurrence hypotheses.

Supports songs where identical single lines recur too often for unique
multi-line chorus proof. Vocal DP is a weak structural prior, never an
independent lyric identity or a Central admission vote.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Sequence

import numpy as np
from segment_recurrence import scan_vocal_recurrences


@dataclass(frozen=True)
class DuplicateLineAcousticHit:
    row: int
    time: float
    similarity: float
    vocal_passage: int
    global_ctc_displacement_s: float


@dataclass(frozen=True)
class DuplicateLineResearch:
    status: str
    reference_row: int | None
    hits: tuple[DuplicateLineAcousticHit, ...]
    reason: str


def infer_duplicate_line_acoustic_hits(
    lyric_texts: Sequence[str],
    dp: object,
    global_ctc_priors: Sequence[float | None],
    features: np.ndarray,
    rms_db: np.ndarray,
    *,
    hop_seconds: float,
    min_occurrences: int = 4,
    min_reference_displacement_s: float = 7.0,
    min_similarity: float = .65,
    max_dp_deviation_s: float = 2.25,
    passage_slack_s: float = .35,
) -> DuplicateLineResearch:
    block=lambda reason: DuplicateLineResearch("BLOCKED_DUPLICATE_LINE",None,(),reason)
    n=len(lyric_texts)
    times=getattr(dp,"entry_times",())
    passage_ids=getattr(dp,"entry_passages",())
    passages=getattr(dp,"passages",())
    if not n or n!=len(global_ctc_priors) or len(times)!=n or len(passage_ids)!=n:
        return block("mismatched-line-dp-priors")
    lexical=[re.sub(r"\s+","",unicodedata.normalize("NFKC",t)).casefold() for t in lyric_texts]
    groups:dict[str,list[int]]={}
    for row,text in enumerate(lexical,1):
        if text:groups.setdefault(text,[]).append(row)
    candidates=[]
    for token,indices in groups.items():
        if len(indices)<min_occurrences:continue
        for row in indices:
            p=passages[passage_ids[row-1]]
            prior=global_ctc_priors[row-1]
            if prior is None or not math.isfinite(float(prior)):continue
            if not (4<=p.duration<=9):continue
            if abs(float(times[row-1])-prior)<min_reference_displacement_s:continue
            candidates.append((row,token,indices,p))
    if not candidates:
        return block("no-large-displacement-acoustic-reference")
    groups_qualified={x[1] for x in candidates}
    if len(groups_qualified)!=1:
        return block("multiple-distinct-duplicate-phrases")
    reference,token,indices,passage=min(candidates,key=lambda x:x[0])
    scan=scan_vocal_recurrences(
        features,rms_db,hop_seconds=hop_seconds,
        template_start=passage.start,template_end=passage.end,
        min_similarity=min_similarity,min_spacing_seconds=4.,
    )
    if scan.status!="RESEARCH_UNVERIFIED_OCCURRENCE":
        return block("unusable-acoustic-recurrence")
    hits=[h for h in scan.matches if h.similarity>=min_similarity]
    used:set[int]=set()
    matched=[]
    for row in indices:
        p=passages[passage_ids[row-1]]
        options=[]
        for j,h in enumerate(hits):
            if j in used or abs(h.time-float(times[row-1]))>max_dp_deviation_s:
                continue
            # A long singing segment can contain multiple repetitions; the
            # second/third lyric starts inside the same passage, not at its
            # beginning. Require actual vocal-window membership instead.
            if not (p.start-passage_slack_s<=h.time<=p.end+passage_slack_s):
                continue
            options.append((abs(h.time-float(times[row-1])),j,h))
        options.sort(key=lambda t:t[0])
        if not options:continue
        if len(options)>1 and abs(options[1][0]-options[0][0])<=.18:
            continue
        _,j,h=options[0]
        used.add(j)
        prior=global_ctc_priors[row-1]
        matched.append(DuplicateLineAcousticHit(
            row=row,time=h.time,similarity=h.similarity,
            vocal_passage=passage_ids[row-1],
            global_ctc_displacement_s=round(abs(float(times[row-1])-float(prior)),3)
              if prior is not None else float("nan"),
        ))
    if not any(h.row==reference for h in matched) or len(matched)<2:
        return block("insufficient-monotone-matched-occurrences")
    if any(b.time<=a.time or b.row<=a.row for a,b in zip(matched,matched[1:])):
        return block("acoustic-match-order-conflict")
    if not any(h.row>reference and h.global_ctc_displacement_s>=min_reference_displacement_s
               for h in matched):
        return block("no-second-severe-occurrence")
    return DuplicateLineResearch(
        "RESEARCH_UNVERIFIED_OCCURRENCE",reference,tuple(matched),
        "timbre-and-lexical-repeat-do-not-prove-lyric-occurrence-identity",
    )