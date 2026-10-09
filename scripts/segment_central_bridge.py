"""Fail-closed bridge from independently bounded S1 local CTC to Central.

This module only stages hypotheses in an in-memory report copy. The existing
Central collector and selector decide whether any candidate is admissible.
No BK clock, raw ASR impersonation, fabricated onset, or file writes.
"""
from __future__ import annotations

import copy
import math
from typing import Sequence, Mapping, Any


def stage_segment_constrained_ctc(
    report: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    entry_count: int,
    source_revision: str,
) -> tuple[dict[str, Any], str]:
    """Stage complete, ordered token-level candidate bundles or nothing.

    Each row requires a real CTC token span array and passage metadata.
    Every row must appear exactly once. Source revision binds the local run,
    and is NOT independent occurrence or onset evidence.
    """
    assignments=report.get("assignments")
    if not isinstance(assignments,list) or len(assignments)!=entry_count or entry_count<1:
        return report,"BLOCKED_REPORT_SHAPE"
    if not isinstance(source_revision,str) or len(source_revision.strip())<12:
        return report,"BLOCKED_SOURCE_REVISION"
    if len(rows)!=entry_count:
        return report,"BLOCKED_PARTIAL_SONG"
    by_row={}
    for record in rows:
        if not isinstance(record,Mapping):
            return report,"BLOCKED_ROW_SHAPE"
        number=record.get("row")
        if type(number) is not int or number<1 or number>entry_count or number in by_row:
            return report,"BLOCKED_ROW_IDENTITY"
        start=record.get("time")
        score=record.get("ctc_score")
        spans=record.get("token_spans")
        passage=record.get("passage")
        if (
            not isinstance(start,(int,float)) or not math.isfinite(start)
            or not isinstance(score,(int,float)) or not math.isfinite(score)
            or not 0<=score<=1 or not isinstance(spans,list) or not spans
            or type(passage) is not int or passage<0
        ):
            return report,"BLOCKED_INCOMPLETE_CTC_EVIDENCE"
        if not all(isinstance(x,dict) and isinstance(x.get("start"),(int,float))
                   and isinstance(x.get("end"),(int,float))
                   and isinstance(x.get("score"),(int,float))
                   and math.isfinite(x["start"]) and math.isfinite(x["end"]) and
                   math.isfinite(x["score"]) and x["start"]<x["end"] and
                   0<=x["score"]<=1 for x in spans):
            return report,"BLOCKED_INVALID_TOKEN_SPANS"
        if abs(float(spans[0]["start"])-float(start))>.021:
            return report,"BLOCKED_OPENING_SPAN_MISMATCH"
        if any(spans[i]["start"]+1e-6<spans[i-1]["start"] for i in range(1,len(spans))):
            return report,"BLOCKED_NONMONOTONIC_SPANS"
        by_row[number]=record
    if set(by_row)!=set(range(1,entry_count+1)):
        return report,"BLOCKED_ROW_COVERAGE"
    sorted_rows=[by_row[i] for i in range(1,entry_count+1)]
    if any(float(a["time"])>=float(b["time"]) for a,b in zip(sorted_rows,sorted_rows[1:])):
        return report,"BLOCKED_NONMONOTONIC_SONG"
    if any(int(a["passage"])>int(b["passage"]) for a,b in zip(sorted_rows,sorted_rows[1:])):
        return report,"BLOCKED_NONMONOTONIC_PASSAGE"
    result=copy.deepcopy(report)
    for i,record in enumerate(sorted_rows):
        if not isinstance(result["assignments"][i],dict):
            return report,"BLOCKED_ASSIGNMENT_SHAPE"
        raw_spans=copy.deepcopy(record["token_spans"])
        hypothesis={
            "source":"segment-constrained-ctc",
            "stage":"s1-structural-vocal-passage-ctc",
            "raw_candidate_time":float(record["time"]),
            "confidence":float(record["ctc_score"]),
            "spans":raw_spans,
            "identity_support":"supported" if any(x["score"]>=.08 for x in raw_spans) else "unavailable",
            "sequence_support":"supported",
            "acoustic_support":"unavailable",
            "direct_onset_support":"unavailable",
            "independent_content_identity":False,
            "authoritative":False,
            "parent":{"source_revision":source_revision,"passage_index":int(record["passage"])},
            "window":{"mode":"vocal-passage-constrained","reused_demucs":True},
        }
        assignment=result["assignments"][i]
        existing=assignment.get("alignment_hypotheses")
        if existing is not None and not isinstance(existing,list):
            return report,"BLOCKED_HYPOTHESIS_SHAPE"
        assignment["alignment_hypotheses"]=[
            x for x in (existing or [])
            if not (isinstance(x,dict) and x.get("source")=="segment-constrained-ctc")
        ]+[hypothesis]
    return result,"STAGED_FOR_CENTRAL_NO_AUTHORITY"
