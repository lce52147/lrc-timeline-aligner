"""A6 posthoc strict acceptance: 7 fixed songs, two real CLI modes each.

This file is ONLY a grader: it never runs or configures inference.
Its BK times never enter segment DP, aligners or Central.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Mapping
import math

FIXED_SONGS=(
 "04 影色舞","09.夢遊","01 My Dearest","03. Choir ‘S’ Choir",
 "01.ミレニアの水槽","10.方舟","01.感情グラス")
NORMAL_CONTROLS=("10.方舟","01.感情グラス")
KAGE="04 影色舞"
DREAM="09.夢遊"


@dataclass(frozen=True)
class Score:
    song:str
    rows:int
    pco200:int
    gt1:int
    backtracks:int
    new_big_errors:tuple[int,...]
    mode:str
    segment_status:str
    trust_rows:int
    errors_cs:tuple[int,...]
    backtrack_pairs:tuple[int,...]


def score_song(*,song:str,mode:str,
        labels:Sequence[tuple[str,int]],
        produced:Sequence[tuple[str,int]],
        baseline:Sequence[tuple[str,int]]|None=None,
        segment_status:str="UNKNOWN",trust_rows:int=0)->Score:
    if not (len(labels)==len(produced) and labels):
        raise ValueError("STRICT_COUNT_MISMATCH")
    if baseline is not None and len(baseline)!=len(labels):
        raise ValueError("BASELINE_COUNT_MISMATCH")
    if not all(x[0]==y[0] and (baseline is None or x[0]==z[0])
               for x,y,z in zip(labels,produced,baseline or labels)):
        raise ValueError("STRICT_LYRIC_IDENTITY")
    if not all(type(x[1]) is int and x[1]>=0
               for row in (labels,produced,baseline or labels) for x in row):
        raise ValueError("INVALID_TIMESTAMPS")
    mistakes=tuple(abs(a[1]-b[1]) for a,b in zip(labels,produced))
    newly=tuple(i+1 for i,e in enumerate(mistakes)
               if e>100 and (baseline is None or
                             abs(labels[i][1]-baseline[i][1])<=100))
    backtrack_pairs=tuple(i+1 for i,(a,b) in enumerate(zip(produced,produced[1:])) if b[1]<a[1])
    backtracks=len(backtrack_pairs)
    return Score(song,len(mistakes),
       sum(e<=20 for e in mistakes),sum(e>100 for e in mistakes),backtracks,
       newly,mode,segment_status,trust_rows,mistakes,backtrack_pairs)


def gate_a6(scores:Sequence[Score])->dict[str,object]:
    by={(x.song,x.mode):x for x in scores}
    missing=[{"song":song,"mode":mode} for song in FIXED_SONGS
             for mode in ("segment","ctc") if (song,mode) not in by]
    if missing:
        return {"status":"INCOMPLETE_A6_PAIRING","missing":missing,
                "available":len(by)}
    blockers=[]
    inherited=[]
    for song in FIXED_SONGS:
        current=by[(song,"segment")]
        base=by[(song,"ctc")]
        if current.rows!=base.rows:
            blockers.append({"song":song,"reason":"BASELINE_ROWS_DIFFER"})
        new_pairs=sorted(set(current.backtrack_pairs)-set(base.backtrack_pairs))
        old_pairs=sorted(set(current.backtrack_pairs)&set(base.backtrack_pairs))
        if new_pairs:
            blockers.append({"song":song,"reason":"NEW_SEGMENT_BACKTRACK","pairs":new_pairs})
        if old_pairs:
            inherited.append({"song":song,"reason":"INHERITED_BACKTRACK","pairs":old_pairs})
        if song in NORMAL_CONTROLS:
            genuinely_new=tuple(i+1 for i,(e,f) in enumerate(zip(
                  current.errors_cs,base.errors_cs)) if e>100 and f<=100)
            if genuinely_new:
                blockers.append({"song":song,"reason":"CONTROL_NEW_GT1",
                                 "rows":genuinely_new})
        if song==KAGE:
            if current.pco200<40 or current.gt1>2:
                blockers.append({"song":song,"reason":"KAGE_TARGET_MISSED",
                                 "pco200":current.pco200,"gt1":current.gt1})
        if song==DREAM:
            if current.rows<27 or current.errors_cs[12]>50 or current.errors_cs[26]>50:
                blockers.append({"song":song,"reason":"DREAM_13_27_TARGET_MISSED",
                    "row13_cs":current.errors_cs[12] if current.rows>12 else None,
                    "row27_cs":current.errors_cs[26] if current.rows>26 else None})
    return {"status":"PASS_A6_REPRESENTATIVE" if not blockers else "BLOCKED_A6",
            "blockers":blockers,"inherited_backtracks":inherited,"pairs":len(FIXED_SONGS)}
