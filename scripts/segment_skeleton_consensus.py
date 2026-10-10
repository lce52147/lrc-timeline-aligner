"""Option A research: two whole-song passage DPs, acoustic-score joint selection.

Candidates originate exclusively from real local GPU CTC on vocal windows,
never from BK or human timestamps. The eventual whole-song selected passage
allocation MUST be re-aligned from scratch before any current is adopted.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math


@dataclass(frozen=True)
class TestedRow:
    passage:int
    time:float
    score:float
    source:str


@dataclass(frozen=True)
class AcousticJointPlan:
    status:str
    rows:tuple[TestedRow,...]
    total_cost:float
    sources:tuple[str,...]
    switches:int


def choose_acoustic_joint_passages(
    paths:Sequence[Sequence[TestedRow]],
    *,
    switch_penalty:float=.36,
    conservative_equal_weight:float=.07,
    min_spacing_seconds:float=.04,
)->AcousticJointPlan:
    fail=lambda code:AcousticJointPlan(code,(),math.inf,(),0)
    if not paths or not paths[0] or any(len(x)!=len(paths[0]) for x in paths):
        return fail("BLOCKED_SHAPE")
    n=len(paths[0])
    if len(paths)>3 or not (0<=switch_penalty<=2 and 0<=conservative_equal_weight<=2):
        return fail("BLOCKED_CONFIG")
    for path in paths:
        if any(row.passage<0 or not math.isfinite(row.time) or
               not math.isfinite(row.score) or row.score<0 or row.score>1
               or not row.source for row in path):
            return fail("BLOCKED_INVALID_ACOUSTIC_DATA")
    costs=[{} for _ in range(n)]
    for j in range(len(paths)):
        v=paths[j][0]
        costs[0][j]=(-math.log(max(.001,v.score))+(j!=0)*conservative_equal_weight,None)
    for row_no in range(1,n):
        for source_index,path in enumerate(paths):
            candidate=path[row_no]
            rank=-math.log(max(.001,candidate.score))
            if source_index:rank+=conservative_equal_weight
            best=None
            for old_source,(old_cost,_) in costs[row_no-1].items():
                before=paths[old_source][row_no-1]
                if candidate.time<=before.time+min_spacing_seconds:continue
                if candidate.passage<before.passage:continue
                boundary_penalty=switch_penalty if old_source!=source_index else 0.
                trial=(old_cost+rank+boundary_penalty,old_source)
                if best is None or trial<best:best=trial
            if best is not None:costs[row_no][source_index]=best
        if not costs[row_no]:return fail("BLOCKED_NO_MONOTONIC_JOINT_PATH")
    final=min(costs[-1],key=lambda x:(costs[-1][x][0],x))
    best_cost=costs[-1][final][0]
    sequence=[]
    for i in range(n-1,-1,-1):
        sequence.append(paths[final][i])
        final=costs[i][final][1]
        if final is None and i>0:return fail("BLOCKED_BROKEN_DP_LINK")
    sequence.reverse()
    switches=sum(a.source!=b.source for a,b in zip(sequence,sequence[1:]))
    return AcousticJointPlan("RESEARCH_JOINT_PATH_NO_AUTHORITY",
      tuple(sequence),round(best_cost,5),
      tuple(x.source for x in sequence),switches)
