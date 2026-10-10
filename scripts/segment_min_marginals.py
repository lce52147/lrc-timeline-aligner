"""Exact min-sum forward/backward line-to-passage costs for monotone DP.

Independently auditable numerical core. No BK times, GPU, or threshold tuning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math

@dataclass(frozen=True)
class RowMarginal:
    entry:int
    best_passage:int | None
    alternative_passage:int | None
    best_cost:float
    alternative_cost:float
    confidence:float
    ambiguity:str

@dataclass(frozen=True)
class MarginalAudit:
    global_cost:float
    rows:tuple[RowMarginal,...]
    forward:tuple[tuple[float,...],...]
    backward:tuple[tuple[float,...],...]
    nonadjacent_ambiguous_rows:tuple[int,...]
    adjacent_ambiguous_rows:tuple[int,...]

def min_sum_passage_marginals(*,passage_count:int,row_count:int,
                             transition:Callable[[int,int,int],float],
                             max_lines_per_passage:int=12,
                             ambiguity_limit:float=.03)->MarginalAudit:
    """State: F[j][done] before passage j. A passage consumes 0..max lines."""
    if passage_count<=0 or row_count<=0 or max_lines_per_passage<1:
        raise ValueError('positive passage_count, row_count and capacity required')
    infinity=math.inf
    m,n=passage_count,row_count
    cache:dict[tuple[int,int,int],float]={}
    def step(j,done,count):
        key=(j,done,count)
        if key not in cache:
            v=float(transition(j,done,count))
            if math.isnan(v) or v==-math.inf:raise ValueError('invalid edge weight')
            cache[key]=v
        return cache[key]
    F=[[infinity]*(n+1) for _ in range(m+1)]
    F[0][0]=0.
    for j in range(m):
        for done in range(n+1):
            if not math.isfinite(F[j][done]):continue
            for count in range(min(max_lines_per_passage,n-done)+1):
                value=F[j][done]+step(j,done,count)
                if value<F[j+1][done+count]:F[j+1][done+count]=value
    B=[[infinity]*(n+1) for _ in range(m+1)]
    B[m][n]=0.
    for j in range(m-1,-1,-1):
        for done in range(n+1):
            for count in range(min(max_lines_per_passage,n-done)+1):
                value=step(j,done,count)+B[j+1][done+count]
                if value<B[j][done]:B[j][done]=value
    if not math.isfinite(F[m][n]):
        raise ValueError('no feasible complete monotone path')
    if abs(F[m][n]-B[0][0])>1e-7:
        raise AssertionError('forward/backward costs diverge')
    M=[[infinity]*m for _ in range(n)]
    for j in range(m):
        for done in range(n):
            if not math.isfinite(F[j][done]):continue
            for count in range(1,min(max_lines_per_passage,n-done)+1):
                value=F[j][done]+step(j,done,count)+B[j+1][done+count]
                if not math.isfinite(value):continue
                for entry_index in range(done,done+count):
                    if value<M[entry_index][j]:
                        M[entry_index][j]=value
    result=[]
    adjacent=[]
    nonadjacent=[]
    for index,costs in enumerate(M):
        ranked=sorted((v,j) for j,v in enumerate(costs) if math.isfinite(v))
        if not ranked:
            raise AssertionError('unreachable row despite complete path')
        best,j=ranked[0]
        second,other=ranked[1] if len(ranked)>1 else (infinity,None)
        confidence=second-best
        ambiguity='unique'
        if confidence<ambiguity_limit and other is not None:
            if abs(j-other)==1:
                ambiguity='adjacent'
                adjacent.append(index+1)
            else:
                ambiguity='nonadjacent'
                nonadjacent.append(index+1)
        result.append(RowMarginal(index+1,j,other,best,second,confidence,ambiguity))
    return MarginalAudit(F[m][n],tuple(result),
                         tuple(tuple(x) for x in F),
                         tuple(tuple(x) for x in B),
                         tuple(nonadjacent),tuple(adjacent))
