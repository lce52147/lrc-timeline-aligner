"""LRC13 Option A: full-song vocal-passage skeleton, never BK times."""
from __future__ import annotations
from dataclasses import dataclass
import math
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Sequence
from segment_min_marginals import min_sum_passage_marginals
from segment_structure import VocalPassage, merge_vocal_passages

MERGE_GAP_S=.65
MIN_PASSAGE_S=.38
MAX_INTERLINE_SILENCE=3.0
MAX_LINES_PER_PASSAGE=12
MIN_PASSAGE_LINE_RATIO=.20
MAX_SINGLE_PASSAGE_DURATION_RATIO=.60
MIN_GLOBAL_MARGIN=.03
MAX_DP_COST_PER_ROW=2.8
W_ASR=1.0
CTC_PRIOR_WEIGHT=.035
MAX_CTC_PRIOR_PENALTY=4.
SKIP_PENALTY=.70

@dataclass(frozen=True)
class SkeletonAssignment:
    passage_index:int
    initial_time_s:float
    confidence:float

@dataclass(frozen=True)
class SkeletonPlan:
    status:str
    reason:str
    passages:tuple[VocalPassage,...]
    assignments:tuple[SkeletonAssignment,...]
    groups:tuple[tuple[int,int,int],...]
    dp_cost:float
    dp_margin:float
    diagnostics:dict[str,object]

def _text_for_match(value:str, romanizer:object=None)->str:
    """Stable, BK-free unicode/romaji normalization for lyric and raw ASR."""
    clean=unicodedata.normalize('NFKC',value).casefold()
    if callable(romanizer):
        candidate=romanizer(clean)
        if isinstance(candidate,str) and candidate.strip():
            clean=unicodedata.normalize('NFKC',candidate).casefold()
    return ''.join(ch for ch in clean if ch.isalnum())

def _dice_bigram(a:str,b:str)->float:
    if not a or not b:return 0.
    if len(a)<2 or len(b)<2:return float(a==b)
    x=Counter(a[i:i+2] for i in range(len(a)-1))
    y=Counter(b[i:i+2] for i in range(len(b)-1))
    overlap=sum((x&y).values())
    return 2.0*overlap/(sum(x.values())+sum(y.values()))

def _asr_for_passages(passages:Sequence[VocalPassage],
                      events:Sequence[tuple[float,float,str]],
                      romanizer:object=None)->tuple[str,...]:
    """Temporal overlap only; paragraph boundaries, never synthetic line times."""
    seen=[[] for _ in passages]
    for event in events:
        if not isinstance(event,(tuple,list)) or len(event)!=3:continue
        start,end,text=event
        if not (isinstance(start,(int,float)) and isinstance(end,(int,float))
                and math.isfinite(start) and math.isfinite(end) and end>start
                and isinstance(text,str) and text.strip()):
            continue
        content=_text_for_match(text,romanizer)
        if not content:continue
        for j,passage in enumerate(passages):
            if min(end,passage.end)>max(start,passage.start):
                seen[j].append(content)
    return tuple(''.join(words) for words in seen)

def _lengths_from_text(texts:Sequence[str],romanizer:object=None)->tuple[int,...]:
    lengths=[]
    for t in texts:
        reading=romanizer(t) if callable(romanizer) else t
        if not isinstance(reading,str):reading=t
        lengths.append(max(1,sum(c.isalnum() for c in reading)))
    return tuple(lengths)

def _path_cost(passage:VocalPassage,counts:Sequence[int],all_chars:int,
               total_voice:float,consumed_chars:int,voice_before:float,
               priors:Sequence[float|None],start_index:int,
               *,lyrics_match:Sequence[str]=(),asr_match:str='',
               asr_weight:float=W_ASR)->float:
    actual=sum(counts)
    expected=max(.12,total_voice*actual/max(1,all_chars))
    ratio=max(passage.duration/expected,.001)
    density=1.12*abs(math.log(ratio))
    progress_lyric=(consumed_chars+actual/2)/max(all_chars,1)
    progress_voice=(voice_before+passage.duration/2)/max(total_voice,.1)
    progress=.85*abs(progress_lyric-progress_voice)
    prior_cost=0.;total=max(1,actual);cumulative=0.
    for k,length in enumerate(counts):
        t=passage.start+.04+max(.0,passage.duration-.08)*cumulative/total
        c=priors[start_index+k]
        if c is not None and math.isfinite(c):
            prior_cost+=CTC_PRIOR_WEIGHT*min(abs(t-c),MAX_CTC_PRIOR_PENALTY)
        cumulative+=length
    crowding=.40*max(0.,len(counts)/max(passage.duration,.2)-2.25)
    content=(asr_weight*(1.0-_dice_bigram(''.join(lyrics_match),asr_match))
             if asr_match and lyrics_match else 0.0)
    return density+progress+prior_cost+crowding+content

def plan_segment_skeleton(segments:Sequence[tuple[float,float]], *,
    duration:float,lyric_texts:Sequence[str],romanizer:object=None,
    ctc_priors:Sequence[float|None]|None=None,
    raw_asr_events:Sequence[tuple[float,float,str]]=())->SkeletonPlan:
    """Monotonic global Viterbi: 2 independent best paths, no unique-repeat anchor."""
    vocal=merge_vocal_passages(segments,duration=duration,
                              merge_gap_s=MERGE_GAP_S,min_duration_s=MIN_PASSAGE_S)
    n,m=len(lyric_texts),len(vocal)
    blocked=lambda reason,**kw:SkeletonPlan("FALLBACK_CTC",reason,vocal,(),(),math.inf,0.,
                                          {"passages":m,"rows":n,**kw})
    if not n or not m or not math.isfinite(duration) or duration<=0:
        return blocked("NO_VOCAL_OR_LYRICS")
    if m<n*MIN_PASSAGE_LINE_RATIO:
        return blocked("TOO_FEW_PASSAGES",ratio=round(m/n,4))
    if max(v.duration for v in vocal)>MAX_SINGLE_PASSAGE_DURATION_RATIO*duration:
        return blocked("DOMINANT_PASSAGE")
    lengths=_lengths_from_text(lyric_texts,romanizer)
    match_texts=tuple(_text_for_match(t,romanizer) for t in lyric_texts)
    asr_by_passage=_asr_for_passages(vocal,raw_asr_events,romanizer)
    priors=tuple(ctc_priors) if ctc_priors is not None else (None,)*n
    if len(priors)!=n:return blocked("INVALID_PRIORS")
    if m*MAX_LINES_PER_PASSAGE<n:return blocked("INSUFFICIENT_CAPACITY")
    prefix=[0]
    for v in lengths:prefix.append(prefix[-1]+v)
    voice_before=[0.]
    for v in vocal:voice_before.append(voice_before[-1]+v.duration)
    total_voice=voice_before[-1]
    dp=[[(math.inf,()),(math.inf,())] for _ in range(n+1)]
    dp[0]=[(0.,()),(math.inf,())]
    for j,passage in enumerate(vocal):
        nxt=[[(math.inf,()),(math.inf,())] for _ in range(n+1)]
        for done,paths in enumerate(dp):
            for cost,path in paths:
                if not math.isfinite(cost):continue
                for count in range(min(MAX_LINES_PER_PASSAGE,n-done)+1):
                    increment=SKIP_PENALTY if count==0 else _path_cost(
                        passage,lengths[done:done+count],prefix[-1],total_voice,
                        prefix[done],voice_before[j],priors,done,
                        lyrics_match=match_texts[done:done+count],
                        asr_match=asr_by_passage[j])
                    score=cost+increment
                    slot=nxt[done+count];step=(*path,count)
                    if score<slot[0][0]-1e-10:
                        slot[1]=slot[0];slot[0]=(score,step)
                    elif step!=slot[0][1] and score<slot[1][0]-1e-10:
                        slot[1]=(score,step)
        dp=nxt
    best,runner=dp[n]
    if not math.isfinite(best[0]):return blocked("NO_COMPLETE_DP_PATH")
    margin=runner[0]-best[0] if math.isfinite(runner[0]) else 999.
    # P2: independent min-sum forward/backward over the exact same graph
    # as the monotonic Viterbi path.  Global runner-up margin is diagnostic
    # only; one ambiguous adjacent boundary must not reject the whole song.
    @lru_cache(maxsize=None)
    def transition(j:int,done:int,count:int)->float:
        if count==0:
            return SKIP_PENALTY
        return _path_cost(
            vocal[j],lengths[done:done+count],prefix[-1],total_voice,
            prefix[done],voice_before[j],priors,done,
            lyrics_match=match_texts[done:done+count],
            asr_match=asr_by_passage[j])
    try:
        marginal=min_sum_passage_marginals(
            passage_count=m,row_count=n,transition=transition,
            max_lines_per_passage=MAX_LINES_PER_PASSAGE,
            ambiguity_limit=MIN_GLOBAL_MARGIN)
    except ValueError as error:
        return blocked("MIN_MARGINAL_UNAVAILABLE",detail=str(error)[:120])
    if abs(marginal.global_cost-best[0])>1e-6:
        return blocked("MIN_MARGINAL_OBJECTIVE_DRIFT",
                       expected=round(best[0],6),
                       computed=round(marginal.global_cost,6))
    structural=tuple(marginal.nonadjacent_ambiguous_rows)
    if len(structural)/n>.10:
        return blocked("STRUCTURAL_DP_AMBIGUITY",
                       nonadjacent_rows=list(structural),
                       nonadjacent_fraction=round(len(structural)/n,5))
    if best[0]/n>MAX_DP_COST_PER_ROW:
        return blocked("HIGH_DP_COST",cost_per_row=round(best[0]/n,4))
    groups=[];assignments=[];idx=0
    for j,(passage,count) in enumerate(zip(vocal,best[1])):
        if not count:continue
        group_lengths=lengths[idx:idx+count]
        total=sum(group_lengths);running=0
        for length in group_lengths:
            t=passage.start+min(.10,passage.duration*.035)
            t+=max(0.,passage.duration-min(.20,passage.duration*.07))*running/max(1,total)
            row_marginal=marginal.rows[len(assignments)]
            assignments.append(SkeletonAssignment(j,round(t,3),
               round(max(0.,row_marginal.confidence),5)
               if math.isfinite(row_marginal.confidence) else 999.0))
            running+=length
        groups.append((idx,idx+count,j));idx+=count
    if idx!=n:return blocked("INCOMPLETE_ROW_ASSIGNMENT")
    times=[q.initial_time_s for q in assignments]
    if any(b<=a+.019 for a,b in zip(times,times[1:])):
        return blocked("NONMONOTONIC_PROPOSED_STARTS")
    return SkeletonPlan("READY","OK",vocal,tuple(assignments),tuple(groups),
      round(best[0],6),round(margin,6),{
       "passages":m,"rows":n,"ratio":round(m/n,4),
       "mean_length":round(sum(lengths)/n,2),
       "raw_asr_events_seen":len(raw_asr_events),
       "DP_cost_per_row":round(best[0]/n,5),
       "DP_margin":round(margin,6),
       "P2_min_marginal":True,
       "adjacent_ambiguous_rows":list(marginal.adjacent_ambiguous_rows),
       "nonadjacent_ambiguous_rows":list(structural),
       "per_line_min_marginal":[
           {"entry":row.entry,"best_passage":row.best_passage,
            "second_passage":row.alternative_passage,
            "confidence":round(row.confidence,6) if math.isfinite(row.confidence) else None,
            "ambiguity":row.ambiguity}
           for row in marginal.rows],
       "params":{"merge_gap_s":MERGE_GAP_S,"min_passage_s":MIN_PASSAGE_S,
        "max_interline_silence":MAX_INTERLINE_SILENCE,
        "max_lines_per_passage":MAX_LINES_PER_PASSAGE,
        "min_passage_line_ratio":MIN_PASSAGE_LINE_RATIO,
        "max_dominant_ratio":MAX_SINGLE_PASSAGE_DURATION_RATIO,
        "min_dp_margin":MIN_GLOBAL_MARGIN,
        "max_cost_per_row":MAX_DP_COST_PER_ROW}
      })

def validate_local_ctc(plan:SkeletonPlan,rows:Sequence[dict[str,object]], *,
                       duration:float,minimum_score:float=.0002)->tuple[bool,str]:
    if plan.status!="READY":return False,plan.reason
    if len(rows)!=len(plan.assignments):return False,"LOCAL_ROW_COUNT"
    last=-1.
    for row,match in zip(rows,plan.assignments):
        t=row.get("start");s=row.get("ctc_score")
        if not isinstance(t,(int,float)) or not math.isfinite(t):
            return False,"LOCAL_INVALID_START"
        if not isinstance(s,(int,float)) or not math.isfinite(s):
            return False,"LOCAL_INVALID_SCORE"
        passage=plan.passages[match.passage_index]
        if not passage.start-.35<=t<=passage.end+.35 or not 0<=t<=duration:
            return False,"LOCAL_OUTSIDE_PASSAGE"
        if t<=last+.019:return False,"LOCAL_NONMONOTONIC"
        spans=row.get("token_spans")
        if not isinstance(spans,list) or not spans:return False,"LOCAL_MISSING_TOKEN_SPANS"
        if s<minimum_score:return False,"LOCAL_WEAK_CTC_SUPPORT"
        last=float(t)
    return True,"VALID_LOCAL_CTC"
