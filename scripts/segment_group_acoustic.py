"""Independent, BK-free local-CTC comparison of each lyric group across passages."""
from __future__ import annotations
import json,math,tempfile
from pathlib import Path
from typing import Callable,Sequence

MIN_PLACEMENT_MARGIN=.02
# An unsupported group's rows take the whole-song CTC clock back on their own
# (segment_skeleton_production); the whole song rolls back only when no group
# is supported.  A share-of-rows cutoff let one misplaced tail section (sister
# c06: 11 of 52 rows) discard every correctly placed group with it.
# Same song, same model, different GPU: group scores differ by ~1e-5.  A
# group whose margin is within this band of the cutoff has no stable verdict;
# when flipping those groups alone flips the all-song rollback, the gate is
# reported as near-edge so the caller evaluates BOTH paths instead of letting
# GPU rounding pick one.
PLACEMENT_EDGE_TOLERANCE=.005

def placement_decision(assigned_score:float, competitor_scores:Sequence[float],
                       margin:float=MIN_PLACEMENT_MARGIN)->bool:
    if not competitor_scores:return True
    if not (math.isfinite(assigned_score) and all(
            math.isfinite(float(x)) for x in competitor_scores)):return False
    return assigned_score>=max(competitor_scores)+margin

def placement_margin(assigned_score:float, competitor_scores:Sequence[float],
                     margin:float=MIN_PLACEMENT_MARGIN)->float|None:
    """Signed distance from the placement cutoff (None: no competitor)."""
    if not competitor_scores:return None
    if not (math.isfinite(assigned_score) and all(
            math.isfinite(float(x)) for x in competitor_scores)):return None
    return assigned_score-(max(competitor_scores)+margin)

def placement_fallback_near_edge(groups:Sequence[dict],
                                 tolerance:float=PLACEMENT_EDGE_TOLERANCE)->bool:
    """Whether marginal groups alone decide the all-song rollback (no group
    supported)."""
    marginal=False
    for group in groups:
        value=group.get('decision_margin')
        if value is not None and abs(float(value))<=tolerance:
            marginal=True
        elif group.get('supported'):
            return False
    return marginal

def contiguous_passage_groups(passages:Sequence[int])->list[tuple[int,int,int]]:
    if not passages:return []
    groups=[]
    begin=0
    for i in range(1,len(passages)+1):
        if i==len(passages) or passages[i]!=passages[begin]:
            groups.append((begin,i,int(passages[begin])))
            begin=i
    return groups

def check_group_placement(*,rows:list[dict], passage_ids:Sequence[int],
                          passages:Sequence[object], lyric_texts:Sequence[str],
                          stem:Path,duration:float,python:Path,helper:Path,
                          device:str,runner:Callable[...,object],
                          timeout:float=180.) ->dict:
    if len(rows)!=len(passage_ids) or len(rows)!=len(lyric_texts):
        return {'status':'ERROR','reason':'GROUP_ROW_MISMATCH'}
    if any(x<0 or x>=len(passages) for x in passage_ids):
        return {'status':'ERROR','reason':'INVALID_PASSAGE_ID'}
    if any(a>b for a,b in zip(passage_ids,passage_ids[1:])):
        return {'status':'ERROR','reason':'NONMONOTONIC_PASSAGE_MAP'}
    groups=contiguous_passage_groups(passage_ids)
    evidence=[]
    invalid=[]
    with tempfile.TemporaryDirectory(prefix='lrc13-group-placement-') as name:
        folder=Path(name)
        for k,(begin,end,j) in enumerate(groups):
            group_rows=rows[begin:end]
            try:
                assigned=sum(float(q['ctc_score']) for q in group_rows)/len(group_rows)
            except (TypeError,KeyError,ValueError,ZeroDivisionError):
                return {'status':'ERROR','reason':'ASSIGNED_SCORE_MISSING','group':k}
            competitors=[]
            for neighbor in (j-1,j+1):
                if not 0<=neighbor<len(passages):continue
                p=passages[j]
                q=passages[neighbor]
                width=max(.5,float(p.end-p.start)+.60)
                center=(float(q.start)+float(q.end))*.5
                start=max(0.,center-width*.5)
                stop=min(float(duration),start+width)
                if stop-start<.38:
                    return {'status':'ERROR','reason':'INVALID_NEIGHBOR_WINDOW','group':k}
                lyric_path=folder/f'g{k:03d}_neighbor{neighbor:03d}.json'
                json_path=folder/f'g{k:03d}_neighbor{neighbor:03d}.aligned.json'
                lyric_path.write_text(json.dumps(list(lyric_texts[begin:end]),ensure_ascii=False),
                                      encoding='utf8')
                cmd=[str(python),str(helper),'--audio',str(stem),
                     '--transcript',str(lyric_path),'--output',str(json_path),
                     '--device',str(device),'--start',str(round(start,3)),
                     '--end',str(round(stop,3))]
                result=runner(cmd,timeout=timeout)
                if result.returncode!=0 or not json_path.is_file():
                    return {'status':'ERROR','reason':'NEIGHBOR_GPU_ERROR','group':k,
                            'neighbor':neighbor,
                            'detail':str(getattr(result,'stderr',''))[-200:]}
                data=json.loads(json_path.read_text(encoding='utf8'))
                alt=data.get('entries')
                if not isinstance(alt,list) or len(alt)!=end-begin:
                    return {'status':'ERROR','reason':'NEIGHBOR_ROW_COUNT','group':k,
                            'neighbor':neighbor}
                try:
                    scores=[float(x['ctc_score']) for x in alt]
                except (TypeError,KeyError,ValueError):
                    return {'status':'ERROR','reason':'NEIGHBOR_SCORE_MISSING',
                            'group':k,'neighbor':neighbor}
                if not all(math.isfinite(s) and 0<=s<=1 for s in scores):
                    return {'status':'ERROR','reason':'NEIGHBOR_SCORE_INVALID',
                            'group':k,'neighbor':neighbor}
                competitors.append({'passage':neighbor,
                                    'ctc_score':round(sum(scores)/len(scores),6),
                                    'window':[round(start,3),round(stop,3)]})
            passes=placement_decision(assigned,[c['ctc_score'] for c in competitors])
            signed=placement_margin(assigned,[c['ctc_score'] for c in competitors])
            diagnostic={'group':k,'rows':[begin+1,end],'passage':j,
                        'assigned_ctc_score':round(assigned,6),
                        'neighbors':competitors,'supported':passes,
                        'decision_margin':None if signed is None else round(signed,6)}
            evidence.append(diagnostic)
            if not passes:
                invalid.extend(range(begin+1,end+1))
    fraction=len(invalid)/max(1,len(rows))
    return {'status':'OK','groups':evidence,'unsupported_rows':invalid,
            'unsupported_fraction':round(fraction,6),
            'fallback':bool(rows) and len(invalid)==len(rows),
            'near_edge':placement_fallback_near_edge(evidence),
            'edge_tolerance':PLACEMENT_EDGE_TOLERANCE}
