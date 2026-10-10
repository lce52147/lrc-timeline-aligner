"""Option A: independent second DP candidate + acoustic whole-song reconciliation.

No BK, labels or timestamps enter this producer. Both candidate paths are
real MMS CTC responses on the same isolated vocal stem. The reconciled lyric
to passage allocation is re-aligned across the ENTIRE song before current is
replaced; never splice old per-line clocks into the output.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable,Sequence
from types import SimpleNamespace
from segment_structure import assign_lyrics_to_vocal_passages
from segment_skeleton_consensus import TestedRow,choose_acoustic_joint_passages
from segment_boundary_search import compete_ctc_adjacent_boundaries


def _align_groups(
    groups:Sequence[tuple[int,int,int]],
    passages:Sequence[object],
    texts:Sequence[str],
    *,stem:Path,duration:float,python:Path,helper:Path,device:str,
    folder:Path,runner:Callable[...,object],
    margin:float=.30,
)->list[dict[str,object]]|None:
    results=[]
    for ordinal,(first,last,passage_idx) in enumerate(groups):
        vocal=passages[passage_idx]
        source=folder/f"dual_{ordinal:03d}_lyrics.json"
        dest=folder/f"dual_{ordinal:03d}_aligned.json"
        source.write_text(json.dumps(list(texts[first:last]),ensure_ascii=False),encoding="utf8")
        response=runner([str(python),str(helper),"--audio",str(stem),
           "--transcript",str(source),"--output",str(dest),
           "--device",str(device),
           "--start",str(round(max(0.,vocal.start-margin),3)),
           "--end",str(round(min(duration,vocal.end+margin),3))],timeout=180)
        if response.returncode!=0 or not dest.is_file():return None
        data=json.loads(dest.read_text(encoding="utf8"))
        entries=data.get("entries")
        if not isinstance(entries,list) or len(entries)!=last-first or any(
            not isinstance(x,dict) for x in entries):return None
        for x in entries:
            results.append({**x,"passage_index":passage_idx})
    return results


def _edges(
    rows:list[dict[str,object]],groups:Sequence[tuple[int,int,int]],
    passages:Sequence[object],texts:Sequence[str], *,
    stem:Path,duration:float,python:Path,helper:Path,device:str,
    folder:Path,runner:Callable[...,object],
)->tuple[list[dict[str,object]]|None,dict[str,object]]:
    first=[];passages_by_row=[]
    for start,end,idx in groups:
        passages_by_row.extend([idx]*(end-start))
    for i,(row,index) in enumerate(zip(rows,passages_by_row)):
        first.append({"row":i+1,"time":row.get("start"),
          "ctc_score":row.get("ctc_score"),
          "token_spans":row.get("token_spans"),"passage":index})
    grouped=[SimpleNamespace(start_row=start+1,end_row=end,passage_index=passage)
             for start,end,passage in groups]
    out,diag=compete_ctc_adjacent_boundaries(
      first,grouped,passages,list(texts),(),
      stem=stem,duration=duration,python_exe=python,helper=helper,
      device=device,temp_dir=folder,max_proposals=10,
      runner=lambda command,**kw:runner(command,timeout=kw.get("timeout",180)))
    if out is None:return None,diag
    fixed=[]
    for old,change in zip(rows,out):
        entry={**old,"start":change["time"],
               "ctc_score":change["ctc_score"],
               "token_spans":change["token_spans"],
               "passage_index":int(change["passage"])}
        if old.get("start")!=change["time"] or old.get("passage_index")!=change["passage"]:
            entry["first_token_candidates"]=None
        fixed.append(entry)
    return fixed,diag


def reconcile_dual_dp(
    *,
    main_rows:list[dict[str,object]],
    main_passage_ids:Sequence[int],
    main_passages:Sequence[object],
    vocal_segments:Sequence[tuple[float,float]],
    texts:Sequence[str],
    priors:Sequence[float],
    duration:float,
    stem:Path,python:Path,helper:Path,device:str,
    folder:Path,runner:Callable[...,object],
)->tuple[list[dict[str,object]]|None,dict[str,object]]:
    n=len(texts)
    blocked=lambda reason,**kw:(None,{"status":"BLOCKED_DUAL_DP","reason":reason,**kw})
    if len(main_rows)!=n or len(main_passage_ids)!=n:
        return blocked("ROW_COUNT")
    alt=assign_lyrics_to_vocal_passages(vocal_segments,
       duration=duration,lyric_lengths=[max(1,len(text)) for text in texts],
       ctc_priors=priors,max_lines_per_passage=12)
    if alt.status!="UNVERIFIED_OCCURRENCE":
        return blocked("ALTERNATE_DP_BLOCKED",DP_status=alt.status)
    if len(alt.passages)!=len(main_passages) or any(
        abs(a.start-b.start)>.001 or abs(a.end-b.end)>.001
        for a,b in zip(alt.passages,main_passages)):
        return blocked("PASSAGE_DETECTION_INCONSISTENT")
    alt_groups=[(b.start_row-1,b.end_row,b.passage_index) for b in alt.bundles]
    primary_groups=[]
    start=0
    for i in range(1,n+1):
        if i==n or main_passage_ids[i]!=main_passage_ids[start]:
            primary_groups.append((start,i,main_passage_ids[start]))
            start=i
    if tuple(alt_groups)==tuple(primary_groups):
        return main_rows,{"status":"SAME_ALLOCATION","switches":0}
    alternate=_align_groups(alt_groups,main_passages,texts,stem=stem,
       duration=duration,python=python,helper=helper,device=device,
       folder=folder,runner=runner)
    if alternate is None:return blocked("ALTERNATE_GPU_FAILED")
    alternate,edge=_edges(alternate,alt_groups,main_passages,texts,stem=stem,
       duration=duration,python=python,helper=helper,device=device,
       folder=folder,runner=runner)
    if alternate is None:return blocked("ALTERNATE_EDGE_FAILED",edge=edge)
    def test(rows:Sequence[dict[str,object]],indices:Sequence[int],tag:str):
        try:
            return tuple(TestedRow(int(p),float(x["start"]),
               float(x["ctc_score"]),tag) for p,x in zip(indices,rows))
        except (ValueError,TypeError,KeyError):
            return ()
    primary=test(main_rows,[r["passage_index"] for r in main_rows],"DP_ROMAJI")
    second=test(alternate,[r["passage_index"] for r in alternate],"DP_CHAR")
    if len(primary)!=n or len(second)!=n:
        return blocked("ACOUSTIC_DATA_INCOMPLETE")
    joint=choose_acoustic_joint_passages([primary,second])
    if joint.status!="RESEARCH_JOINT_PATH_NO_AUTHORITY":
        return blocked(joint.status)
    chosen=[r.passage for r in joint.rows]
    if tuple(chosen)==tuple(main_passage_ids):
        return main_rows,{"status":"PRIMARY_DP_SELECTED","switches":joint.switches,
                          "joint_acoustic_cost":joint.total_cost}
    all_groups=[];start=0
    for i in range(1,n+1):
        if i==n or chosen[i]!=chosen[start]:
            all_groups.append((start,i,chosen[start]))
            start=i
    if len(all_groups)>64:
        return blocked("TOO_MANY_COMBINED_GROUPS")
    # No partial path adoption. Rerun a complete MMS alignment on the
    # newly chosen full-song allocation.
    reconciled=_align_groups(all_groups,main_passages,texts,stem=stem,
      duration=duration,python=python,helper=helper,device=device,
      folder=folder,runner=runner)
    if reconciled is None:return blocked("JOINT_GPU_FAILED")
    if len(reconciled)!=n:return blocked("JOINT_ROW_COUNT")
    times=[r.get("start") for r in reconciled]
    if any(not isinstance(t,(int,float)) or not math.isfinite(t) for t in times):
        return blocked("JOINT_INVALID_TIME")
    if any(b<=a+.019 for a,b in zip(times,times[1:])):
        return blocked("JOINT_NONMONOTONIC")
    if any(not (main_passages[i].start-.35<=t<=main_passages[i].end+.35)
           for t,i in zip(times,chosen)):
        return blocked("JOINT_OUTSIDE_PASSAGE")
    return reconciled,{"status":"JOINT_GPU_RECOMPUTED","switches":joint.switches,
      "joint_acoustic_cost":joint.total_cost,
      "alternate_edge":edge,
      "alternative_groups":len(alt_groups),"result_groups":len(all_groups),
      "chosen_source_count":{
          name:joint.sources.count(name) for name in set(joint.sources)
      }}
