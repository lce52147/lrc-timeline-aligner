"""Bounded CTC boundary competition without reference/BK clocks.

Reruns two adjacent unanchored passage bundles when a first/last row has low
local CTC support. The only admissible output is a research proposal for
Central, never a trusted candidate or independent occurrence proof.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence

from segment_boundary_selection import CtcBoundaryRow, select_ctc_boundary_variant


def compete_ctc_adjacent_boundaries(
    rows: list[dict[str, object]],
    bundles: Sequence[object],
    passages: Sequence[object],
    lyric_texts: list[str],
    anchored: Sequence[tuple[int, int, int]],
    *,
    stem: Path,
    duration: float,
    python_exe: Path,
    helper: Path,
    device: str,
    temp_dir: Path,
    runner: Callable[..., object],
    max_proposals: int = 6,
    priority_rows: Sequence[int] = (),
) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
    """Returns entire corrected row vector, or None on ANY helper/data error."""
    lookup = {int(x["row"]): dict(x) for x in rows}
    if len(lookup) != len(rows) or set(lookup) != set(range(1,len(rows)+1)):
        return None, {"status":"BLOCKED_INVALID_OLD_ROW_COVERAGE"}
    anchor_rows = {i for first,last,_ in anchored for i in range(first,last+1)}
    priority_set = {int(i) for i in priority_rows}
    proposals = []
    used_bundles: set[int] = set()
    for i in range(len(bundles)-1):
        if i in used_bundles or i+1 in used_bundles:
            continue
        left,right=bundles[i],bundles[i+1]
        if left.end_row+1!=right.start_row:
            continue
        last=lookup[left.end_row]
        first=lookup[right.start_row]
        move=None
        if (left.end_row>left.start_row and left.end_row not in anchor_rows
            and (0<=float(last["ctc_score"])<.12 or left.end_row in priority_set)):
            move=(left.end_row,"L_TO_R")
        elif (right.end_row>right.start_row and right.start_row not in anchor_rows
              and (0<=float(first["ctc_score"])<.12 or right.start_row in priority_set)):
            move=(right.start_row,"R_TO_L")
        if move is None:
            continue
        if len(proposals)>=max_proposals:
            return None, {"status":"BLOCKED_TOO_MANY_WEAK_BOUNDARIES"}
        proposals.append((i,move[0],move[1]))
        used_bundles.update((i,i+1))
    decisions=[]
    for ordinal,(index,moved,direction) in enumerate(proposals):
        left,right=bundles[index:index+2]
        if direction=="L_TO_R":
            spec=((left.start_row,left.end_row-1,left.passage_index),
                  (left.end_row,right.end_row,right.passage_index))
        else:
            spec=((left.start_row,right.start_row,left.passage_index),
                  (right.start_row+1,right.end_row,right.passage_index))
        alternative=[]
        for half,(first,last,passage_i) in enumerate(spec):
            voice=passages[passage_i]
            transcript=temp_dir/f"s1-boundary-{ordinal:02}-{half}.lyrics.json"
            out=temp_dir/f"s1-boundary-{ordinal:02}-{half}.ctc.json"
            transcript.write_text(json.dumps(lyric_texts[first-1:last],ensure_ascii=False),encoding="utf-8")
            result=runner([
                str(python_exe),str(helper),"--audio",str(stem),
                "--transcript",str(transcript),"--output",str(out),
                "--device",device,
                "--start",str(round(max(0,voice.start-.30),3)),
                "--end",str(round(min(duration,voice.end+.30),3)),
            ],capture_output=True,text=True,errors="replace",timeout=180)
            if result.returncode!=0 or not out.is_file():
                return None, {"status":"BLOCKED_LOCAL_BOUNDARY_CTC_FAILED",
                              "boundary":index,"direction":direction}
            raw=json.loads(out.read_text(encoding="utf-8"))
            es=raw.get("entries")
            if not isinstance(es,list) or len(es)!=last-first+1:
                return None, {"status":"BLOCKED_BOUNDARY_ROW_COUNT","boundary":index}
            for row_no,item in enumerate(es,first):
                if not isinstance(item,dict):
                    return None, {"status":"BLOCKED_BOUNDARY_ROW_SHAPE","boundary":index}
                alt={
                    "row":row_no,"time":item.get("start"),
                    "ctc_score":item.get("ctc_score"),
                    "token_spans":item.get("token_spans"),
                    "passage":passage_i,
                }
                alternative.append(alt)
        originals=[lookup[row_no] for row_no in range(left.start_row,right.end_row+1)]
        try:
            old_score=[CtcBoundaryRow(int(x["row"]),float(x["time"]),float(x["ctc_score"])) for x in originals]
            new_score=[CtcBoundaryRow(int(x["row"]),float(x["time"]),float(x["ctc_score"])) for x in alternative]
        except (TypeError,ValueError,KeyError):
            return None,{"status":"BLOCKED_BOUNDARY_SCORE_SHAPE","boundary":index}
        decision=select_ctc_boundary_variant(old_score,new_score,moved_row=moved)
        decisions.append({
            "passage_boundary":index,
            "moved_row":moved,
            "status":decision.status,
            "mean_ctc_gain":decision.average_gain,
            "moved_ctc_gain":decision.moved_gain,
            "reason":decision.reason,
        })
        if decision.status=="RESEARCH_ACCEPTED_NOT_CENTRAL":
            # Full neighbor pair changes as an atomic unit.
            for candidate in alternative:
                if int(candidate["row"]) in anchor_rows:
                    # Even same-row retiming in anchored bundles must be
                    # re-certified together by occurrence-specific Central.
                    return None,{"status":"BLOCKED_ANCHORED_ROW_REWRITE","boundary":index}
                lookup[int(candidate["row"])]=candidate
    return [lookup[i] for i in range(1,len(rows)+1)], {
        "status":"BOUNDARY_RESEARCH_COMPLETE",
        "proposals":len(proposals),"decisions":decisions,
        "admitted_by_central":False,
    }
