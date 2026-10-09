"""Strict S5 per-line regression gate; BK enters only as evaluation truth.

Pure post-hoc analysis of frozen output rows. No audio inference, no
reference-timestamp lookup for candidate generation, and no source mutation.
"""
from __future__ import annotations
from collections import Counter
from typing import Mapping,Sequence
import math
import unicodedata


def _normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC",text).casefold().split())


def evaluate_s5_rows(
    rows: Sequence[Mapping[str,object]],
    *,
    song_id: str,
    expected_rows: int | None = None,
) -> dict[str,object]:
    """Reject unmatched, malformed, partially scored or nonmonotonic rows.

    Every input row is an *already aligned* immutable lyric with:
    text, bk_s, main_s and candidate_s, optionally main_trusted/candidate_trusted.
    The gate does not match by proximity or select any new times.
    """
    def reject(status: str) -> dict[str,object]:
        return {"song":song_id,"status":status,"eligible_for_S5":False}
    if not song_id or not rows or expected_rows is not None and len(rows)!=expected_rows:
        return reject("BLOCKED_MISSING_OR_PARTIAL_ROWS")
    text_values=[]
    for row in rows:
        if not isinstance(row,Mapping):
            return reject("BLOCKED_ROW_TYPE")
        text=row.get("text")
        if not isinstance(text,str) or not _normalize(text):
            return reject("BLOCKED_TEXT_IDENTITY")
        for name in ("bk_s","main_s","candidate_s"):
            value=row.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                return reject("BLOCKED_INVALID_CLOCK")
        text_values.append(_normalize(text))
    # A broken main baseline must remain scoreable: detecting its collapse is
    # why the regression exists. Require the NEW candidate and human reference
    # to be ordered; comparison to main must not silently reject bad main.
    # Distinct lines may share a centisecond, so allow equality.
    for name in ("bk_s","candidate_s"):
        vals=[float(row[name]) for row in rows]
        if any(a>b+1e-8 for a,b in zip(vals,vals[1:])):
            return reject("BLOCKED_NONMONOTONIC_"+name.upper())
    counts=Counter(text_values)
    scores=[]
    trust_before=trust_after=trust_before_wrong=trust_after_wrong=0
    trust_known=True
    for i,(row,text) in enumerate(zip(rows,text_values)):
        old=abs(float(row["main_s"])-float(row["bk_s"]))
        new=abs(float(row["candidate_s"])-float(row["bk_s"]))
        rep=counts[text]>1
        o=row.get("main_trusted")
        n=row.get("candidate_trusted")
        if type(o) is not bool or type(n) is not bool:trust_known=False
        else:
            trust_before+=o
            trust_after+=n
            trust_before_wrong+=bool(o and old>1.)
            trust_after_wrong+=bool(n and new>1.)
        scores.append({"row":i+1,"text":row["text"],"repeated":rep,
            "main_error_s":round(old,3),"candidate_error_s":round(new,3),
            "new_gt1s":old<=1. and new>1.,
            "main_wrong_gt1s":old>1.,
            "candidate_wrong_gt1s":new>1.,
            "main_pco200":old<=.2+1e-9,
            "candidate_pco200":new<=.2+1e-9})
    stats={
        "rows":len(scores),
        "main_pco200":sum(v["main_pco200"] for v in scores),
        "candidate_pco200":sum(v["candidate_pco200"] for v in scores),
        "main_wrong_gt1s":sum(v["main_wrong_gt1s"] for v in scores),
        "candidate_wrong_gt1s":sum(v["candidate_wrong_gt1s"] for v in scores),
        "new_wrong_gt1s":sum(v["new_gt1s"] for v in scores),
        "repeated_rows":sum(v["repeated"] for v in scores),
        "repeated_old_wrong":sum(v["repeated"] and v["main_wrong_gt1s"] for v in scores),
        "repeated_new_wrong":sum(v["repeated"] and v["candidate_wrong_gt1s"] for v in scores),
        "unique_old_wrong":sum(not v["repeated"] and v["main_wrong_gt1s"] for v in scores),
        "unique_new_wrong":sum(not v["repeated"] and v["candidate_wrong_gt1s"] for v in scores),
    }
    pass_pco=stats["candidate_pco200"]>=stats["main_pco200"]
    pass_gt1=stats["candidate_wrong_gt1s"]<stats["main_wrong_gt1s"]
    pass_new=stats["new_wrong_gt1s"]==0
    trusted={
        "available":trust_known,
        "main_trusted":trust_before if trust_known else None,
        "main_wrong_trusted_count":trust_before_wrong if trust_known else None,
        "candidate_wrong_trusted_count":trust_after_wrong if trust_known else None,
        "candidate_trusted":trust_after if trust_known else None,
        "main_wrong_trusted_ratio":(trust_before_wrong/trust_before
             if trust_known and trust_before else None),
        "candidate_wrong_trusted_ratio":(trust_after_wrong/trust_after
             if trust_known and trust_after else None),
    }
    trust_pass=(trust_known and trust_before and trust_after and
        trusted["candidate_wrong_trusted_ratio"]<trusted["main_wrong_trusted_ratio"])
    return {
        "song":song_id,
        "status":"POSTHOC_GATE_COMPUTED",
        "eligible_for_S5":True,
        "counts":stats,
        "trusted":trusted,
        "checks":{"PCO_non_decreasing":pass_pco,"gt1_decreasing":pass_gt1,
                  "no_new_gt1":pass_new,"trusted_wrong_ratio_decreasing":bool(trust_pass)},
        "per_song_checks_are_diagnostic_only":True,
        "review_rows":[x for x in scores if x["main_wrong_gt1s"] or x["candidate_wrong_gt1s"]],
    }

def evaluate_s5_corpus(
    song_results: Sequence[Mapping[str,object]],
    *,
    expected_songs: int=39,
) -> dict[str,object]:
    """Official thresholds are *corpus-wide*, never per clean control song."""
    bad={"status":"BLOCKED_INCOMPLETE_S5_CORPUS","release_pass":False}
    if len(song_results)!=expected_songs or len({x.get("song") for x in song_results})!=len(song_results):
        return bad
    if any(r.get("status")!="POSTHOC_GATE_COMPUTED" or not r.get("eligible_for_S5")
           for r in song_results):
        return bad
    keys=("rows","main_pco200","candidate_pco200","main_wrong_gt1s",
          "candidate_wrong_gt1s","new_wrong_gt1s","repeated_rows",
          "repeated_old_wrong","repeated_new_wrong","unique_old_wrong",
          "unique_new_wrong")
    stats={k:sum(int(r["counts"][k]) for r in song_results) for k in keys}
    all_trust=all(isinstance(r.get("trusted"),Mapping) and r["trusted"].get("available")
                  and isinstance(r["trusted"].get("main_trusted"),int)
                  and isinstance(r["trusted"].get("candidate_trusted"),int)
                  for r in song_results)
    old_t=new_t=old_w=new_w=0
    if all_trust:
        for r in song_results:
            q=r["trusted"]
            old_t+=q["main_trusted"];new_t+=q["candidate_trusted"]
            # Per-song ratio cannot reliably recover integer counts; preserve
            # them explicitly in S5 source evidence for a pooled ratio.
            for key in ("main_wrong_trusted_count","candidate_wrong_trusted_count"):
                if not isinstance(q.get(key),int):
                    all_trust=False
        if all_trust:
            old_w=sum(r["trusted"]["main_wrong_trusted_count"] for r in song_results)
            new_w=sum(r["trusted"]["candidate_wrong_trusted_count"] for r in song_results)
    trust_rate_old=old_w/old_t if all_trust and old_t else None
    trust_rate_new=new_w/new_t if all_trust and new_t else None
    checks={
        "PCO_non_decreasing":stats["candidate_pco200"]>=stats["main_pco200"],
        "gt1_strictly_decreasing":stats["candidate_wrong_gt1s"]<stats["main_wrong_gt1s"],
        "no_new_gt1s":stats["new_wrong_gt1s"]==0,
        "trusted_wrong_ratio_decreasing":(
            trust_rate_old is not None and trust_rate_new is not None
            and trust_rate_new<trust_rate_old),
    }
    return {"status":"S5_RELEASE_GATE_EVALUATED","songs":len(song_results),
            "counts":stats,"trusted_ratio_before":trust_rate_old,
            "trusted_ratio_after":trust_rate_new,
            "checks":checks,"release_pass":all(checks.values()),
            "new_wrong_song_ids":[r["song"] for r in song_results
              if r["counts"]["new_wrong_gt1s"]>0]}
