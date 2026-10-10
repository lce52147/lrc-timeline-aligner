"""Option A current skeleton producer: all-song transactional local MMS CTC.

It never writes the LRC or reads checked/BK timestamps.  Failure returns the
ENTIRE unchanged global CTC current report.  No mixture of row clocks.
"""
from __future__ import annotations
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Callable,Sequence
from dataclasses import replace
from types import SimpleNamespace

from ctc_confidence import ctc_alignment_confidence
from segment_skeleton import plan_segment_skeleton,validate_local_ctc

MAX_GPU_PASSAGES=64
BOUNDARY_MARGIN_S=.30
LOCAL_CTC_TIMEOUT_S=180
MIN_GLOBAL_CTC_SCORE_GAIN=.08
HIGH_DISPLACEMENT_FRACTION=.35
HIGH_DISPLACEMENT_MIN_GAIN=.28
DISPLACEMENT_SECONDS=3.
_RUNTIME_CACHE:dict[tuple[object,...],tuple[list[dict[str,object]],dict[str,object]]]={}


def run_segment_skeleton(
    *,
    report:dict[str,object],
    timestamps:Sequence[float],
    lyric_texts:Sequence[str],
    vocal_stem:Path,
    duration:float,
    audio_api:object,
    python_exe:Path,
    ctc_helper:Path,
    device:str,
    helper_runner:Callable[...,object],
    romanizer:Callable[[str],str|None]|None=None,
    raw_asr_events:Sequence[tuple[float,float,str]]=(),
)->tuple[list[float],dict[str,object],dict[str,object]]:
    original=list(timestamps)
    def fallback(reason:str,**kw:object):
        details={'status':'FALLBACK_CTC','reason':reason,'whole_song':True,
          'current_source':'whole-song-ctc','BK_used':False,**kw}
        report['segment_skeleton_status']=details
        return original,report,details
    if report.get('ctc_audio_source')!='vocal-stem' or not vocal_stem.is_file():
        return fallback('NO_ISOLATED_VOCAL_STEM')
    if len(original)!=len(lyric_texts) or not lyric_texts:
        return fallback('MISSING_OR_MISMATCHED_LYRICS')
    if not (python_exe.is_file() and ctc_helper.is_file()):
        return fallback('HELPER_MISSING')
    try:
        features=audio_api.analyze_audio(vocal_stem,duration)
        plan=plan_segment_skeleton(
            features.segments,duration=duration,lyric_texts=lyric_texts,
            romanizer=romanizer,ctc_priors=original,
            raw_asr_events=raw_asr_events)
        if plan.status!='READY':
            return fallback(plan.reason,DP_diagnostics=plan.diagnostics)
        if len(plan.groups)>MAX_GPU_PASSAGES:
            return fallback('TOO_MANY_GPU_PASSAGES',groups=len(plan.groups))
        # Each lyric group is permanently tied to ONE acoustic passage. The
        # actual CTC model aligns only those lyrics to that source audio.
        rows: list[dict[str,object]]=[]
        row_to_passage=[a.passage_index for a in plan.assignments]
        boundary_diag={"status":"NOT_CHECKED"}
        dual_diag={"status":"NOT_CHECKED"}
        cache_key=(
            str(vocal_stem.resolve()),vocal_stem.stat().st_size,
            vocal_stem.stat().st_mtime_ns,str(ctc_helper.resolve()),
            ctc_helper.stat().st_size,ctc_helper.stat().st_mtime_ns,
            tuple(lyric_texts),tuple(round(x,3) for x in original),
            str(device),tuple(plan.groups),tuple(
                (round(p.start,3),round(p.end,3)) for p in plan.passages)
        )
        cached=_RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            rows=copy.deepcopy(cached[0])
            boundary_diag=copy.deepcopy(cached[1].get("boundary_competition",{}))
            row_to_passage=list(cached[1].get("row_to_passage",row_to_passage))
            dual_diag=copy.deepcopy(cached[1].get("two_dp_acoustic",{}))
        else:
            with tempfile.TemporaryDirectory(prefix='lrc13-option-a-') as temporary:
                folder=Path(temporary)
                for group_id,(start,end,passage_index) in enumerate(plan.groups):
                    passage=plan.passages[passage_index]
                    lyric_file=folder/f'group_{group_id:03d}.json'
                    result_file=folder/f'group_{group_id:03d}.aligned.json'
                    lyric_file.write_text(json.dumps(
                       list(lyric_texts[start:end]),ensure_ascii=False),encoding='utf8')
                    command=[str(python_exe),str(ctc_helper),
                      '--audio',str(vocal_stem),'--transcript',str(lyric_file),
                      '--output',str(result_file),'--device',str(device),
                      '--start',str(round(max(0.,passage.start-BOUNDARY_MARGIN_S),3)),
                      '--end',str(round(min(duration,passage.end+BOUNDARY_MARGIN_S),3))]
                    result=helper_runner(command,timeout=LOCAL_CTC_TIMEOUT_S)
                    if result.returncode!=0 or not result_file.is_file():
                        return fallback('LOCAL_GPU_CTC_FAILED',group=group_id,
                          detail=str(getattr(result,'stderr',''))[-300:])
                    data=json.loads(result_file.read_text(encoding='utf8'))
                    chunk=data.get('entries')
                    if not isinstance(chunk,list) or len(chunk)!=end-start:
                        return fallback('LOCAL_GPU_ROWS_WRONG',group=group_id)
                    if any(not isinstance(item,dict) for item in chunk):
                        return fallback('LOCAL_GPU_INVALID_ROW',group=group_id)
                    rows.extend(chunk)
                # A2/A3: global DP assigns passages first. Independent
                # *acoustic* local CTC then tests low-confidence neighboring
                # group edges WITHOUT unique repeat-block anchors. Both
                # adjacent groups are re-aligned atomically; no lyric gets a
                # timestamp from a different song or the human BK.
                from segment_boundary_search import compete_ctc_adjacent_boundaries
                initial=[{
                    "row":i+1,"time":r.get("start"),
                    "ctc_score":r.get("ctc_score"),
                    "token_spans":r.get("token_spans"),
                    "passage":row_to_passage[i],
                } for i,r in enumerate(rows)]
                bundles=[SimpleNamespace(
                    start_row=start+1,end_row=end,passage_index=passage)
                    for start,end,passage in plan.groups]
                refined,boundary_diag=compete_ctc_adjacent_boundaries(
                    initial,bundles,plan.passages,list(lyric_texts),(),
                    stem=vocal_stem,duration=duration,python_exe=python_exe,
                    helper=ctc_helper,device=device,temp_dir=folder,
                    runner=lambda command,**kw:helper_runner(
                        command,timeout=float(kw.get("timeout",LOCAL_CTC_TIMEOUT_S))),
                    max_proposals=10,
                    priority_rows=plan.diagnostics.get('adjacent_ambiguous_rows',()),
                )
                if refined is None:
                    return fallback("BOUNDARY_COMPETITION_FAILED",
                        boundary_details=boundary_diag)
                for i,item in enumerate(refined):
                    old=initial[i]
                    if (item["time"]!=old["time"] or
                        item["ctc_score"]!=old["ctc_score"] or
                        item["passage"]!=old["passage"]):
                        rows[i]["start"]=item["time"]
                        rows[i]["ctc_score"]=item["ctc_score"]
                        rows[i]["token_spans"]=item["token_spans"]
                        # Old helper's first-token alternatives belong to
                        # the abandoned group clock; do not keep them.
                        rows[i]["first_token_candidates"]=None
                    row_to_passage[i]=int(item["passage"])
                for row,index in zip(rows,row_to_passage):
                    row["passage_index"]=index
                # A second generic unanchored DP resolves non-unique lyrical
                # repetitions by comparing genuine acoustic paths. A joint
                # path, when selected, is re-aligned as ONE song. Missing or
                # inconclusive evidence rejects the entire skeleton.
                from segment_skeleton_dual import reconcile_dual_dp
                final_rows,dual_diag=reconcile_dual_dp(
                    main_rows=rows,main_passage_ids=row_to_passage,
                    main_passages=plan.passages,
                    vocal_segments=features.segments,
                    texts=lyric_texts,priors=original,duration=duration,
                    stem=vocal_stem,python=python_exe,helper=ctc_helper,
                    device=device,folder=folder,runner=helper_runner)
                if final_rows is None:
                    return fallback("SECOND_DP_RECONCILIATION_FAILED",
                                    joint_details=dual_diag)
                rows=final_rows
                row_to_passage=[int(x["passage_index"]) for x in rows]
        plan=replace(plan,assignments=tuple(
            replace(a,passage_index=row_to_passage[i])
            for i,a in enumerate(plan.assignments)))
        valid,why=validate_local_ctc(plan,rows,duration=duration)
        if not valid:
            return fallback(why,
                score_min=min(float(r.get('ctc_score') or 0.) for r in rows),
                weak_rows=[i+1 for i,r in enumerate(rows)
                           if float(r.get('ctc_score') or 0.)<.005],
                failed_proposal_rows=len(rows),
            )
        # No BK clocks: require an independent raw-ASR paragraph text
        # improvement for material structural relocation. Raw ASR also feeds
        # P1 DP, so the evidence is corroborating, not independent proof.
        from segment_asr_corroboration import check_asr_corroboration
        output_times=[float(row['start']) for row in rows]
        asr_corroboration=check_asr_corroboration(
            lyric_texts=lyric_texts,global_times=original,
            skeleton_times=output_times,raw_asr_events=raw_asr_events,
            romanizer=romanizer)
        if asr_corroboration.get('fallback'):
            return fallback('ASR_UNCORROBORATED_STRUCTURE',
                            raw_asr_corroboration=asr_corroboration)
        # A local path in the selected passage alone cannot establish
        # occurrence identity. Compare the SAME group text with adjacent
        # passage windows before allowing any structural timing promotion.
        from segment_group_acoustic import check_group_placement
        placement_diag=check_group_placement(
            rows=rows,passage_ids=row_to_passage,passages=plan.passages,
            lyric_texts=lyric_texts,stem=vocal_stem,duration=duration,
            python=python_exe,helper=ctc_helper,device=device,
            runner=helper_runner,timeout=LOCAL_CTC_TIMEOUT_S)
        if placement_diag.get('status')!='OK':
            return fallback('GROUP_PLACEMENT_EVIDENCE_FAILED',
                            group_placement=placement_diag)
        if placement_diag.get('fallback'):
            return fallback('GROUP_PLACEMENT_UNSUPPORTED',
                            group_placement=placement_diag)
        # Same MMS CTC model on the same isolated stem: a wholesale
        # structural move must improve genuine acoustic fit over the global
        # weakly aligned path. High-volume changes require stronger evidence.
        # No lyric reference/BK time, reviewer offset or per-song threshold.
        original_assignments=report.get("assignments")
        if not isinstance(original_assignments,list) or len(original_assignments)!=len(rows):
            return fallback("MISSING_GLOBAL_CTC_EVIDENCE")
        try:
            old_scores=[float(x["ctc_score"]) for x in original_assignments]
            new_scores=[float(x["ctc_score"]) for x in rows]
        except (TypeError,ValueError,KeyError):
            return fallback("MISSING_ACOUSTIC_SCORE_PAIR")
        if not all(math.isfinite(x) and 0<=x<=1 for x in old_scores+new_scores):
            return fallback("INVALID_ACOUSTIC_SCORE_PAIR")
        gain=sum(y-x for x,y in zip(old_scores,new_scores))/len(rows)
        displaced=sum(abs(float(y["start"])-x)>DISPLACEMENT_SECONDS
                      for y,x in zip(rows,original))/len(rows)
        # P3: global-path and short-window CTC path probabilities are not
        # comparable calibration. The old gain/displacement rule is now
        # read-only diagnostic; group-vs-neighbor CTC is the placement gate.
        legacy_gain_diagnostics={
            'mean_ctc_gain':round(gain,5),
            'displaced_fraction':round(displaced,5),
            'would_block_under_legacy_gain':bool(
                gain<MIN_GLOBAL_CTC_SCORE_GAIN or (
                    displaced>HIGH_DISPLACEMENT_FRACTION and
                    gain<HIGH_DISPLACEMENT_MIN_GAIN)),
            'legacy_gain_threshold':MIN_GLOBAL_CTC_SCORE_GAIN,
            'legacy_high_displacement_gain_threshold':HIGH_DISPLACEMENT_MIN_GAIN,
        }
        if cached is None:
            _RUNTIME_CACHE.clear()  # at most one all-song GPU result
            _RUNTIME_CACHE[cache_key]=(copy.deepcopy(rows),{
                "boundary_competition":copy.deepcopy(boundary_diag),
                "two_dp_acoustic":copy.deepcopy(dual_diag),
                "row_to_passage":list(row_to_passage),
            })
        cloned=copy.deepcopy(report)
        assignments=cloned.get('assignments')
        if not isinstance(assignments,list) or len(assignments)!=len(rows):
            return fallback('ASSIGNMENT_COUNT_CHANGED')
        output_times=[float(row['start']) for row in rows]
        # P5's documented F5 condition is a structural warning. While
        # remediating P1, reject the entire new skeleton if that warning
        # contradicts an original repeat occurrence by over 3 seconds.
        # This is an acoustic/global-CTC evidence guard, not a BK lookup.
        from collections import Counter
        normalized=[audio_api.normalize_match_text(t) for t in lyric_texts]
        repetitions=Counter(normalized)
        repeat_conflicts=[i+1 for i,(key,prev,now)
            in enumerate(zip(normalized,original,output_times))
            if key and repetitions[key]>=2 and abs(now-prev)>3.0]
        if repeat_conflicts and asr_corroboration.get('reason')!='CORROBORATED':
            return fallback('REPEATED_SKELETON_GLOBAL_CTC_CONFLICT',
                            entries=repeat_conflicts,
                            comparison_threshold_seconds=3.0)
        for i,(old,row,actual) in enumerate(zip(assignments,rows,output_times)):
            if not isinstance(old,dict):return fallback('ASSIGNMENT_INVALID')
            # Preserve whole-song CTC as independent evidence before replacing
            # the active, passage-local clocks and acoustic confidence.
            old['global_ctc_first_token_start']=copy.deepcopy(old.get('ctc_first_token_start'))
            old['global_ctc_score']=copy.deepcopy(old.get('ctc_score'))
            old['global_ctc_token_spans']=copy.deepcopy(old.get('ctc_token_spans'))
            old.update({
              'timestamp':round(actual,3),
              'ctc_first_token_start':round(actual,3),
              'ctc_token_spans':row.get('token_spans'),
              'ctc_first_token_candidates':row.get('first_token_candidates'),
              'ctc_score':round(float(row['ctc_score']),6),
              'romaji':row.get('romaji'),
              'segment':plan.assignments[i].passage_index,
              'score':ctc_alignment_confidence(float(row['ctc_score'])), 
              'timing_repair':'segment-skeleton-local-ctc',
              'timing_repair_source':'torchaudio-mms-fa',
              'segment_dp_confidence':plan.assignments[i].confidence,
            })
        # Preserve the original full-song diagnostics, without allowing
        # abandoned global proposals to remain authoritative active fields.
        global_keys=(
            'ctc_acoustic_backtracks','ctc_boundary_validation_stages',
            'ctc_local_window_realign','ctc_low_score_runs',
            'ctc_confidence_collapse',
        )
        cloned['global_ctc_diagnostics']={
            key:copy.deepcopy(cloned.pop(key,None)) for key in global_keys
        }
        cloned['strategy']='segment-skeleton-mms-ctc'
        cloned['ctc_low_score_entries']=[
            {'entry':i+1,'text':lyric_texts[i],'ctc_score':a['ctc_score']}
            for i,a in enumerate(assignments) if a['ctc_score']<.03]
        cloned['ctc_very_low_score_entries']=[
            x for x in cloned['ctc_low_score_entries'] if x['ctc_score']<.01]
        cloned['ctc_low_score_count']=len(cloned['ctc_low_score_entries'])
        cloned['ctc_very_low_score_count']=len(cloned['ctc_very_low_score_entries'])
        cloned['ctc_score_mean']=round(sum(a['ctc_score'] for a in assignments)/len(assignments),6)
        cloned['ctc_score_min']=round(min(a['ctc_score'] for a in assignments),6)
        cloned['ctc_missing_entries']=[]
        cloned['ctc_missing_count']=0
        cloned['ctc_low_score_runs']=[]
        cloned['ctc_confidence_collapse']=False
        # Low-confidence review is recalculated from the new acoustic sequence.
        cloned['suspicious_alignments']=[
            {'entry':i+1,'text':lyric_texts[i],
             'flags':['ctc_low_path_confidence'],'severity':'medium','review_required':True}
            for i,a in enumerate(assignments) if a['ctc_score']<.03]
        # Unsupported groups still below the all-song rollback threshold
        # must remain review-required and untrusted without changing clocks.
        for entry in placement_diag.get('unsupported_rows',[]):
            index=int(entry)-1
            assignment=assignments[index]
            assignment['group_placement_unsupported']=True
            flags=assignment.get('flags')
            flags=list(flags) if isinstance(flags,list) else []
            if 'group_placement_unsupported' not in flags:
                flags.append('group_placement_unsupported')
            assignment['flags']=flags
            cloned['suspicious_alignments'].append({
                'entry':index+1,'text':lyric_texts[index],
                'flags':['group_placement_unsupported'],
                'severity':'high','review_required':True})
        # P2: marginal confidence measures another best complete path,
        # not global second-path margin. Structural ambiguity left under
        # the 10% atomic-rollback threshold must still be explicitly reviewed.
        # P6: .0002 is the hard CTC floor; <.005 is additionally actionable.
        for index,assignment in enumerate(assignments):
            marks=[]
            if index+1 in plan.diagnostics.get('nonadjacent_ambiguous_rows',()):
                marks.append('segment_structural_ambiguity')
            if float(assignment.get('ctc_score',0.0))<.005:
                marks.append('segment_weak_ctc_support')
            if marks:
                flags=assignment.get('flags')
                flags=list(flags) if isinstance(flags,list) else []
                for mark in marks:
                    if mark not in flags:flags.append(mark)
                assignment['flags']=flags
                assignment['segment_weak_ctc_untrusted']=True
                cloned['suspicious_alignments'].append({
                    'entry':index+1,'text':lyric_texts[index],
                    'flags':marks,'severity':'high','review_required':True})
        cloned['suspicious_alignment_count']=len(cloned['suspicious_alignments'])
        cloned['review_required_count']=len(cloned['suspicious_alignments'])
        cloned['review_required']=bool(cloned['suspicious_alignments'])
        status={
          'status':'SEGMENT_CURRENT','reason':'VALID_LOCAL_CTC',
          'whole_song':True,'BK_used':False,
          'current_source':'segment-vad-dp-local-ctc',
          'groups':len(plan.groups),'passages':len(plan.passages),
          'rows':len(rows),'DP_cost':plan.dp_cost,'DP_margin':plan.dp_margin,
          'DP_diagnostics':plan.diagnostics,
          'row_to_passage':[x.passage_index for x in plan.assignments],
          'row_DP_confidence':[x.confidence for x in plan.assignments],
          'mean_ctc_gain':round(gain,5),
          'displaced_fraction':round(displaced,5),
          'boundary_competition':boundary_diag,
          'two_dp_acoustic':dual_diag,
          'group_placement_acoustic':placement_diag,
          'raw_asr_corroboration':asr_corroboration,
          'legacy_global_gain_diagnostics':legacy_gain_diagnostics,
          'repeated_global_ctc_conflicts':repeat_conflicts,
          'runtime_cache_hit':cached is not None,
          'passage_energies':[
             {"start":round(p.start,3),"end":round(p.end,3),
              "mean_rms_db":round(float(audio_api.np.mean(
                features.rms_db[(features.frame_times>=p.start)
                                &(features.frame_times<p.end)])),3)
                   if hasattr(features,"rms_db") and hasattr(features,"frame_times")
                   and bool(audio_api.np.any((features.frame_times>=p.start)
                                            &(features.frame_times<p.end)))
                   else None}
             for p in plan.passages
          ],
        }
        cloned['segment_skeleton_status']=status
        audio_api.annotate_ctc_boundary_evidence(assignments)
        audio_api.update_report_confidence_metrics(cloned)
        return output_times,cloned,status
    except (OSError,ValueError,TypeError,RuntimeError,TimeoutError,
            json.JSONDecodeError) as exc:
        return fallback('SEGMENT_EXECUTION_FAILED',
                        failure_type=type(exc).__name__,detail=str(exc)[:200])
