"""Final posthoc-only project report; requires full 7 A/B and final A7 scores.

No reference clocks used by inference. Solely reads frozen reports and BK for
retrospective scoring. Never copies or modifies source BK/audio, main, OOGG.
"""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(V))
from LRC13_A7_FINALIZE_20261010 import parse,report,aligned_trust_assignments
def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
old=read(V/'LRC13_A7_REPORT_20261010.json')
quick=read(V/'LRC13_ASR_CORR_39_SCORE_20261011.json')
ab=read(V/'LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.json')
a7=read(V/'LRC13_SKELETON_FIX_FINAL_A7_REPORT_20261011.json')
assert quick['experimental_asr_correlated_hard_gate_pass']
assert ab['gate']=='PASS_FINAL7_AB','HARD_STOP_AB_NOT_PASS'
assert a7.get('status')=='A7_SCORED_39' and len(a7.get('songs',[]))==39
items=read(V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json')['corpus']
assert len(items)==39
base=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
historic=V/'lrc13_option_a_A7_39_20261010'
current=V/'LRC13_SKELETON_FIX_FINAL_A7_39_20261011'
allflags=[];recall=Counter();dream=[]
for number,item in enumerate(items,1):
    label=item.get('title',item['id'])
    leaf=f'{number:02d}_{item["id"]}'
    expected=parse(Path(item['human_BK']),item['audio_path'])
    candidate=parse(current/leaf/'candidate.lrc',item['audio_path'])
    if len(expected)==len(candidate)+1 and expected[0][1]==0:
        expected=expected[1:]
    if len(expected)!=len(candidate):raise RuntimeError('ROW_COUNT '+label)
    output_report=report(current/leaf/'reports')
    sk=output_report.get('segment_skeleton_status') or {}
    active=sk.get('status')=='SEGMENT_CURRENT'
    assigned=aligned_trust_assignments(output_report,current/leaf/'candidate.lrc',
         item['audio_path'],len(candidate),label,'final')
    check=output_report.get('final_self_check') or {}
    f4f5=check.get('flags_by_entry') or {}
    for index,(gold,found,assignment) in enumerate(zip(expected,candidate,assigned),1):
        error=abs(found[1]-gold[1])
        if not active or error<=1:continue
        recall['active_gt1']+=1
        flags=list(assignment.get('flags') or [])+list(assignment.get('self_check_flags') or [])
        extra=f4f5.get(str(index),[])
        if isinstance(extra,list):flags+=extra
        elif isinstance(extra,dict):flags+=list(extra.get('flags') or [])
        flag_list=[str(x) for x in flags]
        by={'F4':any('F4-' in x or 'ctc-final-disagreement' in x for x in flag_list),
            'F5':any('F5-' in x or 'skeleton_global_ctc_disagreement' in x for x in flag_list),
            'P2':any('segment_structural_ambiguity' in x for x in flag_list),
            'P3':any('group_placement_unsupported' in x for x in flag_list)}
        for mark,ok in by.items():
            if ok:recall[mark]+=1
        if any(by.values()):recall['flagged_any']+=1
        allflags.append({'song':label,'line':index,'error_s':round(error,3),
          'flags':flag_list,'review_required':bool(assignment.get('review_required')),
          'timing_trusted':assignment.get('timing_trusted'),'detection':by})
    if label=='09.夢遊':
        historical=parse(historic/leaf/'candidate.lrc',item['audio_path'])
        orig=parse(base/leaf/'main.lrc',item['audio_path'])
        if not len(expected)==len(historical)==len(orig)==30:raise RuntimeError('DREAM_COUNT')
        for row,(gold,prev_main,prev_seg,now) in enumerate(zip(expected,orig,historical,candidate),1):
            dream.append({'line':row,'text':gold[0],'BK':round(gold[1],3),
                'pinned_main':round(prev_main[1],3),
                'old_A7':round(prev_seg[1],3),
                'final_A7':round(now[1],3),
                'old_A7_abs_err':round(abs(prev_seg[1]-gold[1]),3),
                'final_abs_err':round(abs(now[1]-gold[1]),3)})
assert len(dream)==30
counts_old=old['corpus']['counts']
counts_new=a7['corpus']['counts']
summary={
 'status':'FINAL_ARTIFACTS_SCORED',
 'release_pass':bool(a7.get('release_pass')),
 'branch':'lrc-1.3','quick_gate_pass':True,
 'quick':quick['counts'],'complete7':ab['global'],
 'old_A7':counts_old,'final_A7':counts_new,
 'quick_gate_not_independent_holdout':True,
 'final_A7_active':a7['active_count'],
 'final_A7_active_worse':a7.get('worsening_active_songs',[]),
 'F4_F5_P2_P3_recall':dict(recall),
 'F4_F5_P2_P3_review_rows':allflags,
 'dream_30_rows':dream,
 'a7_new_gt1':len(a7.get('new_gt1_lines',[])),
 'a7_inherited_backtracks':a7.get('inherited_backtracks',[]),
 'a7_new_backtracks':a7.get('new_backtracks',[]),
 'BK_times_posthoc_only':True,'no_push_or_main_merge':True
}
(V/'LRC13_SKELETON_FIX_REPORT_20261011.json').write_text(
 json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# LRC 1.3 Skeleton Fix — Final validation report',
       '',
       f"**Overall release gate: {'PASS' if summary['release_pass'] else 'NOT_PASS'}**",
       '',
       '## Test scope and limitations',
       '- P5 preserved global CTC evidence; P4 shared identical CTC confidence.',
       '- P1 raw-ASR paragraph lexical DP; P2 min-sum per-line marginals.',
       '- P3 local CTC compares group with neighboring passages.',
       '- P6 marks weak CTC score <0.005 as review/untrusted, with hard failure <0.0002.',
       '- Material shifts require raw-ASR paragraph text corroboration; repeat/global conflicts remain diagnostics.',
       '- BK timestamps were never inference priors: only posthoc scoring.',
       '- 39-song quick gate has been used during design; not an independent holdout.',
       '',
       '## Frozen 39-song skeleton quick gate',
       f"Counts: {quick['counts']}",
       '',
       '## Seven-song full Central A/B',
       f"Status: {ab['gate']}; {ab['global']}",'',
       '## Final 39-song A7 versus pinned main and old A7',
       f"Old A7: {counts_old}",
       f"Final A7: {counts_new}",
       f"New >1s: {summary['a7_new_gt1']}; active: {a7['active_count']}/39;",
       f"active-worse: {a7.get('worsening_active_songs',[])}",
       '',
       '## F4 / F5 / P2 / P3 flag recall for active >1s rows',
       str(dict(recall)),
       '',
       '## Dream 30-row ledger',
       '| # | BK | pinned main | old A7 | new A7 | old abs error | new abs error |',
       '|---:|---:|---:|---:|---:|---:|---:|']
for row in dream:
    lines.append(f"| {row['line']} | {row['BK']:.2f} | {row['pinned_main']:.2f} | {row['old_A7']:.2f} | {row['final_A7']:.2f} | {row['old_A7_abs_err']:.2f} | {row['final_abs_err']:.2f} |")
lines+=['','## Full 39 A7 review flags','See LRC13_SKELETON_FIX_FINAL_A7_LISTEN_FLAGS_20261011.md',
        '','No push, no merge into main. BK and audio unchanged.']
(V/'LRC13_SKELETON_FIX_REPORT_20261011.md').write_text(
    '\n'.join(lines)+'\n',encoding='utf8')
print('FINAL_PROJECT_REPORT_WRITTEN',summary['release_pass'],
      summary['final_A7_active'],dict(recall),flush=True)
