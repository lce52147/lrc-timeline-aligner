"""Posthoc BK scoring; runs ONLY after 39 skeleton-layer inference outputs freeze."""
from pathlib import Path
from collections import Counter
import json,sys,math
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/'_review'
sys.path.insert(0,str(V))
from LRC13_A7_FINALIZE_20261010 import parse
items=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
S=V/'LRC13_P1_SKELETON_39_20261011'
man=json.loads((S/'RUN_MANIFEST.json').read_text(encoding='utf8'))
assert man.get('status')=='COMPLETE_39' and len(man.get('runs',[]))==39,('INCOMPLETE',man.get('status'),len(man.get('runs',[])))
results=[]; introduced=[]; all_wrong=0;all_flagged=0
for i,item in enumerate(items,1):
    name=item.get('title',item['id'])
    path=S/f'{i:02d}_{item["id"]}'/'stage.json'
    stage=json.loads(path.read_text(encoding='utf8'))
    raw=parse(Path(item['human_BK']),item['audio_path'])
    if len(raw)==item['BK_sung_rows']+1 and raw[0][1]==0:raw=raw[1:]
    assert len(raw)==len(stage['baseline_times'])==len(stage['skeleton_times']),name
    b=[x[1] for x in raw];a=stage['baseline_times'];c=stage['skeleton_times']
    m=len(b);correct_old=sum(abs(x-y)<=.2 for x,y in zip(a,b))
    correct_new=sum(abs(x-y)<=.2 for x,y in zip(c,b))
    gt1old=sum(abs(x-y)>1 for x,y in zip(a,b))
    gt1new=sum(abs(x-y)>1 for x,y in zip(c,b))
    new=[{'line':j+1,'BK':round(b[j],2),
         'global_CTC':round(a[j],3),'local_skeleton':round(c[j],3)}
         for j in range(m) if abs(a[j]-b[j])<=1 and abs(c[j]-b[j])>1]
    active=stage['status']=='SEGMENT_CURRENT'
    backtracks=[j+1 for j in range(m-1) if c[j+1]<c[j]]
    source_backtracks=[j+1 for j in range(m-1) if a[j+1]<a[j]]
    new_backtracks=[j for j in backtracks if j not in source_backtracks]
    changed=sum(abs(x-y)>.011 for x,y in zip(a,c))
    record={'song':name,'index':i,'active':active,'status':stage['status'],
            'reason':stage.get('reason'),
            'rows':m,'PCO200_old':correct_old,'PCO200_new':correct_new,
            'wrong_gt1_old':gt1old,'wrong_gt1_new':gt1new,
            'new_gt1':new,'new_backtracks':new_backtracks,
            'changed_clocks':changed,'stage_seconds':stage.get('elapsed_s'),
            'worse':active and (correct_new<correct_old or len(new)>0)}
    results.append(record)
    introduced.extend({'song':name,**x} for x in new)
    print('P1_POSTHOC',i, name.encode('unicode_escape').decode(),
          'ACTIVE' if active else 'FALLBACK',gt1old,gt1new,flush=True)
total={'rows':sum(x['rows'] for x in results),
       'pco_old':sum(x['PCO200_old'] for x in results),
       'pco_new':sum(x['PCO200_new'] for x in results),
       'gt1_old':sum(x['wrong_gt1_old'] for x in results),
       'gt1_new':sum(x['wrong_gt1_new'] for x in results),
       'new_gt1':len(introduced),
       'new_backtracks':sum(len(x['new_backtracks']) for x in results),
       'active':sum(x['active'] for x in results)}
fail=[x['song'] for x in results if x['worse']]
data={'status':'P1_SKELETON_LAYER_39_POSTHOC_SCORED',
      'scope':'Skeleton-layer local CTC uses cached vocal stem and frozen pinned main CTC seed; no Central/WhisperX/reviewer',
      'not_same_as_full_A7':'Frozen pinned main CTC alignment evidence may differ from global CTC at lrc-1.3 stage',
      'full_p1_hard_gate_pass':not fail and total['new_backtracks']==0,
      'active_worse_songs':fail,'counts':total,'songs':results,'introduced_gt1':introduced,
      'BK_scoring_only_after_inference':True}
(V/'LRC13_P1_SKELETON_39_SCORE_20261011.json').write_text(
 json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# P1 skeleton-layer 39-song posthoc score','',
       'Frozen pinned-main CTC evidence as seed; true local skeleton inference without Central/WhisperX/reviewer.',
       'BK used only after inference complete. Full A7 scores are a different stage.',
       f'Status: {data["status"]}; hard gate: {data["full_p1_hard_gate_pass"]}',
       f'Counts: {total}; active worse songs: {fail}','',
       '| Song | mode | PCO@0.2 CTC→skeleton | >1s CTC→skeleton | newly >1s | new backtrack |',
       '|---|---|---:|---:|---:|---:|']
for x in results:
    lines.append(f'| {x["song"].replace("|","/")} | {x["status"]} / {x.get("reason")} | {x["PCO200_old"]}→{x["PCO200_new"]}/{x["rows"]} | {x["wrong_gt1_old"]}→{x["wrong_gt1_new"]} | {len(x["new_gt1"])} | {len(x["new_backtracks"])} |')
lines+=['','## New >1s errors']
for v in introduced:
    lines.append(f'- {v["song"]} #{v["line"]}: BK {v["BK"]:.2f}, CTC {v["global_CTC"]:.3f}, skeleton {v["local_skeleton"]:.3f}')
(V/'LRC13_P1_SKELETON_39_SCORE_20261011.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('P1_39_SCORE_DONE',json.dumps(total), 'ACTIVE_WORSE',len(fail),flush=True)