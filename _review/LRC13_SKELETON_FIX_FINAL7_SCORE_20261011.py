"""Score frozen 7x2 full Central A/B runs, never touch candidate LRC."""
from pathlib import Path
import json,sys
v=Path(__file__).resolve().parent
src=v/'LRC13_OPTION_A_FINAL7_FULL_CENTRAL_20261010'/'FINAL7_SERIAL_MANIFEST.json'
man=json.loads(src.read_text(encoding='utf8'))
cases=man.get('cases',[])
assert len(cases)==14,('INCOMPLETE_7_AB',len(cases))
assert all(x.get('status')=='GENERATED' for x in cases),'FAILED_CASES'
by={}
for x in cases:
    pair=by.setdefault(x['title'],{})
    assert x['mode'] not in pair
    pair[x['mode']]=x
assert len(by)==7 and all(set(x)=={'ctc','segment'} for x in by.values())
rows=[];global_count={'skeleton_pco':0,'ctc_pco':0,'skeleton_gt1':0,'ctc_gt1':0,
    'new_gt1':0,'active':0,'active_worse':0,'new_backtracks':0}
for song,pair in by.items():
    a,b=pair['segment'],pair['ctc']
    seg=a.get('posthoc_BK') or {};ctc=b.get('posthoc_BK') or {}
    if not seg.get('rows') or seg.get('rows')!=ctc.get('rows'):
        raise SystemExit('INVALID_BK_SCORE '+song+': '+str((seg,ctc)))
    new=sorted(set(seg['bad_rows'])-set(ctc['bad_rows']))
    active=(a.get('skeleton_report') or {}).get('status')=='SEGMENT_CURRENT'
    worse=bool(active and (seg['PCO200']<ctc['PCO200'] or new))
    regressed=seg['nonmonotonic']>ctc['nonmonotonic']
    rec={'song':song,'active':active,
      'segment_PCO200':seg['PCO200'],'ctc_PCO200':ctc['PCO200'],
      'segment_gt1':seg['wrong_gt1'],'ctc_gt1':ctc['wrong_gt1'],
      'new_gt1':new,'worse':worse,'new_backtracks':max(0,seg['nonmonotonic']-ctc['nonmonotonic']),
      'trusted_segment':a.get('central_trusted'),
      'trusted_ctc':b.get('central_trusted')}
    rows.append(rec)
    global_count['skeleton_pco']+=seg['PCO200'];global_count['ctc_pco']+=ctc['PCO200']
    global_count['skeleton_gt1']+=seg['wrong_gt1'];global_count['ctc_gt1']+=ctc['wrong_gt1']
    global_count['new_gt1']+=len(new)
    global_count['active']+=int(active)
    global_count['active_worse']+=int(worse)
    global_count['new_backtracks']+=rec['new_backtracks']
    print('FINAL7',song.encode('unicode_escape').decode(),'active',active,
      'pcos',seg['PCO200'],ctc['PCO200'],'new>1',len(new),flush=True)
passed=(not global_count['active_worse']
        and global_count['new_gt1']==0
        and global_count['new_backtracks']==0)
out={'gate':'PASS_FINAL7_AB' if passed else 'NOT_PASS_FINAL7_AB',
     'stage':'full CLI Central reviewer-validity segment-vs-CTC 7 songs',
     'does_not_override_A7':True,'global':global_count,'songs':rows}
(v/'LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.json').write_text(
 json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('FINAL7_AB_GATE',out['gate'],global_count,flush=True)
if not passed:raise SystemExit(3)
