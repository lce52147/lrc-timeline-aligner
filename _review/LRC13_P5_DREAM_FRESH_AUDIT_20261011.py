"""Posthoc strict analysis of fresh P5 Dream CLI; read-only."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
sys.path.insert(0,str(V))
from LRC13_A7_FINALIZE_20261010 import parse,report
item=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf-8'))['corpus'][37]
fresh=V/'LRC13_P5_FRESH_DREAM_20261011'
old=V/'lrc13_option_a_A7_39_20261010'/'38_09.夢遊'
a=report(fresh/'reports')
sk=a.get('segment_skeleton_status') or {}
entries=a.get('assignments') or []
gold=parse(Path(item['human_BK']),item['audio_path'])
now=parse(fresh/'candidate.lrc',item['audio_path'])
prior=parse(old/'candidate.lrc',item['audio_path'])
assert len(gold)==len(now)==len(prior)==len(entries)==30
assert all(a[0]==b[0]==c[0] for a,b,c in zip(gold,now,prior))
flags=a.get('final_self_check',{}).get('flags_by_entry',{})
row=entries[27]
rflags=flags.get('28',[])
res={'status':sk.get('status'),'row28':{'BK':gold[27][1],
  'P5_written':now[27][1],'preP5_written':prior[27][1],
  'global_ctc_first_token_start':row.get('global_ctc_first_token_start'),
  'ctc_first_token_start':row.get('ctc_first_token_start'),
  'self_check_flags':row.get('self_check_flags'),
  'final_self_check_flags':rflags,
  'timing_trusted':row.get('timing_trusted')},
 'fresh_global_field_count':sum('global_ctc_first_token_start' in x for x in entries),
 'line_count':len(entries),
 'timestamp_match_rows':sum(abs(q[1]-r[1])<.011 for q,r in zip(now,prior)),
 'timestamp_diffs':[
    {'line':i,'before':b[1],'after':n[1]}
    for i,(n,b) in enumerate(zip(now,prior),1) if abs(n[1]-b[1])>=.011
 ]}
path=V/'LRC13_P5_DREAM_FRESH_AUDIT_20261011.json'
path.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print('P5_DREAM_AUDIT',json.dumps(res,ensure_ascii=True),flush=True)
assert sk.get('status')=='SEGMENT_CURRENT','SKELETON_DISABLED'
assert res['fresh_global_field_count']==30,'MISSING_GLOBAL_EVIDENCE'
assert 'skeleton_global_ctc_disagreement' in rflags,'F5_FLAG_ABSENT'
assert row.get('timing_trusted') is False,'F5_TRUST_NOT_DOWNGRADED'
assert res['timestamp_match_rows']==30,'TIMESTAMP_DRIFT'
print('P5_DREAM_FRESH_ACCEPTED',flush=True)
