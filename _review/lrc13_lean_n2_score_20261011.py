"""BK posthoc scorer: refuses incomplete N2 39-song manifest and outputs."""
from pathlib import Path
from collections import defaultdict,Counter
import sys,json
V=Path(__file__).resolve().parent;ROOT=V.parent
sys.path[:0]=[str(V),str(ROOT/'scripts')]
from LRC13_PATCH_A7_SCORE_20261011 import parse,report,aligned_trust_assignments
OUT=V/'LRC13_LEAN_N2_CTC_FIRST_39_20261011'
m=json.loads((OUT/'MANIFEST.json').read_text(encoding='utf8'))
if m.get('phase')!='N2_39_FROZEN' or len(m.get('cases',[]))!=39 or any(x['status']!='PRODUCED' for x in m['cases']):
 raise SystemExit('N2_SCORING_BLOCKED_NOT_FROZEN_39')
b=json.loads((V/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
data=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
rowmap=defaultdict(list)
for row in b['rows']:rowmap[row['song']].append(row)
flags=json.loads((V/'LRC13_LEAN_N1_FLAGS_20261011.json').read_text(encoding='utf8'))
assert flags['status']=='N1_FLAGS_BK_BLIND_FROZEN_39'
flagmap={(z['song'],z['line']):z for z in flags['rows']}
allrows=[];songs=[];source=Counter()
for i,item in enumerate(data,1):
 name=item.get('title',item['id'])
 leaf=f'{i:02d}_{item["id"]}'
 path=OUT/leaf/'candidate.lrc'
 new=parse(path,item['audio_path'])
 old=sorted(rowmap[name],key=lambda z:z['line'])
 assert len(new)==len(old),(name,len(new),len(old))
 assert all(z[0]==r['text'] for z,r in zip(new,old)),('LYRIC_MISMATCH',name)
 report_new=report(OUT/leaf/'reports')
 aligned=aligned_trust_assignments(report_new,path,item['audio_path'],len(new),name,'N2')
 for j,(z,r,a) in enumerate(zip(new,old,aligned),1):
  f=flagmap[name,j]
  tr=a.get('timing_trusted')
  if type(tr) is not bool:raise RuntimeError('N2_TRUST_MISSING '+name+' '+str(j))
  src=a.get('candidate_source')
  if 'whisper' in str(src).lower() or 'raw' in str(src).lower():
   raise RuntimeError('N2_RESTRICTED_FINAL_TIME_SOURCE '+name+' '+str(j)+' '+str(src))
  source[src or 'unknown']+=1
  allrows.append({'song':name,'line':j,'text':z[0],'BK':r['BK_s'],
      'CTC':r['CTC_s'],'main':r['main_s'],'A7':r['lrc13_s'],
      'N2':z[1],'N2_source':src,'trusted':tr,
      'N1_flagged':f['flagged'],'N1_flags':f['flags']})
 songs.append({'song':name,'rows':len(new)})
assert len(allrows)==1487
def stat(z,key):
 wrong=[x for x in z if abs(x[key]-x['BK'])>1]
 return {'PCO200':sum(abs(x[key]-x['BK'])<=.2+1e-9 for x in z),
 'gt1':len(wrong),'gt05':sum(abs(x[key]-x['BK'])>.5 for x in z)}
scores={k:stat(allrows,k) for k in ('CTC','main','A7','N2')}
new1=[{'song':r['song'],'line':r['line'],'BK':r['BK'],'A7':r['A7'],'N2':r['N2']}
 for r in allrows if abs(r['A7']-r['BK'])<=1 and abs(r['N2']-r['BK'])>1]
newmain=[{'song':r['song'],'line':r['line'],'BK':r['BK'],'main':r['main'],'N2':r['N2']}
 for r in allrows if abs(r['main']-r['BK'])<=1 and abs(r['N2']-r['BK'])>1]
back=[]
by=defaultdict(list)
for r in allrows:by[r['song']].append(r)
for name,rr in by.items():
 for a,c in zip(rr,rr[1:]):
  if a['N2']>c['N2']+1e-8 and a['A7']<=c['A7']+1e-8:
   back.append({'song':name,'line':a['line']})
 trusted=[r for r in allrows if r['trusted']]
 trusted_wrong=[r for r in trusted if abs(r['N2']-r['BK'])>1]
silent=[r for r in allrows if abs(r['N2']-r['BK'])>1 and not r['N1_flagged']]
silent_half=[r for r in allrows if abs(r['N2']-r['BK'])>.5 and not r['N1_flagged']]
active_sources={s:c for s,c in source.items() if 'whisper' in str(s).lower() or 'raw' in str(s).lower()}
assert not active_sources
for q in songs:
 rr=by[q['song']]
 q.update({'N2':stat(rr,'N2'),'A7':stat(rr,'A7'),
   'new_gt1_vs_A7':[r['line'] for r in rr if abs(r['A7']-r['BK'])<=1 and abs(r['N2']-r['BK'])>1]})
result={'status':'N2_POSTHOC_39_COMPLETE','scope':'opt_in_ctc_first','counts':scores,
 'new_gt1_vs_A7':new1,'new_gt1_vs_main':newmain,
 'new_backtracks_vs_A7':back,
 'trusted_total':len(trusted),'trusted_wrong_gt1':len(trusted_wrong),
 'trusted_wrong_gt1_ratio':len(trusted_wrong)/len(trusted) if trusted else None,
 'N1_fixed_flags_listen':sum(r['N1_flagged'] for r in allrows),
 'N1_fixed_flags_silent_gt1':silent,'N1_fixed_flags_silent_gt05':silent_half,
 'source_counts':dict(source),'songs':songs,'rows':allrows,
 'skeleton_reference':{'PCO200':1314,'gt1':83},
 'devset_only':True}
(V/'LRC13_LEAN_N2_SCORE_20261011.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('N2_SCORED',json.dumps({'scores':scores,'new_gt1_vs_A7':len(new1),'backtracks':len(back),
 'trusted_wrong':len(trusted_wrong),'listen':result['N1_fixed_flags_listen'],
 'silent_gt1':len(silent),'sources':dict(source)},ensure_ascii=True)[:2400])
