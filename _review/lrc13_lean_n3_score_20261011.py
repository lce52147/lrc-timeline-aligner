"""N3 BK posthoc across 39 frozen full-song emissions / exact constrained Viterbi.
No score runs until ALL 39 N3 hypothesis files are frozen.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,math
V=Path(__file__).resolve().parent
F=V/'LRC13_LEAN_N3_CONSTRAINED_20261011'
m=json.loads((F/'MANIFEST.json').read_text(encoding='utf8'))
if m.get('status')!='N3_EXACT_39_FROZEN' or len(m.get('cases',[]))!=39 or any(z['status']!='DONE' for z in m['cases']):
 raise SystemExit('NO_BK_UNTIL_N3_39_FROZEN')
B=json.loads((V/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
rows_by_song=defaultdict(list)
for row in B['rows']:rows_by_song[row['song']].append(row)
thresholds=(.10,.12,.15)
out={'status':'N3_39_POSTHOC_SCORED','thresholds':thresholds,'devset_only':True,
 'zero_parity_all':True,'cases':{},'aggregate':{}}
def stats(rows,field):
 return {'PCO200':sum(abs(z[field]-z['BK'])<=.2+1e-9 for z in rows),
         'gt1':sum(abs(z[field]-z['BK'])>1 for z in rows)}
for k in thresholds:
 key=str(k)
 for scope in ('flagged_only','all_39_diagnostic'):
  seq=[];songs=[]
  for case in m['cases']:
   i=case['index'];name=case['id']
   dat=json.loads(Path(case['output']).read_text(encoding='utf8'))
   frozen=next(z for z in json.loads((V/'LRC13_LEAN_N3_GPU_20261011/MANIFEST.json').read_text(encoding='utf8'))['cases'] if z['index']==i)
   assert dat['BK_read'] is False and dat['zero_parity'] and dat['status']=='N3_EXACT_FROZEN'
   # Match the corpus item by exact ID; main report excludes a first legacy provenance row in one song.
   item=next(z for z in json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus'] if z['id']==name)
   song=item.get('title',item['id'])
   base=sorted(rows_by_song[song],key=lambda r:r['line'])
   input_times=dat['H0_starts'];candidate=dat['threshold_results'][key]['starts']
   delta=len(input_times)-len(base)
   assert delta in (0,1),(song,delta)
   if delta: assert frozen['romans'][0]=='ooggrana'
   assert len(candidate)==len(input_times)
   act=bool(dat['flagged']) or scope=='all_39_diagnostic'
   result_rows=[]
   for j,row in enumerate(base):
    c=float(candidate[j+delta]) if act else float(input_times[j+delta])
    h0=float(input_times[j+delta])
    assert abs(h0-row['CTC_s'])<.012,(song,j,h0,row['CTC_s'])
    r={'song':song,'line':j+1,'BK':row['BK_s'],'H0':h0,'N3':c,
       'text':row['text'],'moved':abs(c-h0)>.0001,
       'was_full_39_diagnostic':scope=='all_39_diagnostic',
       'flagged_song':bool(dat['flagged'])}
    result_rows.append(r);seq.append(r)
   songs.append({'song':song,'flagged':dat['flagged'],
     'H0':stats(result_rows,'H0'),'N3':stats(result_rows,'N3'),
     'moved':sum(x['moved'] for x in result_rows)})
  assert len(seq)==1487
  new=[{'song':z['song'],'line':z['line'],'BK':z['BK'],'H0':z['H0'],'N3':z['N3']}
       for z in seq if abs(z['H0']-z['BK'])<=1 and abs(z['N3']-z['BK'])>1]
  back=[]
  by=defaultdict(list)
  for z in seq:by[z['song']].append(z)
  for song,zs in by.items():
   for a,b in zip(zs,zs[1:]):
    if a['N3']>b['N3']+1e-8 and a['H0']<=b['H0']+1e-8:
     back.append({'song':song,'line':a['line']})
  focus=('Choir','ミレニア','My Dearest','影色舞','夢遊','Ever be my love','レインフォール')
  out['cases'][key+'_'+scope]={
    'threshold':k,'scope':scope,
    'baseline':stats(seq,'H0'),'N3':stats(seq,'N3'),'new_gt1':new,
    'new_backtracks':back,
    'max_move':max(abs(x['N3']-x['H0']) for x in seq),
    'moved_rows':sum(z['moved'] for z in seq),
    'songs':songs,
    'focus_rows':[z for z in seq if any(tag in z['song'] for tag in focus)]}
  print('N3_SCORE',key,scope,'H0',stats(seq,'H0'),'N3',stats(seq,'N3'),
    'new_gt1',len(new),'new_backtracks',len(back),flush=True)
(V/'LRC13_LEAN_N3_SCORE_20261011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
