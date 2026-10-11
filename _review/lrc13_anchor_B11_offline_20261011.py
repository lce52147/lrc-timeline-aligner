"""B11 full-song CTC clocks vs main and 1.3 final times, common BK scorer."""
from pathlib import Path
import sys,json,collections,math
V=Path(__file__).resolve().parent
ROOT=V.parent
sys.path[:0]=[str(V),str(ROOT/'scripts')]
from LRC13_PATCH_A7_SCORE_20261011 import parse,report,aligned_trust_assignments
items=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
MAIN=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
NEW=V/'LRC13_FINAL_A7_PATCHED_39_20261011'
rows=[];songs=[];missing=[];source_counts=collections.defaultdict(lambda:collections.Counter())
def layer(source:str)->str:
 s=(source or '').lower()
 if 'ctc' in s and 'fusion' not in s and 'whisper' not in s:return 'CTC/local'
 if 'whisper' in s or 'raw' in s:return 'Whisper/RAW'
 if 'review' in s or 'hubert' in s:return 'reviewer'
 if 'phase4' in s:return 'Phase4B'
 if 'vocal' in s:return 'vocal-onset'
 return source or 'unknown'
for i,item in enumerate(items,1):
 name=item.get('title',item['id'])
 leaf=f'{i:02d}_{item["id"]}'
 gold=parse(Path(item['human_BK']),item['audio_path'])
 main=parse(MAIN/leaf/'main.lrc',item['audio_path'])
 new=parse(NEW/leaf/'candidate.lrc',item['audio_path'])
 if len(gold)==len(main)+1 and gold[0][1]==0:gold=gold[1:]
 assert len(gold)==len(main)==len(new),repr((name,len(gold),len(main),len(new)))
 assert all(a[0]==b[0]==c[0] for a,b,c in zip(gold,main,new)),name
 mrep=report(MAIN/leaf/'reports')
 nrep=report(NEW/leaf/'reports')
 ma=aligned_trust_assignments(mrep,MAIN/leaf/'main.lrc',item['audio_path'],len(main),name,'main')
 na=aligned_trust_assignments(nrep,NEW/leaf/'candidate.lrc',item['audio_path'],len(new),name,'new')
 cs=[]
 for j,(b,m,n,ar,an) in enumerate(zip(gold,main,new,ma,na),1):
  raw=ar.get('ctc_first_token_start')
  ctc=float(raw) if isinstance(raw,(int,float)) else None
  if ctc is None:missing.append([name,j]);continue
  data={'song':name,'line':j,'text':b[0],'BK_s':b[1],'CTC_s':ctc,
        'main_s':m[1],'lrc13_s':n[1],'main_source':ar.get('candidate_source'),
        'lrc13_source':an.get('candidate_source'),'ctc_score':ar.get('ctc_score')}
  data['errors']={k:round(abs(data[k]-b[1]),5) for k in ('CTC_s','main_s','lrc13_s')}
  rows.append(data);cs.append(data)
 def scores(data,key):
  return {'PCO200':sum(abs(r[key]-r['BK_s'])<=.2+1e-9 for r in data),
          'wrong_gt1':sum(abs(r[key]-r['BK_s'])>1 for r in data)}
 songs.append({'song':name,'rows':len(cs),'ctc':scores(cs,'CTC_s'),
               'main':scores(cs,'main_s'),'lrc13':scores(cs,'lrc13_s')})
 assert len(cs)==len(gold),(name,len(cs),len(gold))
def score(key):
 return {'PCO200':sum(abs(r[key]-r['BK_s'])<=.2+1e-9 for r in rows),
         'wrong_gt1':sum(abs(r[key]-r['BK_s'])>1 for r in rows)}
analysis={}
for which,key in [('main','main_s'),('lrc13','lrc13_s')]:
 changed=[r for r in rows if abs(r[key]-r['CTC_s'])>.05]
 improved=[r for r in changed if r['errors']['CTC_s']-r['errors'][key]>.2]
 worsened=[r for r in changed if r['errors'][key]-r['errors']['CTC_s']>.2]
 counters={}
 for r in changed:
  source=layer(r[which+'_source'])
  c=counters.setdefault(source,{'changed':0,'improved':0,'worsened':0,
       'new_wrong_gt1':0,'repaired_gt1':0})
  c['changed']+=1
  if r['errors']['CTC_s']-r['errors'][key]>.2:c['improved']+=1
  if r['errors'][key]-r['errors']['CTC_s']>.2:c['worsened']+=1
  if r['errors']['CTC_s']<=1 and r['errors'][key]>1:c['new_wrong_gt1']+=1
  if r['errors']['CTC_s']>1 and r['errors'][key]<=1:c['repaired_gt1']+=1
 analysis[which]={'changed_count':len(changed),'improved_count':len(improved),
                  'worsened_count':len(worsened),'net_improved_minus_worsened':len(improved)-len(worsened),
                  'layer_counts':counters,
                  'changed_line_records':changed}
out={'status':'B11_COMMON_ALIGNED_39','row_count':len(rows),'missing_ctc':missing,
 'score':{k:score(k) for k in ('CTC_s','main_s','lrc13_s')},
 'hypothesis_reference':{'skeleton_ctc':[1285,115],'main_final':[1268,117]},
 'score_consistent_with_prior':score('CTC_s')=={'PCO200':1285,'wrong_gt1':115},
 'analysis':analysis,'songs':songs,'rows':rows,
 'CTC_source':'frozen pinned main align-report assignments[].ctc_first_token_start; no Central clock'}
(V/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('B11',out['status'],len(rows),'scores',json.dumps(out['score']),flush=True)
print('LAYERS',json.dumps({k:v['layer_counts'] for k,v in analysis.items()},ensure_ascii=True)[:2800],flush=True)