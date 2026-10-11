"""Posthoc only. Refuses incomplete 39-case frozen GPU manifest before BK access."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys,math,statistics
V=Path(__file__).resolve().parent;ROOT=V.parent
sys.path[:0]=[str(V),str(ROOT/'scripts')]
from LRC13_PATCH_A7_SCORE_20261011 import parse,report,aligned_trust_assignments
import auto_lrc as al
F=V/'LRC13_ANCHOR_GPU_V2_20261011'
manifest=json.loads((F/'MANIFEST.json').read_text(encoding='utf8'))
if manifest.get('status')!='ANCHOR_GPU_39_FROZEN' or len(manifest.get('cases',[]))!=39 or any(x['status']!='DONE' for x in manifest['cases']):
 raise SystemExit('BK_POSTHOC_BLOCKED_UNFROZEN_GPU_39')
items=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
B=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
expected=(.5,2.,5.)
bucket={str(x):[] for x in expected}
anchor_songs=[];interval_counts=Counter();coverage=Counter();interval_rows=[]
foci={'09.夢遊','04 影色舞','03. Choir ‘S’ Choir','01.ミレニアの水槽','01 My Dearest'}
focus_rows=[];all_moved=[]
def one_score(rows,key):
 return {'PCO200':sum(abs(x[key]-x['BK'])<=.2+1e-9 for x in rows),
         'wrong_gt1':sum(abs(x[key]-x['BK'])>1 for x in rows)}
def repeat_key(s):
 return al.normalize_match_text(s)
for i,item in enumerate(items,1):
 name=item.get('title',item['id'])
 leaf=f'{i:02d}_{item["id"]}'
 out=json.loads((F/leaf/'FROZEN_HYPOTHESES.json').read_text(encoding='utf8'))
 if out.get('status')!='FROZEN_H0_H1' or out.get('BK_was_accessed') is not False:
  raise SystemExit('UNFROZEN_OR_BK_CONTAMINATED_CASE '+name)
 mainfile=B/leaf/'main.lrc';mrep=report(B/leaf/'reports')
 source=mrep['assignments']
 gold=parse(Path(item['human_BK']),item['audio_path'])
 main=parse(mainfile,item['audio_path'])
 if len(gold)==len(main)+1 and gold[0][1]==0:gold=gold[1:]
 assert len(gold)==len(main),name
 if len(source)==len(gold):
  keep=list(range(len(gold)))
 else:
  raw=al.load_lyrics(mainfile).entries
  raw=al.remove_generated_title_cards(raw,Path(item['audio_path']))
  clean,skipped=al.remove_instrumental_markers(raw)
  assert len(raw)==len(source) and len(clean)==len(gold),name
  keep=[j for j,x in enumerate(raw) if not al.is_instrumental_marker(x)]
 assert len(keep)==len(gold)
 anchors=set(out.get('anchors',[]))
 aligned_starts=[out['delta_results'][str(.5)][k] for k in keep]
 assert len(aligned_starts)==len(gold) and all(isinstance(x,(int,float)) for x in aligned_starts)
 distinct=Counter(repeat_key(z[0]) for z in gold)
 scored_anchors=[]
 for j,k in enumerate(keep):
  anchor=(k+1) in anchors
  c=float(source[k]['ctc_first_token_start'])
  assert abs(c-aligned_starts[j])<.011,(name,j,c,aligned_starts[j])
  if anchor:
   scored_anchors.append({'song':name,'line':j+1,'BK':gold[j][1],
     'CTC':c,'error':round(c-gold[j][1],5),
     'ctc_score':source[k].get('ctc_score'),
     'text':gold[j][0]})
 anchor_songs.append({'song':name,'rows':len(gold),'anchors':scored_anchors,
                      'anchor_count':len(scored_anchors)})
 for t in out['intervals']:
  interval_counts[t['status']]+=1
  if t['status']!='SCORED':continue
  internal=[x for x in t['rows'] if (x-1) in keep]
  if internal:coverage['interval_scored']+=1
  interval_rows.append({'song':name,'id':t['id'],'rows':t['rows'],
    'window':t['window'],'LL_gain_per_line':t['LL_gain_per_line'],
    'coverage_H0':t['coverage_H0'],'coverage_H1':t['coverage_H1'],
    'switch':t['switch']})
 for delta in expected:
  key=str(delta)
  clock=out['delta_results'][key]
  if len(clock)!=len(source):raise ValueError('GPU_ROW_COUNT_MISMATCH '+name)
  flagged=out['delta_review'][key]
  songrows=[]
  for j,k in enumerate(keep):
   mainclock=main[j][1]
   ctctime=aligned_starts[j]
   newtime=float(clock[k])
   bk=gold[j][1]
   row={'song':name,'line':j+1,'text':gold[j][0],'BK':bk,
        'H0':ctctime,'H1_selected':newtime,'main':mainclock,
        'moved':abs(newtime-ctctime)>.00001,'delta':delta,
        'review_required':bool(flagged[k]),
        'repeated':distinct[repeat_key(gold[j][0])]>1,
        'anchor':(k+1) in anchors,
        'movement_s':round(newtime-ctctime,5),
        'h0_error':round(ctctime-bk,5),'new_error':round(newtime-bk,5)}
   assert not row['anchor'] or not row['moved']
   songrows.append(row);bucket[key].append(row)
   if row['moved']:all_moved.append(row)
  if name in foci:
   focus_rows.extend(x for x in songrows if delta==.5)
  old=one_score(songrows,'H0');new=one_score(songrows,'H1_selected')
  # Per-song summary added later from global buckets.
  if delta==.5:
   anchor_songs[-1]['H0']=old
for delta in expected:
 key=str(delta);rows=bucket[key]
 assert len(rows)==1487,(delta,len(rows))
 repeater=lambda r,flag:[x for x in rows if x['repeated']==flag]
 active=[t for t in interval_rows if t['switch'].get(key)]
 per_song=[]
 for item in items:
  name=item.get('title',item['id'])
  sub=[r for r in rows if r['song']==name]
  per_song.append({'song':name,'H0':one_score(sub,'H0'),
   'H1':one_score(sub,'H1_selected'),'main':one_score(sub,'main'),
   'new_gt1':[x['line'] for x in sub if abs(x['H0']-x['BK'])<=1 and abs(x['H1_selected']-x['BK'])>1],
   'moved':sum(x['moved'] for x in sub)})
 summary={'delta':delta,'rows':len(rows),'H0':one_score(rows,'H0'),
    'H1':one_score(rows,'H1_selected'),'main':one_score(rows,'main'),
    'new_gt1':[{'song':r['song'],'line':r['line'],'BK':r['BK'],'H0':r['H0'],'H1':r['H1_selected']}
          for r in rows if abs(r['H0']-r['BK'])<=1 and abs(r['H1_selected']-r['BK'])>1],
    'new_backtracks':[],
    'max_move_s':max(abs(r['movement_s']) for r in rows),
    'active_intervals':len(active),
    'active_interval_worse':[t for t in active if
         sum(abs(z['H1_selected']-z['BK'])-abs(z['H0']-z['BK'])
             for z in rows if z['song']==t['song'] and z['line'] in t['rows'])>.2],
    'repeat':one_score(repeater(rows,True),'H1_selected'),
    'nonrepeat':one_score(repeater(rows,False),'H1_selected'),
    'songs':per_song,
    'moved_rows':sum(r['moved'] for r in rows)}
 for name in {r['song'] for r in rows}:
  sub=[r for r in rows if r['song']==name]
  for a,b in zip(sub,sub[1:]):
   if a['H1_selected']>b['H1_selected']+1e-8 and not a['H0']>b['H0']+1e-8:
    summary['new_backtracks'].append({'song':name,'line':a['line']})
 bucket[key]=summary
anchors=[a for s in anchor_songs for a in s['anchors']]
danger=[a for a in anchors if abs(a['error'])>1.]
base_wrong=[r for r in focus_rows] # Focus is the .5 reference set
anchor_numbers={(s['song'],a['line']) for s in anchor_songs for a in s['anchors']}
# Classify main CTC wrong lines in anchor gaps vs in anchors.
ctc_rows=bucket[str(.5)]['songs']
b11=json.loads((V/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
ctc_gt1=[r for r in b11['rows'] if r['errors']['CTC_s']>1]
dangerous={(r['song'],r['line']) for r in danger}
out={'status':'ANCHOR_39_POSTHOC_SCORED','count':39,'rows':1487,
 'anchor_total':len(anchors),'anchor_PCO200':sum(abs(a['error'])<=.2+1e-9 for a in anchors),
 'anchor_PCO500':sum(abs(a['error'])<=.5+1e-9 for a in anchors),
 'anchor_gt1':danger,'anchor_accuracy_fatal':bool(danger),
 'ctc_wrong_gt1_total':len(ctc_gt1),
 'ctc_wrong_gt1_anchor':sum((r['song'],r['line']) in anchor_numbers for r in ctc_gt1),
 'ctc_wrong_gt1_inside_gaps':sum((r['song'],r['line']) not in anchor_numbers for r in ctc_gt1),
 'interval_statuses':dict(interval_counts),'interval_counts':dict(coverage),
 'songs':anchor_songs,'delta':bucket,'focus_lines':focus_rows,
 'moved_lines':all_moved,'intervals':interval_rows,
 'evaluation_policy':'PCO@0.2 uses <=.2+1e-9; BK only read after all 39 hypotheses frozen',
 'dev_set_only':True}
out_path=V/'LRC13_ANCHOR_POSTHOC_20261011.json'
out_path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('ANCHOR_POSTHOC_DONE',json.dumps({'anchors':len(anchors),
 'fatal':len(danger),'H0':bucket[str(.5)]['H0'],
 'delta':{k:{'score':v['H1'],'active':v['active_intervals'],
             'new_gt1':len(v['new_gt1']),'new_backtracks':len(v['new_backtracks'])}
          for k,v in bucket.items()}},ensure_ascii=True),flush=True)
