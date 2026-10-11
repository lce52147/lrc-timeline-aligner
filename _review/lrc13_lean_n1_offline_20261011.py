"""N1: BK-blind 6 fixed flags, assessed only against already frozen 39-song times.
Flags: CTC score <.08, repeated lyric, adjacent compression <.35 s/syl,
CTC low-score run member, F5 repeated skeleton-global diff >3 s,
and all rows of song with any compression. No flags use BK.
"""
from pathlib import Path
from collections import Counter,defaultdict
import re,json,sys
V=Path(__file__).resolve().parent;ROOT=V.parent
sys.path[:0]=[str(V),str(ROOT/'scripts')]
from LRC13_PATCH_A7_SCORE_20261011 import report,aligned_trust_assignments
import auto_lrc as al
SCORE=.08;SPEED=.35;F5_DELTA=3.
R=re.compile('[aeiouy]+',re.I)
b=json.loads((V/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
items=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
B=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
STAGE=V/'LRC13_ASR_CORR_39_20261011'
new=V/'LRC13_FINAL_A7_PATCHED_39_20261011'
out=[];all_rows=[]
for i,item in enumerate(items,1):
 leaf=f'{i:02d}_{item["id"]}'
 name=item.get('title',item['id'])
 rows=[x for x in b['rows'] if x['song']==name]
 stage=json.loads((STAGE/leaf/'stage.json').read_text(encoding='utf8'))
 assert len(rows)==len(stage['skeleton_times']),name
 ct=[z['CTC_s'] for z in rows]
 sk=stage['skeleton_times']
 a=aligned_trust_assignments(report(B/leaf/'reports'),B/leaf/'main.lrc',item['audio_path'],len(rows),name,'baseline')
 lines=[z['text'] for z in rows]
 cnt=Counter(al.normalize_match_text(z) for z in lines)
 compressed=set()
 for j in range(len(rows)-1):
  syll=max(1,len(R.findall(lines[j])))
  if (ct[j+1]-ct[j])/syll<SPEED:compressed.update((j,j+1))
 low_run=set()
 fullreport=report(B/leaf/'reports')
 for group in fullreport.get('ctc_low_score_runs',[]):
  if isinstance(group,list):
   low_run.update((int(k)-1) for k in group if isinstance(k,int))
 # Historic row excluded by aligned_trust_assignments may offset run indices;
 # map report native row index to scored line when a legacy provenance row exists.
 if len(fullreport.get('assignments',[]))!=len(rows):
  extra=len(fullreport['assignments'])-len(rows)
  low_run={k-extra for k in low_run if k>=extra}
 whole=bool(compressed)
 for j,(r,assignment) in enumerate(zip(rows,a)):
  flags={
   'CTC_score_lt_008':float(assignment.get('ctc_score') or 0)<SCORE,
   'repeated_lyric':cnt[al.normalize_match_text(lines[j])]>1,
   'compression_pair':j in compressed,
   'CTC_low_run':j in low_run,
   'F5_repeat_global_diff':stage['status']=='SEGMENT_CURRENT' and cnt[al.normalize_match_text(lines[j])]>1 and abs(sk[j]-ct[j])>F5_DELTA,
   'whole_song_speed_flag':whole
  }
  out.append({'song':name,'line':j+1,'text':r['text'],'ctc_s':ct[j],
              'skeleton_s':sk[j],'lrc13_s':r['lrc13_s'],
              'flags':flags,'flagged':any(flags.values())})
 assert len(rows)==len(a)
 all_rows.extend(rows)
assert len(out)==1487
# Freeze flags against times: this JSON has no BK values; scorer separates truth lookup.
frozen={'status':'N1_FLAGS_BK_BLIND_FROZEN_39','thresholds':{'ctc_score':SCORE,'speed':SPEED,'F5_diff':F5_DELTA},
 'rows':out,'total':len(out)}
(V/'LRC13_LEAN_N1_FLAGS_20261011.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf8')
print('FLAGS_FROZEN',len(out),'listen',sum(r['flagged'] for r in out),'song_flagged',len({r['song'] for r in out if r['flags']['whole_song_speed_flag']}))
# BK post-hoc: compare only after frozen flags written.
reference={(z['song'],z['line']):z for z in b['rows']}
stats={}
for label,key in [('skeleton','skeleton_s'),('full_lrc13','lrc13_s')]:
 records=[]
 for r in out:
  gold=reference[r['song'],r['line']]['BK_s']
  e=abs(r[key]-gold)
  records.append({'song':r['song'],'line':r['line'],'err':round(e,5),
   'flagged':r['flagged'],'flags':r['flags'],'source_time':r[key], 'BK':gold})
 silent1=[x for x in records if not x['flagged'] and x['err']>1]
 silent05=[x for x in records if not x['flagged'] and x['err']>.5]
 per={}
 for k in out[0]['flags']:
  flagged={i for i,r in enumerate(out) if r['flags'][k]}
  per[k]={'flagged_count':len(flagged),'recall_wrong_gt1':sum(i in flagged for i,z in enumerate(records) if z['err']>1),
          'recall_wrong_gt05':sum(i in flagged for i,z in enumerate(records) if z['err']>.5)}
 stats[label]={'wrong_gt1':sum(x['err']>1 for x in records),
  'wrong_gt05':sum(x['err']>.5 for x in records),'flagged':sum(x['flagged'] for x in records),
  'listen_ratio':sum(x['flagged'] for x in records)/len(records),
  'silent_wrong_gt1':silent1,'silent_wrong_gt05':silent05,
  'per_flag':per}
stats['full_1_3_original_flag_baseline']={'wrong_gt1':84,'flagged_review':583,'untrusted':548,'flag_caught_gt1':80,
 'note':'Original full-pipeline F4 etc; compare separately from N1 fixed flag definition'}
(V/'LRC13_LEAN_N1_COST_20261011.json').write_text(json.dumps({'status':'N1_POSTHOC_39','stats':stats},ensure_ascii=False,indent=2),encoding='utf8')
for label,x in stats.items():
 if label=='full_1_3_original_flag_baseline':continue
 print(label,{k:x[k] if k!='silent_wrong_gt1' else len(x[k]) for k in ('wrong_gt1','wrong_gt05','flagged','listen_ratio','silent_wrong_gt1')},
       'silent_05',len(x['silent_wrong_gt05']))
