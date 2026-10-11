"""Posthoc gap coverage / repeated lyrics bounded by two REAL acoustic anchors."""
from pathlib import Path
from collections import Counter
import json,statistics,sys
v=Path(__file__).resolve().parent
sys.path.insert(0,str(v.parent/'scripts'))
import auto_lrc as al
b=json.loads((v/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
g=json.loads((v/'LRC13_ANCHOR_POSTHOC_20261011.json').read_text(encoding='utf8'))
f=v/'LRC13_ANCHOR_GPU_V2_20261011'
lengths=[];sizes=[];songs=[];rep_total=rep_both=rep_virtual=0
for i,q in enumerate(g['songs'],1):
 rows=[r for r in b['rows'] if r['song']==q['song']]
 anchors={a['line'] for a in q['anchors']}
 counts=Counter(al.normalize_match_text(z['text']) for z in rows)
 repeated=[z for z in rows if counts[al.normalize_match_text(z['text'])]>1]
 both=[z for z in repeated if any(j<z['line'] for j in anchors)
  and any(j>z['line'] for j in anchors)]
 virtual=[z for z in repeated if z['line'] not in anchors]
 rep_total+=len(repeated)
 rep_both+=len(both)
 rep_virtual+=len(virtual)
 leaf=f'{i:02d}_{json.loads((v/"LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json").read_text(encoding="utf8"))["corpus"][i-1]["id"]}'
 file=json.loads((f/leaf/'FROZEN_HYPOTHESES.json').read_text(encoding='utf8'))
 windows=[z for z in file['intervals'] if z.get('rows')]
 for w in windows:
  sizes.append(len(w['rows']))
  if w.get('window'):
   lengths.append(w['window'][1]-w['window'][0])
 songs.append({'song':q['song'],'sung_rows':len(rows),'anchors':len(anchors),
  'repeated_rows':len(repeated),'repeated_with_both_real_anchors':len(both),
  'repeated_nonanchor_with_virtual_edges':len(virtual),
  'intervals_with_internal_rows':len(windows)})
def describe(x):
 z=sorted(x)
 return {'n':len(z),'min':round(z[0],3),'median':round(statistics.median(z),3),
    'p90':round(z[min(len(z)-1,int(round(.9*(len(z)-1))))],3),
    'max':round(z[-1],3)} if z else None
out={'status':'GAPS_SCORED_39','repeated_rows':rep_total,
 'repeat_bounded_two_real_anchors':rep_both,
 'repeat_nonanchor_with_virtual_edges':rep_virtual,
 'interval_seconds':describe(lengths),'interval_row_count':describe(sizes),
 'songs':songs}
(v/'LRC13_ANCHOR_GAP_STATS_20261011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k!='songs'},ensure_ascii=False))
