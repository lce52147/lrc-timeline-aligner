from pathlib import Path
import json
v=Path(__file__).resolve().parent
p5=v/'LRC13_P5_SKELETON_39_20261011'
p4=v/'LRC13_P4_SKELETON_39_20261011'
a=json.loads((p5/'RUN_MANIFEST.json').read_text(encoding='utf-8'))
b=json.loads((p4/'RUN_MANIFEST.json').read_text(encoding='utf-8'))
assert a['status']==b['status']=='COMPLETE_39'
d=[]
for idx,(x,y) in enumerate(zip(a['runs'],b['runs']),1):
    assert x['id']==y['id']
    leaf=f"{idx:02d}_{x['id']}"
    before=json.loads((p5/leaf/'stage.json').read_text(encoding='utf-8'))
    after=json.loads((p4/leaf/'stage.json').read_text(encoding='utf-8'))
    same=before['status']==after['status'] and before['skeleton_times']==after['skeleton_times']
    d.append({'song':x['title'],'same':same,'changed_lines':sum(s!=t for s,t in zip(before['skeleton_times'],after['skeleton_times']))})
out={'total':len(d),'unchanged':sum(z['same'] for z in d),'details':d}
(v/'LRC13_P4_COMPARE_20261011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('P4_PARITY',out['unchanged'],'/',out['total'])
if out['unchanged']!=39:raise SystemExit(3)
