from pathlib import Path
import json,sys,statistics
v=Path(__file__).resolve().parent
sys.path.insert(0,str(v))
from LRC13_A7_FINALIZE_20261010 import report,aligned_trust_assignments,parse
items=json.loads((v/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
man=json.loads((v/'LRC13_P1_SKELETON_39_20261011'/'RUN_MANIFEST.json').read_text(encoding='utf8'))
pin=v/'lrc13_option_a_A7_main_9069dbc8_20261010'
rows=[]
for idx,(item,run) in enumerate(zip(items,man['runs']),1):
    if run['status']!='SEGMENT_CURRENT':continue
    b=report(pin/f'{idx:02d}_{item["id"]}'/'reports')
    l=parse(Path(item['human_BK']),Path(item['audio_path']))
    if len(l)==item['BK_sung_rows']+1 and l[0][1]==0:l=l[1:]
    a=aligned_trust_assignments(b,pin/f'{idx:02d}_{item["id"]}'/'main.lrc',Path(item['audio_path']),len(l),run['title'],'ctc')
    scores=[float(r['ctc_score']) for r in a]
    row={'song':run['title'],'rows':len(scores),'mean':round(statistics.mean(scores),4),
         'median':round(statistics.median(scores),4),
         'count_ge_008':sum(x>=.08 for x in scores),
         'count_ge_003':sum(x>=.03 for x in scores),
         'count_lt_0005':sum(x<.005 for x in scores)}
    rows.append(row)
    print(row,flush=True)
(v/'LRC13_P1_GLOBAL_SCORE_AUDIT_20261011.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
