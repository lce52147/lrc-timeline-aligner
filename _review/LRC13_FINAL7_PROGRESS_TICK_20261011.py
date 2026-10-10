from pathlib import Path
import time,json
time.sleep(16)
v=Path(__file__).resolve().parent
m=v/'LRC13_OPTION_A_FINAL7_FULL_CENTRAL_20261010'/'FINAL7_SERIAL_MANIFEST.json'
d=json.loads(m.read_text(encoding='utf8')) if m.exists() else {'cases':[]}
cases=d['cases'];n=len(cases)
print('FINAL7',n,'/14',d.get('status','STARTING'))
for x in cases[-3:]:
    print(x.get('title'),x.get('mode'),x.get('status'),
          'seconds',x.get('seconds'),
          'bk_gt1',(x.get('posthoc_BK') or {}).get('wrong_gt1'),
          'pco',(x.get('posthoc_BK') or {}).get('PCO200'))
if n<14:
    names=['04 影色舞','09.夢遊','01 My Dearest','03. Choir ‘S’ Choir',
           '01.ミレニアの水槽','10.方舟','01.感情グラス']
    title=names[n//2];mode=('segment','ctc')[n%2]
    log=v/'LRC13_OPTION_A_FINAL7_FULL_CENTRAL_20261010'/f'{n//2}_{mode}'/'full_production.log'
    if log.is_file():
        entries=log.read_text(encoding='utf8',errors='replace').splitlines()
        print('CURRENT',title,mode,'log_lines',len(entries),'last',entries[-1][-260:] if entries else '')
