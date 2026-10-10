"""One-song fresh Option A P5 acceptance probe; no BK clocks at inference."""
from pathlib import Path
import os,sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
item=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf-8'))['corpus'][37]
assert item['id']=='09.夢遊'
folder=V/'LRC13_P5_FRESH_DREAM_20261011'
folder.mkdir(exist_ok=True)
out=folder/'candidate.lrc'
if out.exists():
    raise RuntimeError('REFUSE_OVERWRITE_FROZEN_P5_DREAM')
cmd=[sys.executable,str(ROOT/'scripts/auto_lrc.py'),item['audio_path'],
 '--lyrics',item['human_BK'],'--timing-source','auto',
 '--ignore-lyric-timestamps','--no-checked-lrc-hint','--no-anchor-hints',
 '--skeleton','segment','--output',str(out),
 '--report-dir',str(folder/'reports'),'--arbiter','reviewer-validity']
sys.path.insert(0,str(V))
from run_demucs_37_regression_20261008 import build
job=next((x for x in build() if x['title']==item.get('title',item['id'])),None)
if job:cmd+=['--reviewer-evidence',job['sidecar']]
env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf8','LRC_EVIDENCE_WORKERS':'1'}
print('P5_DREAM_FRESH_BEGIN',flush=True)
t=time.monotonic()
with (folder/'production.log').open('wb') as f:
    proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1300)
status={'state':'PRODUCED' if proc.returncode==0 and out.is_file() else 'FAILED',
        'exit_code':proc.returncode,'duration_s':round(time.monotonic()-t,2),
        'BK_used_for_inference':False,'P5_commit':'3ba1cbcb'}
(folder/'RUN_MANIFEST.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print('P5_DREAM_FRESH_END',json.dumps(status),flush=True)
if status['state']!='PRODUCED':sys.exit(2)
