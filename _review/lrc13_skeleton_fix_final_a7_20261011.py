"""Option A A7 guarded 39-song serial full inference, resume-safe.

Only runs after A6 paired Central gate says PASS_A6_REPRESENTATIVE. Never
uses BK timestamps as a model prior. Always outputs under _review. Never
touches main/checked music/BK, never invokes external source systems.
"""
from pathlib import Path
import sys,os,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
gate_path=V/'LRC13_ASR_CORR_39_SCORE_20261011.json'
if not gate_path.is_file():raise SystemExit('FINAL_A7_BLOCKED_NO_A6_GATE')
gate=json.loads(gate_path.read_text(encoding='utf8'))
if gate.get('experimental_asr_correlated_hard_gate_pass')!=True:
 raise SystemExit('A7_BLOCKED_A6_NOT_PASS: '+str(gate.get('status')))
ab_path=V/'LRC13_SKELETON_FIX_FINAL_7_AB_20261011'/'FINAL7_SERIAL_MANIFEST.json'
if not ab_path.is_file():raise SystemExit('A7_BLOCKED_FULL_7_NOT_RUN')
ab=json.loads(ab_path.read_text(encoding='utf8'))
if len(ab.get('cases',[]))!=14 or any(x.get('status')!='GENERATED' for x in ab.get('cases',[])):
 raise SystemExit('A7_BLOCKED_FULL_7_NOT_ALL_GENERATED')
corpus=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))
assert len(corpus['corpus'])==39 and corpus['summary']['paths_complete']==39
out=V/'LRC13_SKELETON_FIX_FINAL_A7_39_20261011'
out.mkdir(exist_ok=True)
manifest_path=out/'RUN_MANIFEST.json'
existing=(json.loads(manifest_path.read_text(encoding='utf8'))['runs']
          if manifest_path.exists() else [])
known={x['id'] for x in existing if x.get('status')=='PRODUCED'}
for number,item in enumerate(corpus['corpus'],1):
 if item['id'] in known:continue
 target=out/f'{number:02d}_{item["id"]}'
 target.mkdir(exist_ok=True)
 generated=target/'candidate.lrc'
 if generated.exists():raise RuntimeError('REFUSE_UNVERIFIED_EXISTING_CANDIDATE')
 report_dir=target/'reports';log=target/'production.log'
 cmd=[sys.executable,str(ROOT/'scripts/auto_lrc.py'),item['audio_path'],
  '--lyrics',item['human_BK'],'--timing-source','auto',
  '--ignore-lyric-timestamps','--no-checked-lrc-hint',
  '--no-anchor-hints','--skeleton','segment',
  '--output',str(generated),'--report-dir',str(report_dir),
  '--arbiter','reviewer-validity']
 from run_demucs_37_regression_20261008 import build
 job=next((x for x in build() if x['title']==item.get('title',item['id'])),None)
 if job is not None:cmd+=['--reviewer-evidence',job['sidecar']]
 env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf8',
      'LRC_EVIDENCE_WORKERS':'1'}
 t=time.monotonic()
 print('A7_BEGIN',number,'/39',item['title'].encode('unicode_escape').decode(),flush=True)
 try:
  with log.open('wb') as f:
   process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,
                          stderr=subprocess.STDOUT,timeout=1500)
  state='PRODUCED' if process.returncode==0 and generated.is_file() else 'FAILED'
  record={'id':item['id'],'title':item.get('title',item['id']),'status':state,
      'exit':process.returncode}
 except subprocess.TimeoutExpired:
  record={'id':item['id'],'title':item['title'],'status':'TIMED_OUT'}
 record.update({'elapsed_s':round(time.monotonic()-t,2),
    'candidate':str(generated),'log':str(log),
    'human_timestamps_inference':False})
 existing.append(record)
 manifest_path.write_text(json.dumps(
   {'status':'IN_PROGRESS','runs':existing,'BK_clocks_used_inference':False},
    ensure_ascii=False,indent=2),encoding='utf8')
 print('A7_DONE',number,record['status'],record['elapsed_s'],flush=True)
manifest_path.write_text(json.dumps(
   {'status':'COMPLETE_39' if len(existing)==39 and all(
     x['status']=='PRODUCED' for x in existing) else 'INCOMPLETE',
    'runs':existing,'BK_clocks_used_inference':False},
    ensure_ascii=False,indent=2),encoding='utf8')