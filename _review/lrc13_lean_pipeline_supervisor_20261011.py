"""N1/N2/N3 existing-task supervisor, posthoc scoring, and local report commit.
Observes running serial GPU N2 and CPU exact Viterbi N3; never starts inference.
No time-based stopping or resource control. No push/merge.
"""
from pathlib import Path
import json,subprocess,sys,time,traceback
V=Path(__file__).resolve().parent;ROOT=V.parent
STATUS=V/'LRC13_LEAN_SUPERVISOR_STATUS_20261011.json'
FINAL=V/'LRC13_WORK_STATUS_20261011.json'
N2=V/'LRC13_LEAN_N2_CTC_FIRST_39_20261011/MANIFEST.json'
N3=V/'LRC13_LEAN_N3_CONSTRAINED_20261011/MANIFEST.json'
def read(path):
 try:return json.loads(path.read_text(encoding='utf8'))
 except (OSError,ValueError):return {}
def write(phase,**props):
 meta={'phase':phase,'N2':read(N2).get('count',0),
       'N3':read(N3).get('completed',0),
       'script_only_plus_report_commit':True,'no_push':True,'no_merge':True,**props}
 STATUS.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
 print('LEAN_SUPERVISOR',phase,'N2',meta['N2'],'N3',meta['N3'],flush=True)
def call(script):
 log=V/(script.replace('.py','.log'))
 with log.open('wb') as f:
  code=subprocess.run([sys.executable,str(V/script)],cwd=ROOT,
                      stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError(f'POSTHOC_FAILED {script} exit={code}; see {log}')
 print('LEAN_POSTHOC_PASS',script,flush=True)
try:
 n2done=n3done=False
 while True:
  n2=read(N2)
  n3=read(N3)
  if n2.get('phase')=='N2_GPU_BLOCKED' or any(x.get('status')=='FAILED' for x in n2.get('cases',[])):
   write('N2_BLOCKED',failed=[x['index'] for x in n2.get('cases',[]) if x['status']=='FAILED'])
   raise SystemExit(4)
  if n2.get('phase')=='N2_39_FROZEN' and not n2done:
   write('SCORING_N2')
   call('lrc13_lean_n2_score_20261011.py')
   n2done=True
  if n3.get('status')=='N3_EXACT_39_FROZEN' and not n3done:
   write('SCORING_N3')
   call('lrc13_lean_n3_score_20261011.py')
   n3done=True
  if n2done and n3done:break
  time.sleep(10)
 write('REPORT_BUILDING')
 call('lrc13_lean_pipeline_report_20261011.py')
 output=V/'LRC13_LEAN_PIPELINE_REPORT_20261011.json'
 report=json.loads(output.read_text(encoding='utf8'))
 names=[
  '_review/lrc13_lean_n1_offline_20261011.py',
  '_review/lrc13_lean_n3_emissions_20261011.py',
  '_review/lrc13_lean_n3_constrained_20261011.py',
  '_review/lrc13_lean_n3_unit_20261011.py',
  '_review/lrc13_lean_n3_score_20261011.py',
  '_review/lrc13_lean_n2_full39_20261011.py',
  '_review/lrc13_lean_n2_score_20261011.py',
  '_review/lrc13_lean_pipeline_report_20261011.py',
  '_review/LRC13_LEAN_N1_FLAGS_20261011.json',
  '_review/LRC13_LEAN_N1_COST_20261011.json',
  '_review/LRC13_LEAN_N2_SCORE_20261011.json',
  '_review/LRC13_LEAN_N3_SCORE_20261011.json',
  '_review/LRC13_LEAN_PIPELINE_REPORT_20261011.json',
  '_review/LRC13_LEAN_PIPELINE_REPORT_20261011.md'
 ]
 if subprocess.run(['git','add','-f','--',*names],cwd=ROOT).returncode:
  raise RuntimeError('LOCAL_GIT_STAGE_FAILED')
 res=subprocess.run(['git','commit','-m','report(lrc13): freeze N1 N2 N3 lean 39-song developer evaluation'],
    cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
 if res.returncode:raise RuntimeError('LOCAL_GIT_COMMIT_FAILED '+res.stderr[-500:])
 sha=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()
 result={'phase':'LEAN_N1_N2_N3_FINAL_SCORED','N2_done':39,'N3_done':39,
   'B11_reference_ctc':{'pco':1285,'gt1':115},
   'summary':report.get('summary'),'decisions':report.get('decisions'),
   'report':'_review/LRC13_LEAN_PIPELINE_REPORT_20261011.md',
   'commit':sha,'no_push':True,'no_merge':True}
 FINAL.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 write('COMPLETE',sha=sha,decisions=report['decisions'])
except BaseException:
 (V/'LRC13_LEAN_SUPERVISOR_ERROR_20261011.txt').write_text(traceback.format_exc(),encoding='utf8')
 raise
