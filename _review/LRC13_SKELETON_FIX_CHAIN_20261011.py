"""Local serial verifier: await 7x2 full Central, gate, run final 39 A7.

May not push/merge or touch original BK/audio. Result paths are isolated _review.
Only fixed frozen code from lrc-1.3; zero parallel GPU production runs.
"""
from pathlib import Path
import json,sys,time,subprocess,traceback
v=Path(__file__).resolve().parent
root=v.parent
ab=v/'LRC13_OPTION_A_FINAL7_FULL_CENTRAL_20261010'/'FINAL7_SERIAL_MANIFEST.json'
a7=v/'LRC13_SKELETON_FIX_FINAL_A7_39_20261011'/'RUN_MANIFEST.json'
status=v/'LRC13_WORK_STATUS_20261011.json'
score7=v/'LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.json'
score39=v/'LRC13_SKELETON_FIX_FINAL_A7_REPORT_20261011.json'
def load(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else {}
def publish(phase,note,**kwargs):
    payload={'phase':phase,'detail':note,'source':'LRC13_SKELETON_FIX_CHAIN_20261011.py',
             'branch':'lrc-1.3','frozen_code':'48f8513d',
             'quick_gate':'PASS_39_SKELETON_LAYER_2_ACTIVE_0_WORSE',
             'release_gate':'PENDING','no_push':True,'no_merge':True,**kwargs}
    status.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print('CHAIN',phase,note,flush=True)
publish('FULL7_RUNNING','Waiting for original serial 7x2 Central CLI process')
while True:
    data=load(ab)
    cases=data.get('cases',[])
    produced=sum(x.get('status')=='GENERATED' for x in cases)
    if cases and (len(cases)!=getattr(publish,'last',-1)):
        publish.last=len(cases)
        publish('FULL7_RUNNING',f'{produced}/14 modes generated',
                full7_count=len(cases),full7_produced=produced,
                last=cases[-1].get('title','')+' '+cases[-1].get('mode',''))
    if len(cases)==14:
        if any(x.get('status')!='GENERATED' for x in cases):
            publish('BLOCKED_FULL7_CASE_FAILED','Not all full Central cases generated',
                    full7_cases=cases)
            raise SystemExit(2)
        break
    time.sleep(5)
publish('FULL7_POSTHOC','Scoring all 14 frozen complete-Central outputs')
with (v/'LRC13_SKELETON_FIX_CHAIN_20261011.log').open('a',encoding='utf8') as log:
    r=subprocess.run([sys.executable,str(v/'LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.py')],
                     cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
if r.returncode or load(score7).get('gate')!='PASS_FINAL7_AB':
    publish('BLOCKED_FULL7_NOT_PASS','7-song A/B scoring did not pass, final A7 not started',
            full7=load(score7))
    raise SystemExit(3)
publish('FULL7_PASS','14/14 full Central cases gate passed; proceeding to serial final 39')
with (v/'LRC13_SKELETON_FIX_FINAL_A7_WORKER_20261011.log').open('a',encoding='utf8') as log:
    task=subprocess.Popen([sys.executable,str(v/'lrc13_skeleton_fix_final_a7_20261011.py')],
                          cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
last=-1
while task.poll() is None:
    m=load(a7)
    count=len(m.get('runs',[]))
    if count!=last:
        last=count
        publish('FINAL_A7_RUNNING',f'{count}/39 full Central songs produced',
                final_a7_count=count,final_a7_worker_pid=task.pid,
                last=m.get('runs',[])[-1].get('title','') if count else '')
    time.sleep(5)
m=load(a7)
if task.returncode or m.get('status')!='COMPLETE_39':
    publish('BLOCKED_A7_WORKER_FAILED','Final A7 runner failed or incomplete',
            worker_exit=task.returncode,manifest=m)
    raise SystemExit(4)
publish('FINAL_A7_POSTHOC','39/39 complete; scoring frozen final A7')
with (v/'LRC13_SKELETON_FIX_CHAIN_20261011.log').open('a',encoding='utf8') as log:
    r=subprocess.run([sys.executable,str(v/'LRC13_SKELETON_FIX_FINAL_A7_SCORE_20261011.py')],
                     cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
final=load(score39)
if r.returncode or final.get('status')!='A7_SCORED_39':
    publish('BLOCKED_A7_SCORING_FAILED','Final A7 scorer failed or incomplete',
            scorer_exit=r.returncode,scorer_status=final.get('status'))
    raise SystemExit(5)
publish('FINAL_A7_SCORED','All final evaluation artifacts generated',
        release_pass=final.get('release_pass'),release_gate=('PASS' if final.get('release_pass') else 'NOT_PASS'),
        final_a7_active=final.get('active_count'),final_a7_fallback=final.get('fallback_count'),
        final_a7_new_gt1=len(final.get('new_gt1_lines',[])))
