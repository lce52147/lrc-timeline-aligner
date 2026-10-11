"""Serial continuation after full-Central smoke. No BK timing enters production.

Automatically stops on any actual hard-gate failure; no gate JSON mutation.
It does not write to BK/audio/main and never starts multiple GPU production jobs.
"""
from pathlib import Path
import json,sys,subprocess,time,traceback
v=Path(__file__).resolve().parent
root=v.parent
status=v/'LRC13_PEER_OUTLIER_VALIDATION_STATUS_20261011.json'
smoke=v/'LRC13_FINAL7_PEER_OUTLIER_20261011'/'POSTHOC.json'
ab=v/'LRC13_PATCH7_PROVISIONAL_RERUN_20261011'/'PATCH7_SERIAL_MANIFEST.json'
ab_score=v/'LRC13_PATCH7_FULL_SCORE_20261011.json'
a7=v/'LRC13_FINAL_A7_PATCHED_39_20261011'/'RUN_MANIFEST.json'
a7_score=v/'LRC13_FINAL_A7_PATCHED_REPORT_20261011.json'
def load(p):
    return json.loads(p.read_text(encoding='utf8')) if p.is_file() else {}
def mark(phase,**kw):
    value={'phase':phase,'quick_skeleton_gate':'PASS_39_2_ACTIVE_0_WORSE',
        'old_full7_gate':'FAIL_1_NEW_GT1','branch':'lrc-1.3',
        'release_gate':'NOT_PASS_UNTIL_FINAL_A7', 'no_push':True,'no_merge':True,
        **kw}
    status.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print('PATCH_CHAIN',phase,json.dumps(kw,ensure_ascii=True)[:300],flush=True)
def worker(script,logfile,manifest_path,unit,limit):
    with logfile.open('a',encoding='utf8') as log:
        process=subprocess.Popen([sys.executable,str(v/script)],
            cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+limit
    prev=-1
    while process.poll() is None:
        data=load(manifest_path)
        count=len(data.get('cases',data.get('runs',[])))
        if count!=prev:
            prev=count
            mark(unit+'_RUNNING',count=count,pid=process.pid,
                 last=data.get('cases',data.get('runs',[]))[-1].get('title','')
                 if count else '')
        if time.monotonic()>deadline:
            mark(unit+'_TIMED_OUT',pid=process.pid,count=count)
            raise SystemExit('BOUND_DEADLINE '+unit)
        time.sleep(5)
    if process.returncode:
        mark(unit+'_WORKER_FAILED',exit=process.returncode,manifest=load(manifest_path))
        raise SystemExit('WORKER_FAILED '+unit)
    mark(unit+'_WORKER_DONE',exit=process.returncode)
try:
    mark('WAIT_FULL_CENTRAL_SMOKE')
    deadline=time.monotonic()+1600
    while not smoke.is_file():
        if time.monotonic()>deadline:
            mark('SMOKE_NOT_COMPLETED');raise SystemExit(2)
        time.sleep(5)
    d=load(smoke)
    new=d.get('new_seg') or {}
    row=d.get('row10') or {}
    if (d.get('segment_status')!='SEGMENT_CURRENT'
        or new.get('new_gt1')
        or new.get('pco200',0)<d.get('ctc',{}).get('pco200',0)
        or abs(float(row.get('new_segment',-10))-float(row.get('ctc',100)))>0.5):
        mark('BLOCKED_SMOKE_NOT_PASS',result=d)
        raise SystemExit(3)
    mark('SMOKE_PASS',row10=row,new_seg=new)
    worker('lrc13_patch7_full_central_20261011.py',
           v/'LRC13_PATCH7_FULL_WORKER_20261011.log',ab,'PATCH7',10000)
    completed=load(ab)
    if len(completed.get('cases',[]))!=14 or any(
         z.get('status')!='GENERATED' for z in completed['cases']):
        mark('BLOCKED_PATCH7_INCOMPLETE',manifest=completed)
        raise SystemExit(4)
    mark('PATCH7_POSTHOC')
    with (v/'LRC13_PATCH_CHAIN_20261011.log').open('a',encoding='utf8') as f:
        out=subprocess.run([sys.executable,str(v/'LRC13_PATCH7_FULL_SCORE_20261011.py')],
                           cwd=str(root),stdout=f,stderr=subprocess.STDOUT)
    score=load(ab_score)
    if out.returncode or score.get('gate')!='PASS_PATCH7_AB':
        mark('BLOCKED_PATCH7_NOT_PASS',score=score)
        raise SystemExit(5)
    mark('PATCH7_PASS',score=score.get('global'))
    worker('lrc13_patch_a7_39_20261011.py',
           v/'LRC13_PATCHED_A7_WORKER_20261011.log',a7,'PATCHED_A7',30000)
    if load(a7).get('status')!='COMPLETE_39':
        mark('BLOCKED_PATCHED_A7_INCOMPLETE',manifest=load(a7))
        raise SystemExit(6)
    mark('PATCHED_A7_POSTHOC')
    with (v/'LRC13_PATCH_CHAIN_20261011.log').open('a',encoding='utf8') as f:
        out=subprocess.run([sys.executable,str(v/'LRC13_PATCH_A7_SCORE_20261011.py')],
                           cwd=str(root),stdout=f,stderr=subprocess.STDOUT)
    final=load(a7_score)
    if out.returncode or final.get('status')!='A7_SCORED_39':
        mark('BLOCKED_PATCHED_A7_SCORE_ERROR',exit=out.returncode,final=final)
        raise SystemExit(7)
    mark('FINISHED_PATCHED_A7',release_gate='PASS' if final.get('release_pass') else 'NOT_PASS',
        release_pass=final.get('release_pass'),active=final.get('active_count'),
        fallback=final.get('fallback_count'),new_gt1=len(final.get('new_gt1_lines',[])))
except BaseException:
    with (v/'LRC13_PEER_OUTLIER_CHAIN_ERROR_20261011.txt').open('w',encoding='utf8') as f:
        f.write(traceback.format_exc())
    raise