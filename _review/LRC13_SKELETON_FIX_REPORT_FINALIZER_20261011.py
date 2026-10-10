"""Fail-closed final-stage report materializer; no GPU calls or model inference."""
from pathlib import Path
import sys,time,json,subprocess
V=Path(__file__).resolve().parent
ROOT=V.parent
STATUS=V/'LRC13_WORK_STATUS_20261011.json'
FINISHED=V/'LRC13_SKELETON_FIX_REPORT_20261011.json'
def read():
    if not STATUS.is_file():return {}
    try:return json.loads(STATUS.read_text(encoding='utf8'))
    except (ValueError,OSError):return {}
begin=time.monotonic()
last=''
while time.monotonic()-begin<172800:
    obj=read()
    phase=str(obj.get('phase',''))
    if phase!=last:
        print('REPORT_FINALIZER',phase,flush=True)
        last=phase
    if phase.startswith('BLOCKED_'):
        print('REPORT_BLOCKED',str(obj.get('detail',''))[:300],flush=True)
        break
    if phase=='FINAL_A7_SCORED':
        print('REPORT_RENDER_BEGIN',flush=True)
        r=subprocess.run([sys.executable,str(V/'LRC13_SKELETON_FIX_REPORT_BUILDER_20261011.py')],
                         cwd=str(ROOT))
        if r.returncode:
            print('REPORT_RENDER_FAILED',r.returncode,flush=True)
            raise SystemExit(r.returncode)
        data=json.loads(FINISHED.read_text(encoding='utf8'))
        if data['status']!='FINAL_ARTIFACTS_SCORED':
            raise SystemExit('REPORT_NOT_COMPLETE')
        files=[
            '_review/LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.json',
            '_review/LRC13_SKELETON_FIX_FINAL_A7_REPORT_20261011.json',
            '_review/LRC13_SKELETON_FIX_FINAL_A7_REPORT_20261011.md',
            '_review/LRC13_SKELETON_FIX_FINAL_A7_LISTEN_FLAGS_20261011.md',
            '_review/LRC13_SKELETON_FIX_REPORT_20261011.json',
            '_review/LRC13_SKELETON_FIX_REPORT_20261011.md',
            '_review/LRC13_SKELETON_FIX_REPORT_BUILDER_20261011.py',
            '_review/LRC13_SKELETON_FIX_REPORT_FINALIZER_20261011.py',
            '_review/LRC13_SKELETON_FIX_CHAIN_20261011.py',
            '_review/LRC13_SKELETON_FIX_FINAL_A7_SCORE_20261011.py',
            '_review/LRC13_SKELETON_FIX_FINAL7_SCORE_20261011.py',
        ]
        run=subprocess.run(['git','add','-f','--',*files],cwd=str(ROOT))
        if run.returncode:raise SystemExit('STAGING_FAILED')
        done=subprocess.run(['git','commit','-m','test(lrc13): archive final 7-song A/B and 39-song A7 release evidence'],cwd=str(ROOT))
        if done.returncode:raise SystemExit('FINAL_COMMIT_FAILED')
        print('FINAL_REPORT_COMMITTED','PASS' if data['release_pass'] else 'NOT_PASS',
              flush=True)
        break
    time.sleep(10)
else:
    print('REPORT_FINALIZER_DEADLINE_WITHOUT_COMPLETION',flush=True)
    raise SystemExit(3)
