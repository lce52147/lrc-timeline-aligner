"""One-song FULL Central smoke for generic unresolved-evidence fallback; 7/A7 frozen results untouched."""
from pathlib import Path
import sys,subprocess,os,time,json
v=Path(__file__).resolve().parent
root=v.parent
sys.path[:0]=[str(root/'scripts'),str(v)]
from run_demucs_37_regression_20261008 import build
from LRC13_A7_FINALIZE_20261010 import parse,report
jobs={j['title']:j for j in build()}
job=jobs['04 影色舞']
out=v/'LRC13_FINAL7_PEER_OUTLIER_20261011'
out.mkdir(exist_ok=True)
target=out/'segment.lrc'
logs=out/'production.log'
assert not target.exists(),'REFUSE_OVERWRITE_OLD_SMOKE'
cmd=[sys.executable,str(root/'scripts'/'auto_lrc.py'),job['audio'],
     '--lyrics',job['lyrics'],'--ignore-lyric-timestamps',
     '--timing-source','auto','--skeleton','segment','--output',str(target),
     '--no-checked-lrc-hint','--no-anchor-hints',
     '--report-dir',str(out/'reports'),'--arbiter','reviewer-validity']
if job.get('sidecar'):cmd.extend(('--reviewer-evidence',job['sidecar']))
env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf8','LRC_EVIDENCE_WORKERS':'1'}
print('FULL_CENTRAL_PEER_OUTLIER_BEGIN',flush=True)
t=time.monotonic()
with logs.open('wb') as f:
    run=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1500)
if run.returncode:raise SystemExit('FULL_RUN_FAILED '+str(run.returncode))
ref=v/'LRC13_OPTION_A_FINAL7_FULL_CENTRAL_20261010'
old=ref/'0_ctc'/'new.lrc'
prior=ref/'0_segment'/'new.lrc'
gold=parse(Path(job['lyrics']),Path(job['audio']))
now=parse(target,Path(job['audio']))
baseline=parse(old,Path(job['audio']))
previous=parse(prior,Path(job['audio']))
if len(gold)==len(now)+1 and gold[0][1]==0:gold=gold[1:]
assert len(gold)==len(now)==len(baseline)==len(previous)
def score(seq):
    return {'pco200':sum(abs(b[1]-x[1])<=.2 for b,x in zip(gold,seq)),
            'wrong_gt1':sum(abs(b[1]-x[1])>1 for b,x in zip(gold,seq)),
            'new_gt1':[
                i for i,(b,x,o) in enumerate(zip(gold,seq,baseline),1)
                if abs(b[1]-o[1])<=1 and abs(b[1]-x[1])>1]}
d=report(out/'reports')
sk=d.get('segment_skeleton_status') or {}
result={'scope':'Full Central rerun with generic peer-fusion outlier admission',
 'code':'working-tree uncommitted 20261011','seconds':round(time.monotonic()-t,2),
 'segment_status':sk.get('status'),'old_seg':score(previous),
 'ctc':score(baseline),'new_seg':score(now),
 'row10':{'bk':gold[9][1],'ctc':baseline[9][1],
          'previous_segment':previous[9][1],'new_segment':now[9][1],
          'selected_candidate_source':d['assignments'][9].get('candidate_source'),
          'trusted':d['assignments'][9].get('timing_trusted')},
 'changed_rows_vs_old_seg':[i for i,(x,y) in enumerate(zip(now,previous),1) if abs(x[1]-y[1])>.01]}
(out/'POSTHOC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('FULL_CENTRAL_PEER_OUTLIER_RESULT',json.dumps(result,ensure_ascii=True),flush=True)