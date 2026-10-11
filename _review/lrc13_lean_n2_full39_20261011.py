"""N2: serial 39-song full CLI removal A/B (opt-in ctc-first).
Reads frozen BK lyrics as TEXT with --ignore-lyric-timestamps; never accesses
BK timestamps for inference. No GPU resource policy or process stop limit.
Reports and scores are separate posthoc tasks.
"""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
V=Path(__file__).resolve().parent;ROOT=V.parent
OUT=V/'LRC13_LEAN_N2_CTC_FIRST_39_20261011';OUT.mkdir(exist_ok=True)
MANIFEST=OUT/'MANIFEST.json';STATUS=V/'LRC13_WORK_STATUS_20261011.json'
corpus=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
existing=json.loads(MANIFEST.read_text(encoding='utf8')).get('cases',[]) if MANIFEST.is_file() else []
done={r['index'] for r in existing if r.get('status')=='PRODUCED'}
from run_demucs_37_regression_20261008 import build
sidecars={x['title']:x['sidecar'] for x in build()}
def emit(phase,i=0,note=None):
 meta={'phase':phase,'count':len(done),'total':39,'cases':existing,
       'policy':'ctc-first','source_BK_timing_used':False,
       'cli_equal_except_policy':True}
 MANIFEST.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
 STATUS.write_text(json.dumps({'phase':phase,'n2_done':len(done),'total':39,
       'song_index':i,'n3_manifest':str(V/'LRC13_LEAN_N3_CONSTRAINED_20261011/MANIFEST.json'),
       'note':note,'no_push':True,'no_merge':True},ensure_ascii=False,indent=2),encoding='utf8')
 print('N2_PROGRESS',phase,'done',len(done),'index',i,'note',note,flush=True)
for k,item in enumerate(corpus,1):
 if k in done:continue
 leaf=f'{k:02d}_{item["id"]}'
 target=OUT/leaf;target.mkdir(exist_ok=True)
 generated=target/'candidate.lrc';reportdir=target/'reports'
 log=target/'production.log'
 if generated.is_file():raise RuntimeError('REFUSE_UNMANIFESTED_CANDIDATE '+str(generated))
 cmd=[sys.executable,str(ROOT/'scripts/auto_lrc.py'),item['audio_path'],
      '--lyrics',item['human_BK'],'--timing-source','auto',
      '--ignore-lyric-timestamps','--no-checked-lrc-hint','--no-anchor-hints',
      '--skeleton','segment',
      '--output',str(generated),'--report-dir',str(reportdir),
      '--arbiter','reviewer-validity','--timing-source-policy','ctc-first']
 side=sidecars.get(item.get('title',item['id']))
 if side is not None:cmd+=['--reviewer-evidence',str(side)]
 env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf8','LRC_EVIDENCE_WORKERS':'1'}
 emit('N2_GPU_RUNNING',k)
 start=time.monotonic()
 with log.open('wb') as f:
  result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 state='PRODUCED' if result.returncode==0 and generated.is_file() else 'FAILED'
 row={'index':k,'id':item['id'],'song':item.get('title',item['id']),
      'status':state,'exit':result.returncode,'seconds':round(time.monotonic()-start,2),
      'candidate':str(generated),'log':str(log),'sidecar':str(side) if side else None}
 existing.append(row)
 if state=='PRODUCED':done.add(k)
 emit('N2_GPU_CASE_DONE' if state=='PRODUCED' else 'N2_GPU_BLOCKED',k,note=state)
 if state!='PRODUCED':raise SystemExit('N2_FAILED '+str(k)+' '+str(log))
emit('N2_39_FROZEN',39)
