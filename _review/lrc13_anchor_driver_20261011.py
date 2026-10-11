"""39-song serial CTC-only anchor prototype; posthoc scoring is separate."""
from pathlib import Path
import json,subprocess,sys,os,time,traceback
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'_review'
F=V/'LRC13_ANCHOR_GPU_V2_20261011';F.mkdir(exist_ok=True)
manifest=F/'MANIFEST.json'
status=V/'LRC13_WORK_STATUS_20261011.json'
corpus=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
cases=[x for x in json.loads(manifest.read_text(encoding='utf8'))['cases'] if x.get('status')=='DONE'] if manifest.is_file() else []
done={x['index'] for x in cases if x.get('status')=='DONE'}
def emit(state,**details):
 o={'phase':state,'task':'ANCHOR_ARCH_39_CTC_ONLY','done':sum(z.get('status')=='DONE' for z in cases),'total':39,
 'replaced_S1S3_binary_search':True,'production_scripts_changed':False,
 'BK_timestamps_used':False,'no_push':True,'no_merge':True,**details}
 status.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf8')
 manifest.write_text(json.dumps({'status':state,'cases':cases,'BK_timestamps_used':False},ensure_ascii=False,indent=2),encoding='utf8')
 print('ANCHOR_PROGRESS',state,json.dumps(details,ensure_ascii=True)[:800],flush=True)
for i,item in enumerate(corpus,1):
 if i in done:continue
 leaf=f'{i:02d}_{item["id"]}'
 source=list((V/'lrc13_option_a_A7_main_9069dbc8_20261010'/leaf/'reports').glob('*.align-report.json'))
 assert len(source)==1,(i,'ONE_MAIN_REPORT_REQUIRED')
 out=F/leaf;out.mkdir(exist_ok=True)
 output=out/'FROZEN_HYPOTHESES.json'
 if output.exists():raise RuntimeError('REFUSE_UNRECORDED_OUTPUT '+str(output))
 base=json.loads(source[0].read_text(encoding='utf8'))
 stem=Path(base['ctc_audio_path'])
 assert stem.is_file(),str(stem)
 # Duration from archived CTC stem; never consult BK or final LRC clocks.
 cmdff=['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(stem)]
 duration=float(subprocess.check_output(cmdff,text=True,encoding='utf8').strip())
 req={'meta':{'duration':duration,'id':item['id'],'song':item.get('title',item['id'])},
      'report':{'assignments':base['assignments'],'ctc_audio_path':str(stem)}}
 request_path=out/'REQUEST_BK_BLIND.json'
 request_path.write_text(json.dumps(req,ensure_ascii=False),encoding='utf8')
 cmd=[str(ROOT/'.venv-asr/Scripts/python.exe'),str(V/'lrc13_anchor_engine_20261011.py'),str(request_path),str(output)]
 emit('ANCHOR_GPU_RUNNING',index=i,song=item['id'])
 start=time.monotonic()
 with (out/'production.log').open('wb') as file:
  r=subprocess.run(cmd,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,
      env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf8'})
 rec={'index':i,'id':item['id'],'title':item.get('title',item['id']),
      'command':cmd,'status':'DONE' if r.returncode==0 and output.is_file() else 'FAILED',
      'exit_code':r.returncode,'output':str(output),'seconds':round(time.monotonic()-start,2)}
 cases.append(rec)
 emit('ANCHOR_GPU_CASE_DONE',index=i,result=rec['status'])
 if rec['status']!='DONE':
  emit('ANCHOR_GPU_BLOCKED',index=i,reason='GPU_RUN_FAILED')
  raise SystemExit('ANCHOR_GPU_FAIL_SONG '+str(i))
emit('ANCHOR_GPU_39_FROZEN',count=len(cases))
