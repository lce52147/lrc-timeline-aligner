"""Skeleton fix final: actual 7x2 CLI / Central A-B, serial GPU, posthoc BK only.

Stages all generated lyrics/reports in _review. Never touches BK/music/main.
Partial manifest is rewritten after EACH completed mode so resumable.
"""
from pathlib import Path
# Final full A/B may run only after frozen skeleton-layer safety gate.
GATE=Path(__file__).resolve().parent/'LRC13_ASR_CORR_39_SCORE_20261011.json'
if not GATE.is_file():
    raise SystemExit('FULL_7_BLOCKED_MISSING_ZERO_REGRESSION_GATE')
import json as gate_json
if not gate_json.loads(GATE.read_text(encoding='utf8')).get('experimental_asr_correlated_hard_gate_pass',False):
    raise SystemExit('FULL_7_BLOCKED_UNMET_ZERO_REGRESSION_GATE')
import subprocess,sys,os,json,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'_review')]
from run_demucs_37_regression_20261008 import build
import auto_lrc as a
V=ROOT/'_review/LRC13_PATCH7_PROVISIONAL_RERUN_20261011'
V.mkdir(exist_ok=True)
jobs={q['title']:q for q in build()}
name_list=['04 影色舞','09.夢遊','01 My Dearest',
 '03. Choir ‘S’ Choir','01.ミレニアの水槽','10.方舟','01.感情グラス']
extra=json.loads((ROOT/'_review/lrc13_s1_offline_passages/S1_DP_7_SONG_PRECHECK.json').read_text(encoding='utf8'))['songs']
for name in name_list:
 if name in jobs:continue
 stem=Path(next(s['stem'] for s in extra if s['name']==name))
 # A source audio is the parent of the separate production vocal cache.
 # Get the original audio path from the archive of the two extra songs.
 report=next((ROOT/'_review/DREAM_BK_RECALIBRATION_20261009/reports').glob('*.align-report.json')) if name=='09.夢遊' else next((ROOT/'lyricsee/report/lrc_pilot20_audio_20261008/aligner_reports').glob('01 My Dearest--*.align-report.json'))
 metadata=json.loads(report.read_text(encoding='utf8'))
 audio=Path(metadata['audio_path'])
 BK=Path(r'D:\Users\Administrator\Music\LRC tools checked references\LRC bk')/(name+'.lrc')
 jobs[name]={'id':name,'title':name,'audio':str(audio),'lyrics':str(BK),
             'sidecar':None,'baseline':None}
assert len(name_list)==7
manifest=V/'PATCH7_SERIAL_MANIFEST.json'
if manifest.exists():
 previous=json.loads(manifest.read_text(encoding='utf8'))['cases']
else:previous=[]
done={(x['title'],x['mode']) for x in previous
      if x['status'] in ('GENERATED','FAILED','TIMED_OUT','FATAL')}
for title in name_list:
 job=jobs[title]
 for mode in ('segment','ctc'):
  if (title,mode) in done:
   print('PATCH7_RESUME_SKIP',title.encode('unicode_escape').decode(),mode,flush=True)
   continue
  task=V/(str(name_list.index(title))+'_'+mode)
  task.mkdir(exist_ok=True)
  outfile=task/'new.lrc';logfile=task/'full_production.log';reports=task/'reports'
  if outfile.exists():
   print('PATCH7_BLOCK_EXISTING_OUTPUT',str(outfile),flush=True)
   continue
  cmd=[sys.executable,str(ROOT/'scripts/auto_lrc.py'),job['audio'],
    '--lyrics',job['lyrics'],'--ignore-lyric-timestamps',
    '--timing-source','auto','--skeleton',mode,
    '--output',str(outfile),'--no-checked-lrc-hint','--no-anchor-hints',
    '--report-dir',str(reports),'--arbiter','reviewer-validity']
  if job.get('sidecar'):
   cmd+=['--reviewer-evidence',job['sidecar']]
  env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
       'LRC_EVIDENCE_WORKERS':'1'}
  print('PATCH7_CENTRAL_BEGIN',title.encode('unicode_escape').decode(),mode,flush=True)
  t=time.monotonic()
  try:
   with logfile.open('wb') as logs:
    outcome=subprocess.run(cmd,cwd=ROOT,env=env,stdout=logs,
             stderr=subprocess.STDOUT,timeout=1500)
   record={'title':title,'mode':mode,'status':'GENERATED' if outcome.returncode==0 and outfile.is_file() else 'FAILED',
     'exit':outcome.returncode}
  except subprocess.TimeoutExpired:
   record={'title':title,'mode':mode,'status':'TIMED_OUT'}
  except Exception as exc:
   record={'title':title,'mode':mode,'status':'FATAL','error':repr(exc)}
  record.update({'seconds':round(time.monotonic()-t,2),
                 'candidate_file':str(outfile),'log':str(logfile),
                 'BK_inference_time_read':False})
  if record['status']=='GENERATED':
   produced=a.load_lyrics(outfile).entries
   gold=a.load_lyrics(Path(job['lyrics'])).entries
   produced=a.remove_generated_title_cards(produced,Path(job['audio']))
   gold=a.remove_generated_title_cards(gold,Path(job['audio']))
   produced,_=a.remove_instrumental_markers(produced)
   gold,_=a.remove_instrumental_markers(gold)
   if len(gold)==len(produced)+1 and gold[0].source_time_cs==0:
    gold=gold[1:]
   correct=len(produced)==len(gold) and all(
     a.normalize_match_text(a.entry_sung_text(x))==
     a.normalize_match_text(a.entry_sung_text(y)) for x,y in zip(produced,gold))
   if correct:
    errors=[abs(x.source_time_cs-y.source_time_cs)
            for x,y in zip(produced,gold)]
    record['posthoc_BK']={'rows':len(errors),
        'PCO200':sum(x<=20 for x in errors),
        'wrong_gt1':sum(x>100 for x in errors),
        'nonmonotonic':sum(y.source_time_cs<x.source_time_cs
           for x,y in zip(produced,produced[1:])),
        'bad_rows':[i+1 for i,x in enumerate(errors) if x>100],
        'dream_row13_abs_s':round(errors[12]/100,3) if title=='09.夢遊' else None,
        'dream_row27_abs_s':round(errors[26]/100,3) if title=='09.夢遊' else None}
   else:record['posthoc_BK']={'status':'STRICT_LYRIC_MATCH_BLOCKED',
       'gold':len(gold),'produced':len(produced)}
   report_files=list(reports.glob('*.align-report.json'))
   if len(report_files)==1:
    body=json.loads(report_files[0].read_text(encoding='utf8'))
    record['skeleton_report']=body.get('segment_skeleton_status')
    record['central_trusted']=sum(bool(x.get('timing_trusted'))
                                for x in body.get('assignments',[]))
    record['selected_report']=str(report_files[0])
  previous.append(record)
  manifest.write_text(json.dumps({'status':'PATCH7_IN_PROGRESS','cases':previous,
       'BK_used_for_auto':False},ensure_ascii=False,indent=2),encoding='utf8')
  print('PATCH7_CENTRAL_DONE',title.encode('unicode_escape').decode(),mode,
      record['status'],'secs',record['seconds'],
      'BK',record.get('posthoc_BK'),flush=True)
manifest.write_text(json.dumps({'status':'PATCH7_ALL_CASES_PROCESSED',
   'cases':previous,'BK_used_for_auto':False},ensure_ascii=False,indent=2),encoding='utf8')
print('PATCH7_SERIAL_COMPLETE',len(previous),flush=True)