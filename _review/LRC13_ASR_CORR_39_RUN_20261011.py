"""P5: 39-song skeleton-layer regression using frozen pinned main CTC evidence.

No Central, WhisperX, reviewer. Audio is read-only; BK timestamps never enter
skeleton inference. All results are stored in a separate review folder.
"""
from __future__ import annotations
from pathlib import Path
import copy,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(V))
import auto_lrc as al
from segment_skeleton_production import run_segment_skeleton
from LRC13_A7_FINALIZE_20261010 import aligned_trust_assignments,report,parse
DATA=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
assert len(DATA)==39
ASR_CACHE=json.loads((V/'LRC13_P1_RAW_ASR_PARAGRAPHS_20261011.json').read_text(encoding='utf8'))['songs']
assert len(ASR_CACHE)==39
PIN=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
OUT=V/'LRC13_ASR_CORR_39_20261011'
OUT.mkdir(exist_ok=True)
MANIFEST=OUT/'RUN_MANIFEST.json'
records=(json.loads(MANIFEST.read_text(encoding='utf8')).get('runs',[]) if MANIFEST.exists() else [])
done={r['id'] for r in records if r.get('status') in ('PRODUCED','FALLBACK_CTC')}
def manifest(status):
    MANIFEST.write_text(json.dumps({'status':status,'count':len(records),
        'runs':records,'baseline':'pinned_main_CTC_report_9069dbc8',
        'no_BK_timestamps_in_inference':True,
        'code': 'ASR_CORR_REPEAT_GLOBAL_ROLLBACK_W1','GPU_serial':True},ensure_ascii=False,indent=2),encoding='utf8')
manifest('IN_PROGRESS')
for index,item in enumerate(DATA,1):
    if item['id'] in done:continue
    leaf=f'{index:02d}_{item["id"]}'
    name=item.get('title',item['id'])
    target=OUT/leaf
    target.mkdir(exist_ok=True)
    log=target/'stage.json'
    if log.is_file(): raise RuntimeError('OUTPUT_EXISTS_WITHOUT_COMPLETED_MANIFEST '+leaf)
    t=time.monotonic()
    print('ASR_CORR_BEGIN',index,'/39',name.encode('unicode_escape').decode(),flush=True)
    try:
        base=report(PIN/leaf/'reports')
        audio=Path(item['audio_path'])
        # Extract *text only*; no input timestamps influence inference.
        text_rows=parse(Path(item['human_BK']),audio)
        if len(text_rows)==item['BK_sung_rows']+1 and text_rows[0][1]==0:
            text_rows=text_rows[1:]
        assert len(text_rows)==item['BK_sung_rows'],('INPUT_COUNT',name,len(text_rows))
        # Reconstruct whole-song CTC evidence from frozen pinned-main report.
        ass=aligned_trust_assignments(base,PIN/leaf/'main.lrc',audio,len(text_rows),name,'frozen_ctc')
        original=[float(r.get('ctc_first_token_start',r.get('timestamp'))) for r in ass]
        assert len(original)==len(text_rows) and all(0<=x<100000 for x in original)
        frozen=copy.deepcopy(base)
        frozen['assignments']=copy.deepcopy(ass)
        stem=Path(frozen['ctc_audio_path'])
        if frozen.get('ctc_audio_source')!='vocal-stem' or not stem.is_file():
            raise RuntimeError('FROZEN_VOCAL_STEM_MISSING '+repr(stem))
        duration=al.probe_duration(audio)
        assert ASR_CACHE[index-1]['id']==item['id']
        events=tuple((float(x[0]),float(x[1]),str(x[2]))
                      for x in ASR_CACHE[index-1]['events'])
        actual,updated,sk=run_segment_skeleton(
            report=frozen,timestamps=original,
            lyric_texts=[r[0] for r in text_rows],
            vocal_stem=stem,duration=duration,audio_api=al,
            python_exe=al.default_ctc_python(),
            ctc_helper=ROOT/'scripts'/'ctc_align.py',
            device='auto',
            helper_runner=lambda command,timeout: al._run_model_helper_command(command,mode='ctc'),
            romanizer=al.japanese_romaji,raw_asr_events=events)
        assert len(actual)==len(text_rows),('ACTUAL_COUNT',name)
        if sk['status']=='SEGMENT_CURRENT':
            assert sk['DP_diagnostics'].get('P2_min_marginal') is True
            assert len(sk['DP_diagnostics']['per_line_min_marginal'])==len(text_rows)
            assert all('global_ctc_first_token_start' in r for r in updated['assignments'])
            assert all(abs(r['global_ctc_first_token_start']-s)<.002 for r,s in zip(updated['assignments'],original))
        elif sk['status']=='FALLBACK_CTC':
            assert actual==original,('FALLBACK_MUTATED',name)
        else:raise RuntimeError('BAD_STATUS '+str(sk))
        payload={'id':item['id'],'song':name,'status':sk['status'],
                 'reason':sk.get('reason'),
                 'baseline_times':original,'skeleton_times':actual,
                 'DP_margin':sk.get('DP_margin'),
                 'BK_not_used_for_inference':True,
                 'raw_asr_paragraphs':len(events),
                 'elapsed_s':round(time.monotonic()-t,3),
                 'rows':len(text_rows)}
        log.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        record={'id':item['id'],'title':name,'status':sk['status'],
                'reason':sk.get('reason'),'elapsed_s':payload['elapsed_s'],
                'rows':len(text_rows)}
    except BaseException as exc:
        record={'id':item['id'],'title':name,'status':'FAILED',
                'exception':str(exc),'trace':traceback.format_exc(),
                'elapsed_s':round(time.monotonic()-t,3)}
        (target/'error.txt').write_text(record['trace'],encoding='utf8')
    records.append(record)
    manifest('IN_PROGRESS')
    print('ASR_CORR_DONE',index,record['status'],round(record['elapsed_s'],2),
          str(record.get('reason',''))[:80],flush=True)
    if record['status']=='FAILED':
        manifest('FAILED'); raise RuntimeError('ASR_CORR_FAILED '+record['title']+' '+record['exception'])
manifest('COMPLETE_39' if len(records)==39 else 'INCOMPLETE')
print('ASR_CORR_ALL_DONE',len(records),flush=True)