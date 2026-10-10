"""Posthoc F4/F5 audit using frozen real A6 Dream global/local reports."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import auto_lrc as a
review=ROOT/'_review'
base=review/'LRC13_OPTION_A_A6_FULL_CENTRAL_20261010'
gold=Path(r'D:\Users\Administrator\Music\LRC tools checked references\LRC bk\09.夢遊.lrc')
audio=Path(json.loads((review/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf-8'))['corpus'][37]['audio_path'])
def rfile(folder):
    return json.loads(next((base/folder/'reports').glob('*.align-report.json')).read_text(encoding='utf-8'))
local=rfile('1_segment');ctc=rfile('1_ctc')
e=a.load_lyrics(gold).entries
e=a.remove_generated_title_cards(e,audio)
e,_=a.remove_instrumental_markers(e)
assert len(e)==30
assert len(local['assignments'])==len(ctc['assignments'])==30
s=a.load_lyrics(base/'1_segment'/'new.lrc').entries
s=a.remove_generated_title_cards(s,audio)
s,_=a.remove_instrumental_markers(s)
assert len(s)==30
times=[entry.source_time_cs/100. for entry in s]
report=copy.deepcopy(local)
for l,g in zip(report['assignments'],ctc['assignments']):
    l['global_ctc_first_token_start']=g.get('ctc_first_token_start')
    l['global_ctc_score']=g.get('ctc_score')
    l['global_ctc_token_spans']=copy.deepcopy(g.get('ctc_token_spans'))
report['ctc_audio_source']='posthoc-no-vocal-io'
state=SimpleNamespace(decisions=[
    SimpleNamespace(written_time=SimpleNamespace(seconds=t),
                    candidate=SimpleNamespace(source='ctc-current'))
    for t in times])
flags=a.compute_final_timing_self_check(report,state,e)
row=flags['flags_by_entry'].get('28',[])
assert abs(times[27]-160.89)<.001, (times[27],row)
assert abs(e[27].source_time_cs/100.-164.65)<.001
assert 'F4-ctc-final-disagreement' in row and 'skeleton_global_ctc_disagreement' in row,row
out={'audit':'reconstructed from frozen A6 reports; no new inference',
     'BK_row_28':164.65,'segment_row_28':times[27],
     'global_ctc_evidence_row_28':ctc['assignments'][27].get('ctc_first_token_start'),
     'row_28_flags':row,'flags_by_entry':flags['flags_by_entry'],
     'flag_count_by_type':flags['flag_count_by_type']}
path=review/'LRC13_P5_DREAM_REAL_EVIDENCE_20261011.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('P5_DREAM_REAL_EVIDENCE_OK',json.dumps({k:v for k,v in out.items() if k not in ('flags_by_entry','flag_count_by_type')},ensure_ascii=True))
