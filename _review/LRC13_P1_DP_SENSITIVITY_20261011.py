"""P1 W_ASR sensitivity: all 39 *DP plans*, fixed 1.0 production weight.
This is not an acoustic/local-CTC acceptance test. No BK scoring.
"""
from pathlib import Path
import copy,json,sys,functools
v=Path(__file__).resolve().parent
root=v.parent
sys.path[:0]=[str(root/'scripts'),str(v)]
import auto_lrc as al
import segment_skeleton as seg
from LRC13_A7_FINALIZE_20261010 import aligned_trust_assignments,report,parse
items=json.loads((v/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
asr=json.loads((v/'LRC13_P1_RAW_ASR_PARAGRAPHS_20261011.json').read_text(encoding='utf8'))['songs']
assert len(asr)==len(items)==39,'ALL 39 RAW PARAGRAPHS ARE REQUIRED'
frozen=v/'lrc13_option_a_A7_main_9069dbc8_20261010'
weights=(.5,1.0,2.)
out={'stage':'DP-only ASR weight sensitivity; not a full local acoustic CTC test',
     'production_weight':1.0,'weights':{},'songs':[]}
normal=seg._path_cost
for idx,item in enumerate(items,1):
    assert asr[idx-1]['id']==item['id'] and asr[idx-1]['index']==idx
    leaf=f'{idx:02d}_{item["id"]}'
    audio=Path(item['audio_path'])
    raw=parse(Path(item['human_BK']),audio)
    if len(raw)==item['BK_sung_rows']+1 and raw[0][1]==0:raw=raw[1:]
    timestamps=report(frozen/leaf/'reports')
    ass=aligned_trust_assignments(timestamps,frozen/leaf/'main.lrc',audio,len(raw),item.get('title',item['id']),'frozen_ctc')
    priors=[float(x.get('ctc_first_token_start',x.get('timestamp'))) for x in ass]
    stem=Path(timestamps['ctc_audio_path'])
    duration=al.probe_duration(audio)
    feature=al.analyze_audio(stem,duration)
    events=tuple(tuple(e) for e in asr[idx-1]['events'])
    row={'song':item.get('title',item['id']),'asr_events':len(events),'plans':{}}
    for w in weights:
        def cost(*args,**kwargs):
            kwargs['asr_weight']=w
            return normal(*args,**kwargs)
        seg._path_cost=cost
        plan=seg.plan_segment_skeleton(
           feature.segments,duration=duration,
           lyric_texts=[x[0] for x in raw],romanizer=al.japanese_romaji,
           ctc_priors=priors,raw_asr_events=events)
        row['plans'][str(w)]={'status':plan.status,'reason':plan.reason,
           'DP_margin':plan.dp_margin,'groups':list(plan.groups),
           'passage_ids':[a.passage_index for a in plan.assignments]}
    seg._path_cost=normal
    out['songs'].append(row)
    print('P1_DP_SENSITIVITY',idx,'/39',
          *(f'w={w}:{row["plans"][str(w)]["status"]}' for w in weights),flush=True)
for w in weights:
    out['weights'][str(w)]={'DP_READY':sum(s['plans'][str(w)]['status']=='READY' for s in out['songs']),
        'DP_FALLBACK':sum(s['plans'][str(w)]['status']!='READY' for s in out['songs'])}
    out['weights'][str(w)]['groups_changed_vs_w1']=sum(
        s['plans'][str(w)]['groups']!=s['plans']['1.0']['groups']
        for s in out['songs'])
target=v/'LRC13_P1_DP_SENSITIVITY_20261011.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print('DP_SENSITIVITY_DONE',json.dumps(out['weights']),flush=True)
