"""Audit ASR-to-passages evidence without reading BK timestamps."""
from pathlib import Path
import json,sys,statistics
v=Path(__file__).resolve().parent
sys.path[:0]=[str(v.parent/'scripts'),str(v)]
import auto_lrc as al
import segment_skeleton as ss
from LRC13_A7_FINALIZE_20261010 import parse,report,aligned_trust_assignments
c=json.loads((v/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
asr=json.loads((v/'LRC13_P1_RAW_ASR_PARAGRAPHS_20261011.json').read_text(encoding='utf8'))['songs']
manifest=json.loads((v/'LRC13_P1_SKELETON_39_20261011'/'RUN_MANIFEST.json').read_text(encoding='utf8'))
pin=v/'lrc13_option_a_A7_main_9069dbc8_20261010'
rows=[]
for idx,(item,record) in enumerate(zip(c,manifest['runs']),1):
    if record['status']!='SEGMENT_CURRENT':continue
    audio=Path(item['audio_path']); duration=al.probe_duration(audio)
    raw=parse(Path(item['human_BK']),audio)
    if len(raw)==item['BK_sung_rows']+1 and raw[0][1]==0:raw=raw[1:]
    frozen=report(pin/f'{idx:02d}_{item["id"]}'/'reports')
    assignments=aligned_trust_assignments(frozen,pin/f'{idx:02d}_{item["id"]}'/'main.lrc',audio,len(raw),record['title'],'frozen_ctc')
    prior=[float(r.get('ctc_first_token_start',r.get('timestamp'))) for r in assignments]
    stem=Path(frozen['ctc_audio_path'])
    features=al.analyze_audio(stem,duration)
    ev=tuple(tuple(x) for x in asr[idx-1]['events'])
    plan=ss.plan_segment_skeleton(features.segments,duration=duration,
         lyric_texts=[x[0] for x in raw],romanizer=al.japanese_romaji,
         ctc_priors=prior,raw_asr_events=ev)
    assert plan.status=='READY',(idx,plan.status)
    vocal=plan.passages
    contents=ss._asr_for_passages(vocal,ev,al.japanese_romaji)
    groups=[]
    for first,end,j in plan.groups:
        lyr=''.join(ss._text_for_match(raw[i][0],al.japanese_romaji) for i in range(first,end))
        matched=contents[j]
        dice=ss._dice_bigram(lyr,matched) if matched else None
        groups.append({'rows':[first+1,end],'passage':j,
             'ASR_chars':len(matched),'lyric_chars':len(lyr),
             'dice':round(dice,4) if dice is not None else None})
    known=[x['dice'] for x in groups if x['dice'] is not None]
    q={'index':idx,'song':record['title'],'groups':groups,
       'asr_paragraphs':len(ev),'asr_uncovered_groups':len(groups)-len(known),
       'mean_dice':round(statistics.mean(known),4) if known else None,
       'min_dice':min(known) if known else None,
       'groups_lt_030':sum(x<.30 for x in known),
       'groups_lt_050':sum(x<.50 for x in known),
       'groups_lt_070':sum(x<.70 for x in known)}
    print(idx,'mean_dice',q['mean_dice'],'weak_0.30',q['groups_lt_030'],'groups',len(groups),flush=True)
    rows.append(q)
out={'rule':'No BK timestamps or scoring used. Acoustic and raw ASR transcript content only.',
     'songs':rows}
(v/'LRC13_P1_ASR_CONTENT_AUDIT_20261011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
