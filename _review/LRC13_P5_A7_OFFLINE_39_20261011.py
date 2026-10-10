"""39-song frozen-A7 posthoc replay of P5 F4/F5; no acoustic re-inference."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys
V=Path(__file__).resolve().parent
ROOT=V.parent
sys.path.insert(0,str(ROOT/'scripts'))
from LRC13_A7_FINALIZE_20261010 import parse,report,aligned_trust_assignments
import auto_lrc as a
items=json.loads((V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf-8'))['corpus']
A=V/'lrc13_option_a_A7_39_20261010'
B=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
records=[]
for k,item in enumerate(items,1):
    song=item.get('title',item['id'])
    folder=f'{k:02d}_{item["id"]}'
    src=A/folder/'candidate.lrc'
    old=B/folder/'main.lrc'
    gold=Path(item['human_BK'])
    bk=parse(gold,item['audio_path'])
    now=parse(src,item['audio_path'])
    main=parse(old,item['audio_path'])
    if len(bk)==len(now)+1 and bk[0][1]==0: bk=bk[1:]
    assert len(bk)==len(now)==len(main)==item['BK_sung_rows'],song
    assert all(x[0]==y[0]==z[0] for x,y,z in zip(bk,now,main)),song
    candidate=report(A/folder/'reports')
    baseline=report(B/folder/'reports')
    tb=aligned_trust_assignments(candidate,src,item['audio_path'],len(now),song,'segment')
    ta=aligned_trust_assignments(baseline,old,item['audio_path'],len(now),song,'pinned_main')
    entries=a.load_lyrics(gold).entries
    entries=a.remove_generated_title_cards(entries,Path(item['audio_path']))
    entries,_=a.remove_instrumental_markers(entries)
    if len(entries)==len(now)+1 and entries[0].source_time_cs==0:entries=entries[1:]
    assert len(entries)==len(now),song
    local=copy.deepcopy(candidate)
    local['assignments']=copy.deepcopy(tb)
    sk=local.get('segment_skeleton_status') or {}
    active=sk.get('status')=='SEGMENT_CURRENT'
    if active:
        for row,ref in zip(local['assignments'],ta):
            row['global_ctc_first_token_start']=ref.get('ctc_first_token_start')
            row['global_ctc_score']=ref.get('ctc_score')
            row['global_ctc_token_spans']=copy.deepcopy(ref.get('ctc_token_spans'))
    local['ctc_audio_source']='posthoc-skip-vocal-F1'
    st=SimpleNamespace(decisions=[SimpleNamespace(
        written_time=SimpleNamespace(seconds=x[1]),
        candidate=SimpleNamespace(source='ctc-current'))
        for x in now])
    before=[x.written_time.seconds for x in st.decisions]
    flags=a.compute_final_timing_self_check(local,st,entries)['flags_by_entry']
    assert before==[x.written_time.seconds for x in st.decisions]
    wrong=[i for i,(bkrow,nowrow) in enumerate(zip(bk,now),1) if abs(bkrow[1]-nowrow[1])>1]
    flagged=[i for i in wrong if str(i) in flags]
    f5=[i for i in range(1,len(now)+1) if 'skeleton_global_ctc_disagreement' in flags.get(str(i),[])]
    records.append({'song':song,'status':sk.get('status'),'active':active,'rows':len(now),
        'wrong_gt1':len(wrong),'wrong_gt1_flagged_F4_F5_F3_F2':len(flagged),
        'F5_flag_rows':f5,'wrong_gt1_unflagged':sorted(set(wrong)-set(flagged)),
        'F4_rows':[i for i in range(1,len(now)+1) if 'F4-ctc-final-disagreement' in flags.get(str(i),[])],
        'written_clocks_unchanged':True})
    print('REPLAY',k, 'ACTIVE' if active else 'FALLBACK',len(wrong),len(flagged),flush=True)
dream=next(x for x in records if x['song']=='09.夢遊')
assert 28 in dream['F5_flag_rows'],dream
active=[r for r in records if r['active']]
summary={'scope':'FROZEN_A7_OFFLINE_REPLAY_NO_FRESH_CTC',
         'pinned_main_ctc_evidence_as_proxy':True,
         'songs':len(records),'active':len(active),
         'active_wrong_gt1':sum(r['wrong_gt1'] for r in active),
         'active_wrong_flagged':sum(r['wrong_gt1_flagged_F4_F5_F3_F2'] for r in active),
         'active_f5_count':sum(len(r['F5_flag_rows']) for r in active),
         'dream_f5_row28':28 in dream['F5_flag_rows'],
         'no_time_mutation':all(r['written_clocks_unchanged'] for r in records),
         'requires_fresh_39_song_GPU_skeleton_layer_for_full_P5_acceptance':True,
         'records':records}
(V/'LRC13_P5_A7_OFFLINE_39_20261011.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# P5 39-song frozen A7 offline self-check replay','',
       'Frozen A7 outputs + pinned main CTC evidence; no fresh GPU/WhisperX/Central.',
       'Requires separate fresh 39-song skeleton-layer validation before full P5 acceptance.','',
       f'Active: {len(active)}/39; active >1s rows: {summary["active_wrong_gt1"]}; flagged: {summary["active_wrong_flagged"]}; F5 rows: {summary["active_f5_count"]}.',
       f'Dream #28 F5 flag: {summary["dream_f5_row28"]}; time mutation: {not summary["no_time_mutation"]}.','',
       '| Song | active | >1s wrong | flagged wrong | F5 rows |',
       '|---|---|---:|---:|---|']
for r in records:
    lines.append(f'| {r["song"].replace("|","/")} | {r["active"]} | {r["wrong_gt1"]} | {r["wrong_gt1_flagged_F4_F5_F3_F2"]} | {r["F5_flag_rows"]} |')
(V/'LRC13_P5_A7_OFFLINE_39_20261011.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPLAY39_OK',json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True),flush=True)
