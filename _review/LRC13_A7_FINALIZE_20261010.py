"""A7 finalization: wait for serial 39+39 runs, evaluate frozen outputs,
write full human-review report and commit only lrc-1.3 locally.
BK clocks are read exclusively by this posthoc process, not by inference.
"""
from __future__ import annotations
from pathlib import Path
import ctypes, json, subprocess, sys, traceback
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'_review'
A=V/'lrc13_option_a_A7_39_20261010'
B=V/'lrc13_option_a_A7_main_9069dbc8_20261010'
OUT=V/'LRC13_A7_REPORT_20261010'
FLAGS=V/'LRC13_A7_LISTEN_FLAGS_20261010.md'
MAIN='9069dbc832b7b06c43b395f610483a8b1537b5ed'
sys.path.insert(0,str(ROOT/'scripts'))
import auto_lrc as al
from s5_regression_gate import evaluate_s5_rows,evaluate_s5_corpus

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):
    text=json.dumps(obj,ensure_ascii=False,indent=2) if isinstance(obj,dict) else obj
    p.write_text(text,encoding='utf-8')
def process(args,check=True):
    return subprocess.run(args,cwd=ROOT,check=check,capture_output=True,text=True,encoding='utf-8',errors='replace')
def wait_pid(pid):
    handle=ctypes.windll.kernel32.OpenProcess(0x00100000,False,pid)
    if handle:
        print('WAIT_EXISTING_PID',pid,flush=True)
        ctypes.windll.kernel32.WaitForSingleObject(handle,0xFFFFFFFF)
        ctypes.windll.kernel32.CloseHandle(handle)
def parse(path,audio):
    entries=al.load_lyrics(path).entries
    entries=al.remove_generated_title_cards(entries,Path(audio))
    entries,_=al.remove_instrumental_markers(entries)
    return [(al.normalize_match_text(al.entry_sung_text(e)),e.source_time_cs/100.) for e in entries]
def report(path):
    found=list(path.glob('*.align-report.json'))
    if len(found)!=1:raise ValueError('EXPECTED_ONE_ALIGN_REPORT '+str(path)+' found '+str(len(found)))
    return load(found[0])
TRUST_MARKER_EXCLUSIONS=[]
def aligned_trust_assignments(data, lrc_path, audio_path, expected_n, name, side):
    assignments=data.get('assignments',[])
    if len(assignments)==expected_n:
        return assignments
    raw=al.load_lyrics(lrc_path).entries
    raw=al.remove_generated_title_cards(raw,Path(audio_path))
    clean,skipped=al.remove_instrumental_markers(raw)
    if len(raw)!=len(assignments) or len(clean)!=expected_n or not skipped:
        raise ValueError('TRUST_ASSIGNMENTS_COUNT '+repr((name,side,len(assignments),len(raw),len(clean),expected_n)))
    if not all(isinstance(row,dict) and type(row.get('timing_trusted')) is bool for row in assignments):
        raise ValueError('TRUST_ASSIGNMENTS_MISSING '+repr((name,side)))
    if not all(row.get('entry')==i+1 for i,row in enumerate(assignments)):
        raise ValueError('TRUST_ENTRY_INDEX_UNMATCHED '+repr((name,side)))
    if any(abs(float(row.get('timestamp',-1000))-entry.source_time_cs/100.0)>.021
           for row,entry in zip(assignments,raw)):
        raise ValueError('TRUST_ENTRY_CLOCK_UNMATCHED '+repr((name,side)))
    keep=[i for i,entry in enumerate(raw) if not al.is_instrumental_marker(entry)]
    aligned=[assignments[i] for i in keep]
    if len(aligned)!=expected_n:
        raise ValueError('TRUST_FILTER_COUNT_UNMATCHED '+repr((name,side)))
    TRUST_MARKER_EXCLUSIONS.append({'song':name,'side':side,
                                    'markers':skipped,'raw_assignments':len(assignments),
                                    'scored_assignments':len(aligned)})
    return aligned

def mdt(v):
    if v is None:return '未提供'
    if isinstance(v,(int,float)):return f'{v:.5g}'
    return str(v)
def commits():
    return process(['git','log','--oneline',MAIN+'..HEAD']).stdout.strip().splitlines()
def status(payload):
    write(OUT.with_suffix('.json'),payload)
    write(OUT.with_suffix('.md'),'# LRC 1.3 A7（2026-10-10）\n\n驗收狀態：'+payload['status']+'\n\n'+
        json.dumps(payload.get('details',{}),ensure_ascii=False,indent=2)+'\n')
    write(FLAGS,'# A7 自檢聆聽旗標（2026-10-10）\n\n完整 A7 結果尚未生成，不能取代人工聆聽。\n')
def main():
    # Both frozen manifests are already complete; no stale PID wait on resume.
    manifest_paths=[A/'RUN_MANIFEST.json',B/'RUN_MANIFEST.json']
    missing=[str(x) for x in manifest_paths if not x.exists()]
    if missing:
        status({'status':'INCOMPLETE_A7_MANIFEST','details':{'missing':missing}})
        return
    ma,mb=(load(x) for x in manifest_paths)
    bad={'segment':{'status':ma.get('status'),'count':len(ma.get('runs',[])),
                    'failed':[r for r in ma.get('runs',[]) if r.get('status')!='PRODUCED']},
         'pinned':{'status':mb.get('status'),'count':len(mb.get('runs',[])),
                   'failed':[r for r in mb.get('runs',[]) if r.get('status')!='PRODUCED']}}
    if ma.get('status')!='COMPLETE_39' or mb.get('status')!='COMPLETE_39':
        status({'status':'INCOMPLETE_A7_GPU','details':bad})
        return
    print('A7_ALL_78_COMPLETE; POSTHOC_ONLY',flush=True)
    items=load(V/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json')['corpus']
    if len(items)!=39:raise ValueError('EXPECTED_39_CORPUS')
    gate_a6=load(V/'LRC13_OPTION_A_A6_POSTHOC_20261010.json')
    if gate_a6.get('status')!='PASS_A6_REPRESENTATIVE' or len(gate_a6.get('scores',[]))!=14:
        raise RuntimeError('A6_NOT_VALIDATED')
    songs=[]; gates=[]; listen=[]; introduced=[]; incomplete=[]
    for i,item in enumerate(items,1):
        name=item.get('title',item['id'])
        leaf=f'{i:02d}_{item["id"]}'
        gold=parse(Path(item['human_BK']),item['audio_path'])
        orig=parse(B/leaf/'main.lrc',item['audio_path'])
        now=parse(A/leaf/'candidate.lrc',item['audio_path'])
        if len(gold)==len(now)+1 and gold[0][1]==0:gold=gold[1:]
        n=item['BK_sung_rows']
        if len(gold)!=len(orig) or len(gold)!=len(now) or len(gold)!=n:
            incomplete.append({'song':name,'reason':'COUNT','counts':[len(gold),len(orig),len(now)],'expected':n})
            continue
        if any(a[0]!=b[0] or a[0]!=c[0] for a,b,c in zip(gold,orig,now)):
            incomplete.append({'song':name,'reason':'LEXICAL_IDENTITY'})
            continue
        report_main=report(B/leaf/'reports')
        report_now=report(A/leaf/'reports')
        try:
            ta=aligned_trust_assignments(report_main,B/leaf/'main.lrc',item['audio_path'],n,name,'pinned_main')
            tb=aligned_trust_assignments(report_now,A/leaf/'candidate.lrc',item['audio_path'],n,name,'segment')
        except (ValueError,KeyError,TypeError) as exc:
            incomplete.append({'song':name,'reason':'TRUST_ASSIGNMENTS_COUNT','detail':str(exc)})
            continue
        if not all(type(r.get('timing_trusted')) is bool for r in ta+tb):
            incomplete.append({'song':name,'reason':'TRUST_ASSIGNMENTS_MISSING'})
            continue
        rows=[{'text':q[0],'bk_s':q[1],
              'main_s':o[1],'candidate_s':c[1],
              'main_trusted':ta[j]['timing_trusted'],
              'candidate_trusted':tb[j]['timing_trusted']}
              for j,(q,o,c) in enumerate(zip(gold,orig,now))]
        g=evaluate_s5_rows(rows,song_id=name,expected_rows=n)
        gates.append(g)
        if not g.get('eligible_for_S5'):
            incomplete.append({'song':name,'reason':g['status']})
            continue
        sk=report_now.get('segment_skeleton_status') or {}
        dp=sk.get('DP_diagnostics') or {}
        rep=Counter(x[0] for x in gold)
        row_records=[]
        newgt1=[]
        for j,(q,o,c) in enumerate(zip(gold,orig,now),1):
            old=abs(o[1]-q[1]);new=abs(c[1]-q[1])
            row={'line':j,'text':q[0],'bk_s':round(q[1],2),
                 'main_s':round(o[1],2),'segment_s':round(c[1],2),
                 'main_error_s':round(old,2),'segment_error_s':round(new,2),
                 'repeated':rep[q[0]]>1}
            if old>1 or new>1:row_records.append(row)
            if old<=1 and new>1:
                newgt1.append(row)
                introduced.append({'song':name,**row,'decision':'PENDING_HUMAN_REVIEW',
                                   'active_or_fallback':sk.get('status','UNKNOWN')})
            flags=tb[j-1].get('flags') or []
            required=bool(tb[j-1].get('review_required'))
            if required or flags:
                listen.append({'song':name,'line':j,'time_s':round(c[1],2),
                               'text':q[0],'review_required':required,
                               'flags':flags,'timing_trusted':tb[j-1]['timing_trusted']})
        audioaudit=report_now.get('post_score_all_line_vocal_audit') or []
        for f in audioaudit:
            if f.get('review_required_before_audit'):
                ix=f.get('entry')
                if type(ix) is int and 1<=ix<=n and not any(z['song']==name and z['line']==ix for z in listen):
                    listen.append({'song':name,'line':ix,'time_s':round(now[ix-1][1],2),
                                   'text':gold[ix-1][0],'review_required_before_audit':True})
        counts=g['counts']
        songs.append({'song':name,'id':item['id'],'group':item.get('group'),'rows':n,
            'segment_status':sk.get('status','UNKNOWN'),'fallback_reason':sk.get('reason'),
            'DP_margin':sk.get('DP_margin',dp.get('DP_margin')),
            'DP_cost':sk.get('DP_cost',dp.get('DP_cost')),
            'DP_cost_per_row':dp.get('DP_cost_per_row'),
            'passages':sk.get('passages',dp.get('passages')),
            'passage_row_ratio':dp.get('ratio'),
            'mean_ctc_gain':sk.get('mean_ctc_gain'),
            'main_pco200':counts['main_pco200'],'segment_pco200':counts['candidate_pco200'],
            'main_gt1':counts['main_wrong_gt1s'],'segment_gt1':counts['candidate_wrong_gt1s'],
            'new_gt1':newgt1,'new_backtracks':g['new_backtrack_pairs'],
            'inherited_backtracks':g['inherited_backtrack_pairs'],
            'trusted':g['trusted'],'repeated_rows':counts['repeated_rows'],
            'repeated_main_gt1':counts['repeated_old_wrong'],
            'repeated_segment_gt1':counts['repeated_new_wrong'],
            'unique_main_gt1':counts['unique_old_wrong'],
            'unique_segment_gt1':counts['unique_new_wrong'],
            'review_rows':row_records})
        print('A7_POSTHOC',i,name.encode('unicode_escape').decode(),flush=True)
    if incomplete:
        status({'status':'INCOMPLETE_A7_SCORING','details':{'problems':incomplete,'paired_scored':len(songs)}})
        return
    corpus=evaluate_s5_corpus(gates,expected_songs=39)
    if corpus['status']!='S5_RELEASE_GATE_EVALUATED':
        status({'status':corpus['status'],'details':corpus})
        return
    active=[s for s in songs if s['segment_status']=='SEGMENT_CURRENT']
    fallbacks=[s for s in songs if s['segment_status']!='SEGMENT_CURRENT']
    diff_repeated={'rows':sum(s['repeated_rows'] for s in songs),
         'main_gt1':sum(s['repeated_main_gt1'] for s in songs),
         'segment_gt1':sum(s['repeated_segment_gt1'] for s in songs)}
    diff_unique={'rows':sum(s['rows']-s['repeated_rows'] for s in songs),
         'main_gt1':sum(s['unique_main_gt1'] for s in songs),
         'segment_gt1':sum(s['unique_segment_gt1'] for s in songs)}
    focus=['01 My Dearest','03. Choir ‘S’ Choir','01.ミレニアの水槽']
    coverage=[s for s in songs if s['song'] in focus or
              (s['segment_status']!='SEGMENT_CURRENT' and s['main_gt1']>=3)]
    worsening=[s['song'] for s in active if
               s['segment_gt1']>s['main_gt1'] or s['segment_pco200']<s['main_pco200']]
    cause_counts=Counter(s.get('fallback_reason') or 'UNSPECIFIED' for s in coverage)
    result={'status':'A7_SCORED_39','release_pass':corpus['release_pass'],
            'checks':corpus['checks'],'corpus':corpus,
            'a6':{'status':gate_a6['status'],'scored':14,
                  'inherited_backtracks':gate_a6.get('inherited_backtracks',[]),
                  'blockers':gate_a6.get('blockers',[])},
            'pinned_main':MAIN,'candidate_branch':'lrc-1.3',
            'active_count':len(active),'fallback_count':len(fallbacks),
            'activation_rate':len(active)/39.,
            'worsening_active_songs':worsening,
            'new_gt1_lines':introduced,
            'inherited_backtracks':corpus['inherited_backtracks'],
            'new_backtracks':corpus['new_backtracks'],
            'repeated_group':diff_repeated,'unique_group':diff_unique,
            'coverage':{'targets':coverage,'fallback_causes':dict(cause_counts),
             'generic_hypotheses':[
                 '對所有歌曲校準分段信心分數與 DP margin 分布；避免對特定歌設閾值',
                 '針對低 passages/rows 比率，評估通用 VAD 分段合併與行數容量聯合模型',
                 '使用與 BK 無關的局部 CTC 改善證據，驗證高代價 DP 是否具真正聲學增益'],
             'risks':['放寬 margin 易將副歌錯配到相似唱段',
                      'VAD 合併可能吞噬間奏或多位歌手重疊',
                      '任何調整都要在 39 首全部重新盲評']},
            'songs':songs,'trust_marker_exclusions':TRUST_MARKER_EXCLUSIONS,
            'BK_posthoc_only':True,
            'no_source_audio_or_BK_modified':True,
            'commits_before_report_commit':commits()}
    write(OUT.with_suffix('.json'),result)
    lines=['# LRC 1.3 A7 — 39 首正式回歸（2026-10-10）','',
       f"**正式 gate：{'PASS' if corpus['release_pass'] else 'NOT_PASS'}**",
       f"A6：{gate_a6['status']}，14/14；繼承倒退 {gate_a6.get('inherited_backtracks',[])}",'',
       '## 全池比較',
       f"PCO@0.2：{corpus['counts']['main_pco200']} → {corpus['counts']['candidate_pco200']} / {corpus['counts']['rows']}",
       f">1s：{corpus['counts']['main_wrong_gt1s']} → {corpus['counts']['candidate_wrong_gt1s']}；新增 >1s {corpus['counts']['new_wrong_gt1s']}",
       f"可信行 >1s 比例：{mdt(corpus['trusted_ratio_before'])} → {mdt(corpus['trusted_ratio_after'])}",
       f"啟用 {len(active)}/39（{100*len(active)/39:.1f}%）；退回 {len(fallbacks)} 首",
       f"新增倒退 {corpus['counts']['new_backtracks']}；INHERITED_BACKTRACK {corpus['counts']['inherited_backtracks']}；檢核：{corpus['checks']}",
       f"重複歌詞：{diff_repeated}；非重複：{diff_unique}",
       f"啟用卻變差：{', '.join(worsening) if worsening else '無'}",'',
       '## 39 首逐首',
       '| 歌 | 方案／原因 | PCO main→A7 | >1s main→A7 | 新 >1s | 新倒退／繼承 | DP margin |',
       '|---|---|---:|---:|---:|---:|---:|']
    for s in songs:
        lines.append(f"| {s['song'].replace('|','／')} | {s['segment_status']} / {s.get('fallback_reason') or '—'} | {s['main_pco200']}→{s['segment_pco200']}/{s['rows']} | {s['main_gt1']}→{s['segment_gt1']} | {len(s['new_gt1'])} | {len(s['new_backtracks'])}/{len(s['inherited_backtracks'])} | {mdt(s['DP_margin'])} |")
    lines+=['','## 新增 >1 秒逐行（交使用者判定）']
    if not introduced:lines.append('無。')
    for r in introduced:
        lines.append(f"- {r['song']} #{r['line']}：BK {r['bk_s']:.2f}，main {r['main_s']:.2f}，segment {r['segment_s']:.2f}；{r['active_or_fallback']}；**待使用者判定**")
    lines+=['','## INHERITED_BACKTRACK']
    if not corpus['inherited_backtracks']:lines.append('無。')
    for t in corpus['inherited_backtracks']:
        lines.append(f"- {t['song']}：相鄰行起點 {t['pairs']}；同一對 pinned main 已有倒退。")
    lines+=['','## 最差三首逐行']
    for s in sorted(songs,key=lambda r:(-r['segment_gt1'],-r['main_gt1']))[:3]:
        lines+=[f"### {s['song']}（{s['segment_gt1']} 行 >1s）",
                '| 行 | BK | main | segment | main 誤差 | segment 誤差 |','|---:|---:|---:|---:|---:|---:|']
        for r in s['review_rows']:
            lines.append(f"| {r['line']} | {r['bk_s']:.2f} | {r['main_s']:.2f} | {r['segment_s']:.2f} | {r['main_error_s']:.2f} | {r['segment_error_s']:.2f} |")
    lines+=['','## 涵蓋率分析（僅分析，未修改分段演算法）',
            '| 歌 | 退回條件 | DP margin | passage/rows | CTC 增益 | main >1s |',
            '|---|---|---:|---:|---:|---:|']
    for s in coverage:
        lines.append(f"| {s['song'].replace('|','／')} | {s.get('fallback_reason') or s['segment_status']} | {mdt(s['DP_margin'])} | {s['passages']}/{s['rows']} = {mdt(s['passage_row_ratio'])} | {mdt(s['mean_ctc_gain'])} | {s['main_gt1']} |")
    lines+=['',f'退回條件分布：{dict(cause_counts)}',
       '共通方向：先量測低 passage/rows 比率與 DP margin 的分布，再在所有歌曲上統一校準 VAD 分段及 DP 的聲學支持；不可針對單曲調參。',
       '風險：副歌錯配、間奏合併、放寬信心門檻後新增 >1s；需在完整 39 首重新驗證。','',
       '## 本地 commit 列表（報告寫入前）','']
    lines+=['- '+x for x in result['commits_before_report_commit']]
    lines+=['','不 push、不合併 main。LRC bk 與音樂唯讀；BK 只用於凍結後評分。']
    write(OUT.with_suffix('.md'),'\n'.join(lines)+'\n')
    fl=['# A7 聆聽旗標（2026-10-10）','',
        '只列 align-report 的自檢標記；不是已完成人工聆聽。','']
    if not listen:fl.append('無自檢旗標。')
    for r in listen:
        fl.append(f"- {r['song']} #{r['line']} @{r['time_s']:.2f}s：{r['text']}；{r.get('flags',[])}；review_required={r.get('review_required',r.get('review_required_before_audit',False))}")
    write(FLAGS,'\n'.join(fl)+'\n')
    print('A7_FINISHED_SCORING_AND_REPORTS',corpus['release_pass'],'flags',len(listen),flush=True)
    # Commit the evaluation scripts/reports without adding user-owned files.
    process(['git','add','-f','--',
         '_review/LRC13_A7_REPORT_20261010.md',
         '_review/LRC13_A7_REPORT_20261010.json',
         '_review/LRC13_A7_LISTEN_FLAGS_20261010.md',
         '_review/LRC13_A7_FINALIZE_20261010.py'])
    try:
        print(process(['git','commit','-m','test(lrc13): document 39-song A7 pinned-main regression']).stdout,flush=True)
    except subprocess.CalledProcessError as exc:
        print('REPORT_FILES_WRITTEN_BUT_COMMIT_FAILED',str(exc),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc()
        write(V/'LRC13_A7_FINALIZE_ERROR_20261010.txt',error)
        print(error,flush=True)
        raise
