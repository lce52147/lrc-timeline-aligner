"""Frozen N2 posthoc flags + source attribution audit, no inference or production edits."""
import json
from pathlib import Path
from collections import Counter,defaultdict
R=Path(__file__).resolve().parent
n2base=R/'LRC13_LEAN_N2_CTC_FIRST_39_20261011'
a7base=R/'LRC13_FINAL_A7_PATCHED_39_20261011'
m=json.loads((n2base/'MANIFEST.json').read_text(encoding='utf8'))
assert m.get('phase')=='N2_39_FROZEN' and len(m['cases'])==39 and all(c['status']=='PRODUCED' for c in m['cases'])
sc=json.loads((R/'LRC13_LEAN_N2_SCORE_20261011.json').read_text(encoding='utf8'))
nr=json.loads((R/'LRC13_LEAN_N1_FLAGS_20261011.json').read_text(encoding='utf8'))
corpus=json.loads((R/'LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json').read_text(encoding='utf8'))['corpus']
assert len(sc['rows'])==len(nr['rows'])==1487 and len(corpus)==39
rmap={(x['song'],x['line']):x for x in sc['rows']}
fmap={(x['song'],x['line']):x for x in nr['rows']}
def report(base,leaf):
    files=list((base/leaf/'reports').glob('*.align-report.json'))
    assert len(files)==1,(leaf,len(files))
    return json.loads(files[0].read_text(encoding='utf8'))
def aslist(v):
    if isinstance(v,str):return [v] if v else []
    return [str(x) for x in v] if isinstance(v,list) else []
def flag(rep,line,target,name):
    c=rep['central_timing_state'][line-1];a=rep['assignments'][line-1]
    assert c['entry']==a['entry']==line,(name,line)
    assert abs(c['timestamp']-target)<.021,(name,line,c['timestamp'],target)
    return {'marked':not bool(c.get('timing_trusted')) or bool(a.get('review_required')),
      'untrusted':not bool(c.get('timing_trusted')),
      'review':bool(a.get('review_required')),
      'flags':set(aslist(a.get('flags'))),
      'failures':set(aslist(a.get('unresolved_reason'))),
      'reasons':set(aslist(a.get('review_reason_classes')))}
rows=[];per=[];ca=Counter();cn=Counter();fa=Counter();fn=Counter();ra=Counter();rn=Counter()
for i,item in enumerate(corpus,1):
    name=item.get('title',item['id']);leaf=f'{i:02d}_{item["id"]}'
    a=report(a7base,leaf);n=report(n2base,leaf)
    subset=sorted([x for (s,_),x in rmap.items() if s==name],key=lambda x:x['line'])
    assert len(subset)==len(a['assignments'])==len(n['assignments'])
    sa=Counter();sn=Counter();sfa=Counter();sfn=Counter();sra=Counter();srn=Counter()
    lost=[];gained=[];ac=0;nc=0
    for z in subset:
        j=z['line'];af=flag(a,j,z['A7'],name);nf=flag(n,j,z['N2'],name);q=fmap[name,j]
        row={**z,'skeleton':q['skeleton_s'],'A7_mark':af['marked'],'N2_mark':nf['marked'],
          'N1_mark':q['flagged'],'A7_flags':sorted(af['flags']),'N2_flags':sorted(nf['flags']),
          'A7_failures':sorted(af['failures']),'N2_failures':sorted(nf['failures']),
          'A7_review_reasons':sorted(af['reasons']),'N2_review_reasons':sorted(nf['reasons'])}
        rows.append(row)
        ac+=af['marked'];nc+=nf['marked']
        if af['marked'] and not nf['marked']:lost.append(j)
        if nf['marked'] and not af['marked']:gained.append(j)
        for k in af['flags']:ca[k]+=1;sa[k]+=1
        for k in nf['flags']:cn[k]+=1;sn[k]+=1
        for k in af['failures']:fa[k]+=1;sfa[k]+=1
        for k in nf['failures']:fn[k]+=1;sfn[k]+=1
        for k in af['reasons']:ra[k]+=1;sra[k]+=1
        for k in nf['reasons']:rn[k]+=1;srn[k]+=1
    def diff(x,y):
        return {k:y[k]-x[k] for k in sorted(x.keys()|y.keys()) if x[k]!=y[k]}
    per.append({'song':name,'rows':len(subset),'A7_marked':ac,'N2_marked':nc,
     'lost_lines':lost,'gained_lines':gained,'flags_delta':diff(sa,sn),
     'failures_delta':diff(sfa,sfn),'review_reasons_delta':diff(sra,srn)})
assert len(rows)==1487
def metric(time,mark):
    return {'PCO200':sum(abs(z[time]-z['BK'])<=.2+1e-9 for z in rows),
      'gt1':sum(abs(z[time]-z['BK'])>1 for z in rows),
      'gt05':sum(abs(z[time]-z['BK'])>.5 for z in rows),
      'listen':sum(z[mark] for z in rows),
      'listen_ratio':sum(z[mark] for z in rows)/len(rows),
      'silent_gt1':sum(abs(z[time]-z['BK'])>1 and not z[mark] for z in rows),
      'silent_gt05':sum(abs(z[time]-z['BK'])>.5 and not z[mark] for z in rows)}
metrics={k:metric(*v) for k,v in {
 'skeleton_A7':('skeleton','A7_mark'),
 'skeleton_N2':('skeleton','N2_mark'),
 'N2_N2':('N2','N2_mark'),
 'N2_A7':('N2','A7_mark'),
 'A7_A7':('A7','A7_mark'),
 'skeleton_N1':('skeleton','N1_mark'),
 'N2_N1':('N2','N1_mark'),
 'main_A7':('main','A7_mark')}.items()}
assert metrics['skeleton_A7']['silent_gt1']==3 and metrics['skeleton_A7']['silent_gt05']==13,metrics['skeleton_A7']
assert metrics['N2_N2']['PCO200']==1317 and metrics['N2_N2']['gt1']==82,metrics['N2_N2']
assert metrics['skeleton_A7']['PCO200']==1314 and metrics['skeleton_A7']['gt1']==83
def newwrong(old,new):
    return [{'song':z['song'],'line':z['line'],'BK':z['BK'],
      'old':z[old],'new':z[new]} for z in rows if abs(z[old]-z['BK'])<=1 and abs(z[new]-z['BK'])>1]
def silent(time,mark,t):
    return [{'song':z['song'],'line':z['line'],'BK':z['BK'],
             'time':z[time],'err':round(z[time]-z['BK'],3),
             'A7_flags':z['A7_flags'],'N2_flags':z['N2_flags'],
             'A7_review_reasons':z['A7_review_reasons'],'N2_review_reasons':z['N2_review_reasons']}
            for z in rows if abs(z[time]-z['BK'])>t and not z[mark]]
def layer(source):
    s=(source or '').lower()
    if 'ctc' in s and 'fusion' not in s and 'whisper' not in s:return 'CTC/local'
    if 'whisper' in s or 'raw' in s:return 'Whisper/RAW'
    if 'review' in s or 'hubert' in s:return 'reviewer'
    if 'phase4' in s:return 'Phase4B'
    if 'vocal' in s:return 'vocal-onset'
    return source or 'unknown'
sources=defaultdict(Counter);names=defaultdict(Counter)
for z in rows:
    s=layer(z.get('N2_source'));names[s][z['N2_source']]+=1
    if abs(z['N2']-z['CTC'])<=.05:continue
    q=sources[s];q['changed']+=1
    e0=abs(z['CTC']-z['BK']);e1=abs(z['N2']-z['BK'])
    if e0-e1>.2:q['improved']+=1
    if e1-e0>.2:q['worsened']+=1
    if e0<=1 and e1>1:q['new_gt1']+=1
    if e0>1 and e1<=1:q['repaired_gt1']+=1
for q in sources.values():
    q['net_improved']=q['improved']-q['worsened']
    q['net_gt1_repairs']=q['repaired_gt1']-q['new_gt1']
lost=[{'song':z['song'],'line':z['line'],'N2_err':round(z['N2']-z['BK'],3),
       'A7_flags':z['A7_flags'],'N2_flags':z['N2_flags'],
       'A7_review_reasons':z['A7_review_reasons'],'N2_review_reasons':z['N2_review_reasons']}
      for z in rows if z['A7_mark'] and not z['N2_mark']]
gained=[{'song':z['song'],'line':z['line']} for z in rows if z['N2_mark'] and not z['A7_mark']]
back=[]
for song in per:
    sr=[z for z in rows if z['song']==song['song']]
    for a,b in zip(sr,sr[1:]):
        if a['N2']>b['N2']+1e-8 and a['skeleton']<=b['skeleton']+1e-8:
            back.append({'song':song['song'],'line':a['line']})
audit={'status':'N2_FINISH_AUDITED_39','metrics':metrics,
  'flag_A7_counts':dict(ca),'flag_N2_counts':dict(cn),'flag_deltas':diff(ca,cn),
  'failure_deltas':diff(fa,fn),'review_reason_deltas':diff(ra,rn),
  'marked_lost':lost,'marked_gained':gained,'per_song':per,
  'new_gt1_vs_A7':newwrong('A7','N2'),
  'new_gt1_vs_skeleton':newwrong('skeleton','N2'),
  'new_backtracks_vs_skeleton':back,
  'silent_gt1_skeleton_A7':silent('skeleton','A7_mark',1),
  'silent_gt1_N2_N2':silent('N2','N2_mark',1),
  'silent_gt05_N2_N2':silent('N2','N2_mark',.5),
  'source_net_effect':{k:dict(v) for k,v in sources.items()},
  'source_names':{k:dict(v) for k,v in names.items()},
  'notes':[
   '1.3 A7 flags same definition as Opus: cts timing_trusted false OR assignment review_required true.',
   'Code: scripts/auto_lrc.py 26158/26185 removes Whisper/RAW before build_final_timing_state (4198), select_timing_decision (3875).',
   '_final_post_central_material_conflict_entries at 24032 uses rejected candidate decision graph; commit_final_timing_state at 24801 flags unresolved-material-alternative.',
   'Some precentral candidate_disagreement may persist from add_alignment_diagnostics or earlier report flags; absence of alternatives removes Central-specific evidence.',
   'Proposal only, NOT IMPLEMENTED: keep full candidate_sets and restrict winner eligibility within select_timing_decision, preserving ineligible candidates in rejected alternatives for material conflict analysis; propagate policy through full evaluation/re-evaluation and test default identity, CTC final restriction and review flag preservation.',
   'B11 source layering and >.05s change / >.2s benefit thresholds reproduced from lrc13_anchor_B11_offline_20261011.py.'
 ]}
pmd=R/'LRC13_LEAN_PIPELINE_REPORT_20261011.md'
pjson=R/'LRC13_LEAN_PIPELINE_REPORT_20261011.json'
oldmd=pmd.read_text(encoding='utf8')
old=json.loads(pjson.read_text(encoding='utf8'))
old['N2_finish_audit']=audit
old['status']='LEAN_PIPELINE_N2_FINISH_AUDITED_39'
pjson.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
A=metrics['skeleton_A7'];N=metrics['N2_N2']
lead=f"N2 時間 PCO {N['PCO200']}、>1s {N['gt1']}；精簡時間 PCO {A['PCO200']}、>1s {A['gt1']}。N2 標記需聆聽 {N['listen_ratio']:.1%}、沉默 >1s {N['silent_gt1']}、>0.5s {N['silent_gt05']}；基準精簡時間＋A7 標記 {A['listen_ratio']:.1%}／{A['silent_gt1']}／{A['silent_gt05']}。"
assessment="達標" if N['listen_ratio']<=A['listen_ratio'] and N['silent_gt1']<=A['silent_gt1'] and N['silent_gt05']<=A['silent_gt05'] else "未達標"
sec=['## N2 完成後的正式驗收補充（2026-10-11）','','**'+lead+'標記'+assessment+'。**',
 '','39 首為開發集；N2 相較 A7 新增 >1s 行 '+str(len(audit['new_gt1_vs_A7']))+'，相較精簡新增 '+str(len(audit['new_gt1_vs_skeleton']))+' 行，不能稱為逐行無退化。',
 '','### 時間與標記比較','| 時間＋標記 | PCO | >1s | >0.5s | 需聆聽 | 沉默 >1s | 沉默 >0.5s |',
 '|---|---:|---:|---:|---:|---:|---:|']
labels={'skeleton_A7':'精簡時間＋A7 標記','skeleton_N2':'精簡時間＋N2 標記','N2_N2':'N2＋N2 標記','N2_A7':'N2＋A7 標記','A7_A7':'A7＋A7 標記','skeleton_N1':'精簡時間＋N1 六規則','N2_N1':'N2＋N1 六規則','main_A7':'main＋A7 標記'}
for k,x in metrics.items():
    sec.append(f"| {labels[k]} | {x['PCO200']} | {x['gt1']} | {x['gt05']} | {x['listen']}/1487 ({x['listen_ratio']:.1%}) | {x['silent_gt1']} | {x['silent_gt05']} |")
sec+=['','### N2 相較精簡時間新增 >1s（逐行）','| 歌曲 | 行 | BK | 精簡時間 | N2 |','|---|---:|---:|---:|---:|']
for x in audit['new_gt1_vs_skeleton']:
    sec.append(f"| {x['song']} | {x['line']} | {x['BK']} | {x['old']} | {x['new']} |")
sec+=['','### N2 相較精簡時間新增倒退','新增倒退共 '+str(len(back))+' 處：'+json.dumps(back,ensure_ascii=False)+'。','']
sec+=['','### 逐首原生標記差異','| 歌曲 | A7 標記 | N2 標記 | 差異 | 流失 | 新增 |','|---|---:|---:|---:|---:|---:|']
for x in per:
    sec.append(f"| {x['song']} | {x['A7_marked']} | {x['N2_marked']} | {x['N2_marked']-x['A7_marked']:+d} | {len(x['lost_lines'])} | {len(x['gained_lines'])} |")
sec+=['','流失標記 '+str(len(lost))+' 行，新增標記 '+str(len(gained))+' 行；逐首詳細原因差異、流失位置、所有沉默錯誤見 JSON N2_finish_audit。',
 '','### 標記原因差異（逐行 flag/failure/review_reason）','| 類型 | 名稱 | A7 | N2 | 差異 |','|---|---|---:|---:|---:|']
for title,aa,nn in [('flags',ca,cn),('failures',fa,fn),('review_reasons',ra,rn)]:
    for k in sorted(aa.keys()|nn.keys(),key=lambda k:(0 if k in ('candidate_disagreement','unresolved-material-alternative') else 1,k)):
        sec.append(f"| {title} | {k} | {aa[k]} | {nn[k]} | {nn[k]-aa[k]:+d} |")
sec+=['','### B11 最終來源淨效益（CTC 對 N2）',
 'B11 原分類：CTC/local、Whisper/RAW、reviewer、Phase4B、vocal-onset、其他 source；時間移動 >0.05s 才計，改善／惡化 >0.2s。',
 '| 層 | 改動 | 改善 | 惡化 | 淨改善 | 新增 >1s | 修復 >1s |','|---|---:|---:|---:|---:|---:|---:|']
for k,q in sorted(sources.items(),key=lambda z:-z[1]['changed']):
    sec.append(f"| {k} | {q['changed']} | {q['improved']} | {q['worsened']} | {q['net_improved']:+d} | {q['new_gt1']} | {q['repaired_gt1']} |")
sec+=['','### 程式層級標記流失與修法（僅提案，未實作）',
 'scripts/auto_lrc.py:26158 apply_central_timing_source_policy 在 evaluate_backend_timing（26185）入口刪除 Whisper/RAW candidates；build_final_timing_state（4198）以及 select_timing_decision（3875）無法保留這些候選於 rejected_candidates。_final_post_central_material_conflict_entries（24032）看 decision graph，因此其 unresolved-material-alternative 路徑會失去已刪除來源的證據。commit_final_timing_state（24801）又依此產生 audit/review 標記。前置 candidate_disagreement 由 add_alignment_diagnostics 及 report flags 生成的部分可能繼續存在，需看上表實測數字。',
 '提案：保留完整候選供不一致判斷；僅在 select_timing_decision 最終 winner 資格處套用 ctc-first，禁止 Whisper/RAW 決定時間，但保留其 rejected alternatives 和相應證據；透過 build_final_timing_state 及 rerun 路徑傳遞 policy。補上 identity、禁用來源、衝突標記保留及 39 首回歸測試後交 Opus 審查。本輪不修改生產程式。',
 '','### N1／N3：Opus 審查結論與來源',
 '直接引用 _review/OPUS_LEAN_N1_N3_REVIEW_20261011.md：N3 s_min=0 的 39 首同位測試通過；僅對旗標歌使用 0.10／0.12／0.15 門檻時 >1s 分別 117／112／113，各新增 2／2／3 行 >1s，未能安全修正壓縮錯位。N1 六規則：聆聽 50.9%、沉默 >1s 6 行、>0.5s 13 行；精簡時間＋A7 原標記：聆聽 41.6%、沉默 >1s 3 行、>0.5s 13 行。39 首屬開發集，無獨立 holdout。','']
marker='## N2 完成後的正式驗收補充（2026-10-11）'
if marker in oldmd:oldmd=oldmd.split(marker)[0].rstrip()
pmd.write_text(oldmd.rstrip()+'\n\n'+'\n'.join(sec).rstrip()+'\n',encoding='utf8')
print('AUDIT_COMPLETE',json.dumps({'metrics':metrics,'lost':len(lost),'gained':len(gained),'focus':{k:(ca[k],cn[k],fa[k],fn[k],ra[k],rn[k]) for k in ['candidate_disagreement','unresolved-material-alternative']},'new_vs_skeleton':len(audit['new_gt1_vs_skeleton']),'source_net':dict(sources)},ensure_ascii=False))
