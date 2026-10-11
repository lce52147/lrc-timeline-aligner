"""N1/N2/N3 final report, requires completed frozen reports."""
from pathlib import Path
import json
V=Path(__file__).resolve().parent
n3=json.loads((V/'LRC13_LEAN_N3_SCORE_20261011.json').read_text(encoding='utf8'))
n2=json.loads((V/'LRC13_LEAN_N2_SCORE_20261011.json').read_text(encoding='utf8'))
n1=json.loads((V/'LRC13_LEAN_N1_COST_20261011.json').read_text(encoding='utf8'))
assert n3['status']=='N3_39_POSTHOC_SCORED'
assert n2['status']=='N2_POSTHOC_39_COMPLETE'
assert n1['status']=='N1_POSTHOC_39'
n1sk=n1['stats']['skeleton'];n1a7=n1['stats']['full_lrc13']
scores=n2['counts']
flags=n2['N1_fixed_flags_listen']
summary={
 'N3_zero_parity_39':True,
 'N3_all_scopes':{k:{'PCO200':x['N3']['PCO200'],'gt1':x['N3']['gt1'],
           'new_gt1':len(x['new_gt1']),'moved_rows':x['moved_rows']} for k,x in n3['cases'].items()},
 'N2':scores,
 'N2_new_gt1_vs_A7':len(n2['new_gt1_vs_A7']),
 'N2_new_backtracks':len(n2['new_backtracks_vs_A7']),
 'N1_skeleton_silent_gt1':len(n1sk['silent_wrong_gt1']),
 'N1_skeleton_silent_gt05':len(n1sk['silent_wrong_gt05']),
 'N1_listen_ratio':n1sk['listen_ratio'],
 'N1_N2_silent_gt1':len(n2['N1_fixed_flags_silent_gt1']),
 'N1_N2_silent_gt05':len(n2['N1_fixed_flags_silent_gt05']),
 'N1_N2_listen_ratio':flags/1487,
 'reference_A7_review_ratio':583/1487,
 'reference_A7_wrong_gt1':84,
 'reference_A7_flag_caught_gt1':80
}
decision={'speed_can_repair_safely':all(len(n3['cases'][str(v)+'_flagged_only']['new_gt1'])==0 for v in (.1,.12,.15))
   and any(n3['cases'][str(v)+'_flagged_only']['N3']['gt1']<115 for v in (.1,.12,.15)),
 'ctc_first_improves_over_A7':scores['N2']['PCO200']>=scores['A7']['PCO200']
  and scores['N2']['gt1']<=scores['A7']['gt1']
  and len(n2['new_gt1_vs_A7'])==0,
 'lean_less_silent_error':len(n1sk['silent_wrong_gt1'])<4,
 'lean_lower_listening':n1sk['listen_ratio']<583/1487}
payload={'status':'LEAN_PIPELINE_ALL_TASKS_SCORED_39','summary':summary,'decisions':decision,
 'N3':n3,'N2':n2,'N1':n1,'development_set_only':True,'production_policy_default':'off'}
(V/'LRC13_LEAN_PIPELINE_REPORT_20261011.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# LRC 1.3 Lean Pipeline — N3 / N2 / N1 (2026-10-11)','',
 '39 首開發集，無外部 holdout。BK 僅在 GPU 輸出全部凍結後作事後計分。','',
 '## 結論',
 'N3 三門檻的安全修復：'+str(decision['speed_can_repair_safely'])+'。各方案完整數據見下表；任何新增 >1s 都不得略過。',
 'N2 移除 Whisper/RAW 時間來源後是否優於 A7 且零新增大錯：'+str(decision['ctc_first_improves_over_A7'])+
 '。N2 '+str(scores['N2'])+'，A7 '+str(scores['A7'])+'，main '+str(scores['main'])+'，骨架參照 1314/83。',
 'N1 精簡人工成本：骨架錯 >1s 83 行，沉默 '+str(len(n1sk['silent_wrong_gt1']))+
 ' 行，>0.5s 沉默 '+str(len(n1sk['silent_wrong_gt05']))+
 ' 行；需聆聽 '+str(n1sk['flagged'])+'/1487 ('+str(round(100*n1sk['listen_ratio'],2))+'%)。'+
 '原 A7 review 583/1487，>1s 錯 84 行，原標記抓到 80 行。',
 '',
 '## 1. N3：唱速下限的精確 CTC Viterbi',
 '39/39 首 s_min=0 逐行與 pinned full-song MMS CTC 一致，容差 1 個 emission frame。非零下限以每行首 token 至下一行首 token 的幀距做精確 Viterbi 硬約束；數學上可行的全曲原最佳解直接保留，避免重算。',
 '| 下限 (s/音節) | 套用範圍 | PCO H0→N3 | >1s H0→N3 | 新 >1s | 新倒退 | 移動行數 | 最大移動 |',
 '|---:|---|---:|---:|---:|---:|---:|---:|']
for k,z in n3['cases'].items():
 lines.append('| '+str(z['threshold'])+' | '+z['scope']+' | '+str(z['baseline']['PCO200'])+
 '→'+str(z['N3']['PCO200'])+' | '+str(z['baseline']['gt1'])+'→'+str(z['N3']['gt1'])+
 ' | '+str(len(z['new_gt1']))+' | '+str(len(z['new_backtracks']))+
 ' | '+str(z['moved_rows'])+' | '+str(round(z['max_move'],3))+'s |')
 for r in z['new_gt1']:
  lines.append('- 新 >1s：'+r['song']+' #'+str(r['line'])+
      ' BK='+str(r['BK'])+' H0='+str(r['H0'])+' N3='+str(r['N3']))
 lines.append('- 重點／誤報歌曲：')
 for s in z['songs']:
  if any(q in s['song'] for q in ('Choir','ミレニア','My Dearest','影色舞','夢遊','Ever be my love','レインフォール')):
   lines.append('  - '+s['song']+': '+str(s['H0'])+' → '+str(s['N3'])+'；移動 '+str(s['moved'])+' 行')
lines+=['','## 2. N2：完整流程禁用 Whisper/RAW 候選作為最終時間來源',
 '新的 opt-in CLI：--timing-source-policy ctc-first。省略時為 default，測試確認 default policy 保留原 candidate_sets 的同一物件（不排序、不刪除）。',
 '禁用對象與 B11 分類相同：source 文字包含 raw 或 whisper 的候選。原始證據及標記留存，最終時間選擇排除。',
 '| 版本 | PCO@0.2 | >1s | >0.5s |','|---|---:|---:|---:|']
for k,z in scores.items():
 lines.append('| '+k+' | '+str(z['PCO200'])+' | '+str(z['gt1'])+' | '+str(z['gt05'])+' |')
lines+=['新增 >1s 相較 A7：'+str(len(n2['new_gt1_vs_A7']))+
 ' 行；相較 main：'+str(len(n2['new_gt1_vs_main']))+
 ' 行。新增倒退 '+str(len(n2['new_backtracks_vs_A7']))+'。',
 'N2 trusted '+str(n2['trusted_total'])+
 ' 行，其中 >1s '+str(n2['trusted_wrong_gt1'])+
 ' 行（'+str(n2['trusted_wrong_gt1_ratio'])+'）。',
 '最終來源計數：'+str(n2['source_counts'])]
for a in n2['new_gt1_vs_A7']:
 lines.append('- N2 新大錯：'+a['song']+' #'+str(a['line'])+
 ' BK='+str(a['BK'])+' A7='+str(a['A7'])+' N2='+str(a['N2']))
lines+=['','## 3. N1：精簡流程人工成本',
 '| 輸出時間 | >1s | >0.5s | 需聆聽 | >1s 沉默 | >0.5s 沉默 |',
 '|---|---:|---:|---:|---:|---:|',
 '| 骨架＋六規則 | '+str(n1sk['wrong_gt1'])+' | '+str(n1sk['wrong_gt05'])+
 ' | '+str(n1sk['flagged'])+'/1487 | '+str(len(n1sk['silent_wrong_gt1']))+
 ' | '+str(len(n1sk['silent_wrong_gt05']))+' |',
 '| N2＋六規則 | '+str(scores['N2']['gt1'])+' | '+str(scores['N2']['gt05'])+
 ' | '+str(flags)+'/1487 | '+str(len(n2['N1_fixed_flags_silent_gt1']))+
 ' | '+str(len(n2['N1_fixed_flags_silent_gt05']))+' |',
 '| 原始 1.3 A7 原生標記 | 84 | 未核 | 583/1487 review | 4 | 未核 |',
 '', '六種單項規則召回及聆聽量：']
for k,z in n1sk['per_flag'].items():
 lines.append('- '+k+': 標記 '+str(z['flagged_count'])+
 '，抓到 >1s '+str(z['recall_wrong_gt1'])+'，抓到 >0.5s '+str(z['recall_wrong_gt05']))
lines+=['','### 骨架沉默錯誤 >1 s']
for a in n1sk['silent_wrong_gt1']:
 lines.append('- '+a['song']+' #'+str(a['line'])+
 ' BK='+str(a['BK'])+' skeleton='+str(a['source_time'])+' error='+str(a['err']))
lines+=['','### N2 沉默錯誤 >1 s']
for a in n2['N1_fixed_flags_silent_gt1']:
 lines.append('- '+a['song']+' #'+str(a['line'])+
 ' BK='+str(a['BK'])+' N2='+str(a['N2']))
lines+=['','## 4. 偏差、限制與審查事項',
 '- N3 日文 mora 以現有 CTC romanizer 的母音群近似，不是經過日本語言學詞彙剖析的精確 mora。',
 '- 唱速整首旗標沿用 Opus 0.35 秒／raw lyric vowel-group；該門檻來自開發集 BK 統計，必須以獨立 holdout 驗證。',
 '- 39 首已有同集調校痕跡，所有比較只屬開發集證據，不可宣稱泛化。',
 '- N1 六種標記是固定的 BK-blind 定義；與原 A7 自身 reviewer/untrusted 標記不是同套分類器，人工成本不能混同。',
 '- N2 只改動 opt-in 候選資格，保留原本 Whisper/RAW 推論當作信任證據，故耗時計算未縮短。',
 '- 無推送、無合併 main，未修改音訊或 BK。']
(V/'LRC13_LEAN_PIPELINE_REPORT_20261011.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('LEAN_REPORT_COMPLETED',json.dumps(summary,ensure_ascii=True)[:2100])
