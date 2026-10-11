from pathlib import Path
import json
v=Path(__file__).resolve().parent
g=json.loads((v/'LRC13_ANCHOR_POSTHOC_20261011.json').read_text(encoding='utf8'))
b=json.loads((v/'LRC13_ANCHOR_B11_OFFLINE_20261011.json').read_text(encoding='utf8'))
a=json.loads((v/'LRC13_ANCHOR_ASR_LYRIC_TRIAGE_20261011.json').read_text(encoding='utf8'))
assert g['status']=='ANCHOR_39_POSTHOC_SCORED' and g['count']==39
d=g['delta']
danger=g['anchor_gt1']
b11_ok=b['score']['CTC_s']['PCO200']>b['score']['main_s']['PCO200'] and b['score']['CTC_s']['wrong_gt1']<b['score']['main_s']['wrong_gt1']
summary={'status':'ANCHOR_ARCH_DEVSET_39_EVALUATED',
 'B11_main_net_negative':b11_ok,'B11':b['score'],
 'B11_by_layer':{k:z['layer_counts'] for k,z in b['analysis'].items()},
 'anchor_total':g['anchor_total'],'anchor_gt1_count':len(danger),
 'anchor_gt1':danger,'anchor_PCO200':g['anchor_PCO200'],
 'anchor_PCO500':g['anchor_PCO500'],
 'anchor_interval_status':g['interval_statuses'],
 'delta':{k:{key:z[key] for key in ('H0','H1','main','new_gt1','new_backtracks','max_move_s','active_intervals','active_interval_worse','moved_rows','repeat','nonrepeat')} for k,z in d.items()},
 'songs':g['songs'],'focus_lines':g['focus_lines'],'asr_triage':a,
 'only_dev_set':True,'BK_posthoc_only':True,'no_production_change':True}
(v/'LRC13_ANCHOR_ARCH_REPORT_20261011.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# LRC 1.3 — 錨點架構驗證（2026-10-11）','',
 '本次僅測試 39 首開發集，不宣稱泛化效能。所有 δ 在 BK 計分前已固定。','',
 '## 先回答三個問題',
 'B11 main：'+('成立，完整後處理淨退步。' if b11_ok else '不成立。')+
 ' 原始 CTC '+str(b['score']['CTC_s'])+'，main '+str(b['score']['main_s'])+'，1.3 '+str(b['score']['lrc13_s'])+'。',
 '錨點：'+str(g['anchor_total'])+' 行，PCO@0.2 '+str(g['anchor_PCO200'])+
 '、PCO@0.5 '+str(g['anchor_PCO500'])+'，超過 1 秒錯錨 '+str(len(danger))+' 行。',
 '原型：δ=.5/2/5 啟用區間 '+str([d[k]['active_intervals'] for k in ('0.5','2.0','5.0')])+
 '，新增 >1s '+str([len(d[k]['new_gt1']) for k in ('0.5','2.0','5.0')])+'。',
 '']
if danger:
 lines+=['### 重要：錨點本身超過 1 秒的錯行',
 '| 歌曲 | 行 | BK | CTC 時間 | 誤差 | CTC 分數 |',
 '|---|---:|---:|---:|---:|---:|']
 for x in danger:
  lines.append('| '+x['song']+' | '+str(x['line'])+' | '+str(x['BK'])+' | '+str(x['CTC'])+' | '+str(x['error'])+' | '+str(x['ctc_score'])+' |')
lines+=['','## 1. B11 共通基準核對',
 'CTC 起點取 pinned main align-report 的 ctc_first_token_start，與 S5 使用相同歌詞行過濾及 PCO@0.2 含 1e-9 容差。',
 '| 最終 | 來源層 | 改動行數 | 變好 >0.2s | 變壞 >0.2s | 新 >1s | 修好 >1s |',
 '|---|---|---:|---:|---:|---:|---:|']
for version,z in b['analysis'].items():
 for layer,q in z['layer_counts'].items():
  lines.append('| '+version+' | '+layer+' | '+str(q['changed'])+' | '+str(q['improved'])+
   ' | '+str(q['worsened'])+' | '+str(q['new_wrong_gt1'])+' | '+str(q['repaired_gt1'])+' |')
lines+=['','## 2. 錨點精確度與覆蓋',
 'CTC 錯誤 >1s 總行數 '+str(g['ctc_wrong_gt1_total'])+'；其中錯錨行數 '+
 str(g['ctc_wrong_gt1_anchor'])+'，非錨點區間內 '+str(g['ctc_wrong_gt1_inside_gaps'])+'。',
 '區間狀態 '+str(g['interval_statuses'])+'。',
 '| 歌曲 | 行數 | 錨點數 | >1s 錯錨 |','|---|---:|---:|---:|']
for q in g['songs']:
 lines.append('| '+q['song'].replace('|','/')+' | '+str(q['rows'])+' | '+str(q['anchor_count'])+
 ' | '+str(sum(abs(x['error'])>1 for x in q['anchors']))+' |')
lines+=['','## 3. 三組 δ 的獨立評分',
 '| δ | PCO H0→H1 | >1 s H0→H1 | 新 >1 s | 新倒退 | 啟用區間 | 啟用後變差區間 | 最大移動 |',
 '|---|---|---|---:|---:|---:|---:|---:|']
for k,q in d.items():
 lines.append('| '+k+' | '+str(q['H0']['PCO200'])+'→'+str(q['H1']['PCO200'])+
 ' | '+str(q['H0']['wrong_gt1'])+'→'+str(q['H1']['wrong_gt1'])+
 ' | '+str(len(q['new_gt1']))+' | '+str(len(q['new_backtracks']))+' | '+
 str(q['active_intervals'])+' | '+str(len(q['active_interval_worse']))+
 ' | '+str(round(q['max_move_s'],3))+'s |')
 if q['new_gt1']:
  lines.append('δ='+k+' 新錯行：')
  for x in q['new_gt1']:
   lines.append('- '+x['song']+' #'+str(x['line'])+' BK='+str(x['BK'])+' H0='+str(x['H0'])+' H1='+str(x['H1']))
lines+=['','## 4. 五首重點逐行（δ=0.5）',
 '| 歌曲 | 行 | BK | H0 | H1 | 移動 | review |','|---|---:|---:|---:|---:|---:|---|']
for x in g['focus_lines']:
 lines.append('| '+x['song'].replace('|','/')+' | '+str(x['line'])+' | '+str(x['BK'])+
 ' | '+str(x['H0'])+' | '+str(round(x['H1_selected'],3))+' | '+
 str(x['movement_s'])+' | '+str(x['review_required'])+' |')
lines+=['','## 5. 局部 CTC 提早偏差']
for k,q in d.items():
 moved=[x for x in g['moved_lines'] if str(x['delta'])==k]
 ahead=[x for x in moved if x['movement_s']<0]
 behind=[x for x in moved if x['movement_s']>0]
 lines.append('- δ='+k+'：移動 '+str(len(moved))+' 行，向前 '+str(len(ahead))+' 行，向後 '+str(len(behind))+' 行。')
lines+=['','## 6. 三首整曲歌詞／Whisper 轉錄初查']
for z in a:
 lines+=['### '+z['song'],
 '- raw ASR 事件 '+str(z['ASR_events'])+'，歌詞 '+str(z['lyric_rows'])+
 ' 行，文字弱匹配 '+str(len(z['weak_ASR_matching_lines']))+
 ' 行，原始 CTC >1s '+str(len(z['large_CTC_errors']))+' 行。']
 for x in z['large_CTC_errors'][:12]:
  lines.append('- #'+str(x['line'])+'：CTC 誤差 '+str(x['CTC_abs_error_posthoc'])+
   's，ASR 相似度 '+str(x['similarity'])+
   '，ASR 匹配 @'+str(x['match_start'])+'s，'+x['label'])
 lines.append('以上只是 Whisper 文字一致性線索：重複副歌、多行對單事件及 ASR 幻覺都可能造成假警報，不能直接宣告歌詞有誤。')
lines+=['','## 7. 偏差、限制與處理',
 '- 與全曲 CTC 使用同一份 MMS emission，H0 固定起音時鐘的 constrained CTC Viterbi；H1 在同一視窗內不限制起音，兩者同目標函數。',
 '- 若 constrained H0 失敗，保留全曲 CTC 時間並標記人工審查，不生成假分數。',
 '- 原始 CTC 從 archived pinned main report 取得，非最新主程式重跑；聲學 emission 在此原型重新產生。',
 '- BK 完全不進入 GPU 推論；39 個結果凍結後才事後評分。',
 '- 39 首已用於研發，需另取新 holdout 才能談泛化。',
 '- 未修改生產 scripts、音訊或 BK，不推送、不合併 main。']
(v/'LRC13_ANCHOR_ARCH_REPORT_20261011.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('ANCHOR_REPORT_FINISHED',json.dumps({'B11':b11_ok,'anchor_gt1':len(danger),
 'delta':{k:{'active':q['active_intervals'],'new_gt1':len(q['new_gt1'])} for k,q in d.items()}}),flush=True)
