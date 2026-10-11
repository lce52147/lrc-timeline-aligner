# LRC 1.3 Lean Pipeline — N3 / N2 / N1 (2026-10-11)

**N2 完成驗收結論：時間指標略優於精簡時間（PCO 1317 vs 1314；>1s 82 vs 83），但新增 6 行 >1s；原生標記未達標（需聆聽 28.0% vs 41.6%；沉默 >1s 4 vs 3、>0.5s 14 vs 13）。Whisper/RAW 候選於 Central 入口即遭刪除，實測 material alternative 標記 405→72；修改方案僅提出，未改生產程式。**

39 首開發集，無外部 holdout。BK 僅在 GPU 輸出全部凍結後作事後計分。

## 結論
N3 三門檻的安全修復：False。各方案完整數據見下表；任何新增 >1s 都不得略過。
N2 移除 Whisper/RAW 時間來源後是否優於 A7 且零新增大錯：False。N2 {'PCO200': 1317, 'gt1': 82, 'gt05': 106}，A7 {'PCO200': 1295, 'gt1': 84, 'gt05': 120}，main {'PCO200': 1268, 'gt1': 117, 'gt05': 146}，骨架參照 1314/83。
N1 精簡人工成本：骨架錯 >1s 83 行，沉默 6 行，>0.5s 沉默 13 行；需聆聽 757/1487 (50.91%)。原 A7 review 583/1487，>1s 錯 84 行，原標記抓到 80 行。

## 1. N3：唱速下限的精確 CTC Viterbi
39/39 首 s_min=0 逐行與 pinned full-song MMS CTC 一致，容差 1 個 emission frame。非零下限以每行首 token 至下一行首 token 的幀距做精確 Viterbi 硬約束；數學上可行的全曲原最佳解直接保留，避免重算。
| 下限 (s/音節) | 套用範圍 | PCO H0→N3 | >1s H0→N3 | 新 >1s | 新倒退 | 移動行數 | 最大移動 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | flagged_only | 1285→1283 | 115→117 | 2 | 0 | 150 | 18.501s |
- 新 >1s：01.ミレニアの水槽 #22 BK=116.06 H0=116.065 N3=120.125122
- 新 >1s：01.ミレニアの水槽 #23 BK=120.03 H0=120.025 N3=122.105206
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 16, 'gt1': 22}；移動 33 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 18, 'gt1': 32}；移動 43 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 30, 'gt1': 12}；移動 38 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 20, 'gt1': 23}；移動 36 行
| 0.1 | all_39_diagnostic | 1285→1282 | 115→119 | 4 | 0 | 210 | 18.501s |
- 新 >1s：01.ミレニアの水槽 #22 BK=116.06 H0=116.065 N3=120.125122
- 新 >1s：01.ミレニアの水槽 #23 BK=120.03 H0=120.025 N3=122.105206
- 新 >1s：06. あの世行きのバスに乗ってさらば。 #3 BK=11.9 H0=11.901 N3=13.481584
- 新 >1s：12. 終点の先が在るとするならば。 #28 BK=127.34 H0=127.628 N3=128.507898
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 16, 'gt1': 22}；移動 33 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 18, 'gt1': 32}；移動 43 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 30, 'gt1': 12}；移動 38 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 20, 'gt1': 23}；移動 36 行
| 0.12 | flagged_only | 1285→1287 | 115→112 | 2 | 0 | 157 | 42.585s |
- 新 >1s：04 影色舞 #9 BK=43.7 H0=43.625 N3=40.884305
- 新 >1s：04 影色舞 #13 BK=56.68 H0=56.706 N3=45.344775
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 17, 'gt1': 21}；移動 36 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 18, 'gt1': 31}；移動 48 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 32, 'gt1': 10}；移動 37 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 21, 'gt1': 22}；移動 36 行
| 0.12 | all_39_diagnostic | 1285→1286 | 115→113 | 4 | 0 | 327 | 42.585s |
- 新 >1s：04 影色舞 #9 BK=43.7 H0=43.625 N3=40.884305
- 新 >1s：04 影色舞 #13 BK=56.68 H0=56.706 N3=45.344775
- 新 >1s：06. あの世行きのバスに乗ってさらば。 #3 BK=11.9 H0=11.901 N3=13.481584
- 新 >1s：12. 終点の先が在るとするならば。 #28 BK=127.34 H0=127.628 N3=128.507898
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 17, 'gt1': 21}；移動 36 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 18, 'gt1': 31}；移動 48 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 32, 'gt1': 10}；移動 37 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 21, 'gt1': 22}；移動 36 行
| 0.15 | flagged_only | 1285→1286 | 115→113 | 3 | 0 | 155 | 33.284s |
- 新 >1s：04 影色舞 #25 BK=100.61 H0=100.631 N3=102.530797
- 新 >1s：01.ミレニアの水槽 #22 BK=116.06 H0=116.065 N3=120.825152
- 新 >1s：01.ミレニアの水槽 #23 BK=120.03 H0=120.025 N3=122.525224
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 18, 'gt1': 20}；移動 36 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 19, 'gt1': 31}；移動 45 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 29, 'gt1': 12}；移動 38 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 21, 'gt1': 22}；移動 36 行
| 0.15 | all_39_diagnostic | 1285→1284 | 115→112 | 5 | 0 | 326 | 33.284s |
- 新 >1s：04 影色舞 #25 BK=100.61 H0=100.631 N3=102.530797
- 新 >1s：01. ふり #30 BK=85.87 H0=85.885 N3=84.024485
- 新 >1s：01.ミレニアの水槽 #22 BK=116.06 H0=116.065 N3=120.825152
- 新 >1s：01.ミレニアの水槽 #23 BK=120.03 H0=120.025 N3=122.525224
- 新 >1s：06. あの世行きのバスに乗ってさらば。 #3 BK=11.9 H0=11.901 N3=13.481584
- 重點／誤報歌曲：
  - 03. Choir ‘S’ Choir: {'PCO200': 16, 'gt1': 22} → {'PCO200': 18, 'gt1': 20}；移動 36 行
  - 04 影色舞: {'PCO200': 18, 'gt1': 32} → {'PCO200': 19, 'gt1': 31}；移動 45 行
  - Ever be my love: {'PCO200': 18, 'gt1': 1} → {'PCO200': 18, 'gt1': 1}；移動 0 行
  - 01.ミレニアの水槽: {'PCO200': 32, 'gt1': 10} → {'PCO200': 29, 'gt1': 12}；移動 38 行
  - 09. ツユ - レインフォール: {'PCO200': 71, 'gt1': 0} → {'PCO200': 71, 'gt1': 0}；移動 0 行
  - 09.夢遊: {'PCO200': 25, 'gt1': 4} → {'PCO200': 25, 'gt1': 4}；移動 0 行
  - 01 My Dearest: {'PCO200': 20, 'gt1': 23} → {'PCO200': 21, 'gt1': 22}；移動 36 行

## 2. N2：完整流程禁用 Whisper/RAW 候選作為最終時間來源
新的 opt-in CLI：--timing-source-policy ctc-first。省略時為 default，測試確認 default policy 保留原 candidate_sets 的同一物件（不排序、不刪除）。
禁用對象與 B11 分類相同：source 文字包含 raw 或 whisper 的候選。**初版實作雖保留部分前置診斷，Central 決策前已刪除候選，material-alternative 標記實測大量流失；詳見文末驗收補充。**
| 版本 | PCO@0.2 | >1s | >0.5s |
|---|---:|---:|---:|
| CTC | 1285 | 115 | 135 |
| main | 1268 | 117 | 146 |
| A7 | 1295 | 84 | 120 |
| N2 | 1317 | 82 | 106 |
新增 >1s 相較 A7：15 行；相較 main：15 行。新增倒退 1。
N2 trusted 741 行，其中 >1s 3 行（0.004048582995951417）。
最終來源計數：{'ctc-current': 1343, 'ctc-vocal-activity-reentry': 52, 'ctc-opening-local-retry': 6, 'ctc-local-window': 32, 'global-acoustic-recurrence': 11, 'ctc-vocal-onset-confirmed': 12, 'mutual-acoustic-recurrence': 10, 'ctc-joint-boundary-consensus': 17, 'audio-validated-current': 4}
- N2 新大錯：06.ビビッド #47 BK=174.8 A7=174.16 N2=172.63
- N2 新大錯：01. Elegy Dedicated With Love #32 BK=228.81 A7=228.8 N2=227.02
- N2 新大錯：01.ミレニアの水槽 #15 BK=89.52 A7=89.08 N2=83.44
- N2 新大錯：01.ミレニアの水槽 #16 BK=93.94 A7=93.9 N2=90.5
- N2 新大錯：01.ミレニアの水槽 #32 BK=177.01 A7=176.99 N2=173.21
- N2 新大錯：01. 雨模様 #24 BK=72.79 A7=72.78 N2=74.57
- N2 新大錯：09. ツユ - レインフォール #70 BK=201.61 A7=201.68 N2=199.2
- N2 新大錯：12. ツユ - 朧月夜物語 #9 BK=86.62 A7=86.64 N2=72.21
- N2 新大錯：12. 終点の先が在るとするならば。 #6 BK=18.59 A7=18.71 N2=17.4
- N2 新大錯：12. 終点の先が在るとするならば。 #19 BK=72.82 A7=72.96 N2=71.58
- N2 新大錯：12. 終点の先が在るとするならば。 #27 BK=123.24 A7=123.24 N2=126.07
- N2 新大錯：01 My Dearest #19 BK=132.94 A7=132.94 N2=150.08
- N2 新大錯：01 My Dearest #20 BK=139.02 A7=139.02 N2=161.46
- N2 新大錯：01 My Dearest #26 BK=175.56 A7=175.56 N2=176.6
- N2 新大錯：01 My Dearest #37 BK=270.0 A7=269.96 N2=268.89

## 3. N1：精簡流程人工成本
| 輸出時間 | >1s | >0.5s | 需聆聽 | >1s 沉默 | >0.5s 沉默 |
|---|---:|---:|---:|---:|---:|
| 骨架＋六規則 | 83 | 105 | 757/1487 | 6 | 13 |
| N2＋六規則 | 82 | 106 | 757/1487 | 5 | 13 |
| 原始 1.3 A7 原生標記 | 84 | 未核 | 583/1487 review | 4 | 未核 |

六種單項規則召回及聆聽量：
- CTC_score_lt_008: 標記 374，抓到 >1s 67，抓到 >0.5s 79
- repeated_lyric: 標記 412，抓到 >1s 33，抓到 >0.5s 38
- compression_pair: 標記 43，抓到 >1s 28，抓到 >0.5s 30
- CTC_low_run: 標記 84，抓到 >1s 22，抓到 >0.5s 26
- F5_repeat_global_diff: 標記 11，抓到 >1s 0，抓到 >0.5s 0
- whole_song_speed_flag: 標記 282，抓到 >1s 56，抓到 >0.5s 65

### 骨架沉默錯誤 >1 s
- 01. The Day the Starlight Arrived 星辉抵达之日 #1 BK=1.0 skeleton=2.26 error=1.26
- 06.ビビッド #1 BK=16.3 skeleton=17.362 error=1.062
- 06.ビビッド #30 BK=120.3 skeleton=117.611 error=2.689
- 01. Elegy Dedicated With Love #36 BK=258.94 skeleton=261.701 error=2.761
- 02. JANE DOE #14 BK=113.49 skeleton=111.245 error=2.245
- 12. ツユ - 朧月夜物語 #17 BK=143.43 skeleton=141.716 error=1.714

### N2 沉默錯誤 >1 s
- 01. The Day the Starlight Arrived 星辉抵达之日 #1 BK=1.0 N2=2.26
- 01. Elegy Dedicated With Love #36 BK=258.94 N2=261.7
- 02. JANE DOE #14 BK=113.49 N2=111.25
- 05.遠雷 #20 BK=135.46 N2=132.68
- 12. ツユ - 朧月夜物語 #17 BK=143.43 N2=141.72

## 4. 偏差、限制與審查事項
- N3 日文 mora 以現有 CTC romanizer 的母音群近似，不是經過日本語言學詞彙剖析的精確 mora。
- 唱速整首旗標沿用 Opus 0.35 秒／raw lyric vowel-group；該門檻來自開發集 BK 統計，必須以獨立 holdout 驗證。
- 39 首已有同集調校痕跡，所有比較只屬開發集證據，不可宣稱泛化。
- N1 六種標記是固定的 BK-blind 定義；與原 A7 自身 reviewer/untrusted 標記不是同套分類器，人工成本不能混同。
- N2 只改動 opt-in 候選資格，保留原本 Whisper/RAW 推論當作信任證據，故耗時計算未縮短。
- 無推送、無合併 main，未修改音訊或 BK。

## N2 完成後的正式驗收補充（2026-10-11）

**N2 時間 PCO 1317、>1s 82；精簡時間 PCO 1314、>1s 83。N2 標記需聆聽 28.0%、沉默 >1s 4、>0.5s 14；基準精簡時間＋A7 標記 41.6%／3／13。標記未達標。**

39 首為開發集；N2 相較 A7 新增 >1s 行 15，相較精簡新增 6 行，不能稱為逐行無退化。

### 時間與標記比較
| 時間＋標記 | PCO | >1s | >0.5s | 需聆聽 | 沉默 >1s | 沉默 >0.5s |
|---|---:|---:|---:|---:|---:|---:|
| 精簡時間＋A7 標記 | 1314 | 83 | 105 | 618/1487 (41.6%) | 3 | 13 |
| 精簡時間＋N2 標記 | 1314 | 83 | 105 | 416/1487 (28.0%) | 4 | 13 |
| N2＋N2 標記 | 1317 | 82 | 106 | 416/1487 (28.0%) | 4 | 14 |
| N2＋A7 標記 | 1317 | 82 | 106 | 618/1487 (41.6%) | 7 | 17 |
| A7＋A7 標記 | 1295 | 84 | 120 | 618/1487 (41.6%) | 4 | 19 |
| 精簡時間＋N1 六規則 | 1314 | 83 | 105 | 757/1487 (50.9%) | 6 | 13 |
| N2＋N1 六規則 | 1317 | 82 | 106 | 757/1487 (50.9%) | 5 | 13 |
| main＋A7 標記 | 1268 | 117 | 146 | 618/1487 (41.6%) | 5 | 19 |

### N2 相較精簡時間新增 >1s（逐行）
| 歌曲 | 行 | BK | 精簡時間 | N2 |
|---|---:|---:|---:|---:|
| 01. Elegy Dedicated With Love | 32 | 228.81 | 229.278 | 227.02 |
| 05.遠雷 | 20 | 135.46 | 135.515 | 132.68 |
| 09. ツユ - レインフォール | 70 | 201.61 | 201.701 | 199.2 |
| 12. 終点の先が在るとするならば。 | 6 | 18.59 | 18.721 | 17.4 |
| 12. 終点の先が在るとするならば。 | 19 | 72.82 | 72.764 | 71.58 |
| 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian | 24 | 156.63 | 157.055 | 157.89 |

### N2 相較精簡時間新增倒退
新增倒退共 1 處：[{"song": "12. ツユ - 朧月夜物語", "line": 8}]。


### 逐首原生標記差異
| 歌曲 | A7 標記 | N2 標記 | 差異 | 流失 | 新增 |
|---|---:|---:|---:|---:|---:|
| 01. The Day the Starlight Arrived 星辉抵达之日 | 13 | 16 | +3 | 2 | 5 |
| 01.エガクミライ | 7 | 1 | -6 | 6 | 0 |
| 01.感情グラス | 12 | 6 | -6 | 8 | 2 |
| 03. Choir ‘S’ Choir | 35 | 38 | +3 | 0 | 3 |
| 03. 過去に囚われている | 16 | 8 | -8 | 8 | 0 |
| 04 影色舞 | 39 | 35 | -4 | 5 | 1 |
| 06.ビビッド | 35 | 27 | -8 | 11 | 3 |
| 10.方舟 | 5 | 3 | -2 | 3 | 1 |
| 11.ユートピア | 7 | 0 | -7 | 7 | 0 |
| Ever be my love | 15 | 9 | -6 | 8 | 2 |
| 01 - フロム | 8 | 5 | -3 | 3 | 0 |
| 01. Elegy Dedicated With Love | 11 | 4 | -7 | 8 | 1 |
| 01.おやすみモノクローム | 19 | 5 | -14 | 15 | 1 |
| 01. それでも雨は降るんだね | 30 | 19 | -11 | 12 | 1 |
| 01. ふり | 25 | 22 | -3 | 4 | 1 |
| 01. やっぱり雨は降るんだね | 22 | 10 | -12 | 15 | 3 |
| 01.ミレニアの水槽 | 28 | 26 | -2 | 9 | 7 |
| 01.最大幸福度 | 22 | 11 | -11 | 15 | 4 |
| 01. 芽吹くとき | 9 | 6 | -3 | 3 | 0 |
| 01. 雨模様 | 19 | 10 | -9 | 9 | 0 |
| 02. Georgette Me, Georgette You | 10 | 9 | -1 | 1 | 0 |
| 02. JANE DOE | 13 | 3 | -10 | 10 | 0 |
| 03.春を待つ (feat. 倚水) | 6 | 3 | -3 | 3 | 0 |
| 04. Scarborough Fair | 4 | 5 | +1 | 1 | 2 |
| 04.可惜夜 | 4 | 1 | -3 | 3 | 0 |
| 05.遠雷 | 8 | 4 | -4 | 5 | 1 |
| 06. あの世行きのバスに乗ってさらば。 | 17 | 8 | -9 | 10 | 1 |
| 08.月の唄 | 16 | 10 | -6 | 7 | 1 |
| 08 春日影 (MyGO!!!!! Ver.) | 12 | 3 | -9 | 10 | 1 |
| 09. ツユ - レインフォール | 29 | 17 | -12 | 13 | 1 |
| 10.深夜特急 | 3 | 0 | -3 | 3 | 0 |
| 12. ツユ - 朧月夜物語 | 11 | 7 | -4 | 4 | 0 |
| 12.琥珀の国 | 12 | 3 | -9 | 9 | 0 |
| 12. 終点の先が在るとするならば。 | 15 | 12 | -3 | 7 | 4 |
| 13.薔薇の下で | 12 | 3 | -9 | 9 | 0 |
| 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian | 8 | 9 | +1 | 0 | 1 |
| 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki | 19 | 15 | -4 | 8 | 4 |
| 09.夢遊 | 12 | 12 | +0 | 4 | 4 |
| 01 My Dearest | 30 | 31 | +1 | 2 | 3 |

流失標記 260 行，新增標記 58 行；逐首詳細原因差異、流失位置、所有沉默錯誤見 JSON N2_finish_audit。

### 標記原因差異（逐行 flag/failure/review_reason）
| 類型 | 名稱 | A7 | N2 | 差異 |
|---|---|---:|---:|---:|
| flags | candidate_disagreement | 159 | 159 | +0 |
| flags | unresolved-material-alternative | 405 | 72 | -333 |
| flags | F1-vocal-absent | 7 | 9 | +2 |
| flags | F2-repeated-content-identity | 19 | 0 | -19 |
| flags | F3-compressed-or-nonmonotonic | 45 | 45 | +0 |
| flags | F3-nonmonotonic | 2 | 2 | +0 |
| flags | F4-ctc-final-disagreement | 72 | 45 | -27 |
| flags | ctc_zero_gap_compressed_prefix | 10 | 10 | +0 |
| flags | duplicate_occurrence_offset_mismatch | 5 | 5 | +0 |
| flags | insufficient-minimum-proof | 151 | 261 | +110 |
| flags | long_line_disagreement | 22 | 22 | +0 |
| flags | no-strict-opening-proof | 121 | 215 | +94 |
| flags | nonmonotonic-line-start | 1 | 1 | +0 |
| flags | phonetic_anchor_disagreement | 35 | 35 | +0 |
| flags | previous-tail-overlap | 9 | 14 | +5 |
| flags | previous-tail-ownership-unknown | 128 | 211 | +83 |
| flags | short_line_onset_uncertain | 5 | 5 | +0 |
| flags | skeleton_global_ctc_disagreement | 11 | 11 | +0 |
| flags | tail-conflict-without-independent-current-opening | 137 | 225 | +88 |
| flags | temporal-incoherence | 25 | 32 | +7 |
| failures | unresolved-material-alternative | 405 | 72 | -333 |
| failures | F1-vocal-absent | 7 | 9 | +2 |
| failures | F2-repeated-content-identity | 19 | 0 | -19 |
| failures | F3-compressed-or-nonmonotonic | 45 | 45 | +0 |
| failures | F3-nonmonotonic | 2 | 2 | +0 |
| failures | F4-ctc-final-disagreement | 72 | 45 | -27 |
| failures | ctc_zero_gap_compressed_prefix | 10 | 10 | +0 |
| failures | insufficient-minimum-proof | 151 | 261 | +110 |
| failures | no-strict-opening-proof | 121 | 215 | +94 |
| failures | nonmonotonic-line-start | 1 | 1 | +0 |
| failures | previous-tail-overlap | 9 | 14 | +5 |
| failures | previous-tail-ownership-unknown | 128 | 211 | +83 |
| failures | skeleton_global_ctc_disagreement | 11 | 11 | +0 |
| failures | tail-conflict-without-independent-current-opening | 137 | 225 | +88 |
| failures | temporal-incoherence | 25 | 32 | +7 |
| review_reasons | unresolved-material-alternative | 405 | 72 | -333 |
| review_reasons | actionable-suspicious-conflict | 209 | 169 | -40 |
| review_reasons | actionable-timing-failure | 42 | 50 | +8 |
| review_reasons | final-self-check | 132 | 97 | -35 |

### B11 最終來源淨效益（CTC 對 N2）
B11 原分類：CTC/local、Whisper/RAW、reviewer、Phase4B、vocal-onset、其他 source；時間移動 >0.05s 才計，改善／惡化 >0.2s。
| 層 | 改動 | 改善 | 惡化 | 淨改善 | 新增 >1s | 修復 >1s |
|---|---:|---:|---:|---:|---:|---:|
| CTC/local | 48 | 39 | 5 | +34 | 1 | 36 |
| mutual-acoustic-recurrence | 8 | 3 | 5 | -2 | 4 | 2 |
| global-acoustic-recurrence | 3 | 1 | 1 | +0 | 1 | 1 |

### 程式層級標記流失與修法（僅提案，未實作）
scripts/auto_lrc.py:26158 apply_central_timing_source_policy 在 evaluate_backend_timing（26185）入口刪除 Whisper/RAW candidates；build_final_timing_state（4198）以及 select_timing_decision（3875）無法保留這些候選於 rejected_candidates。_final_post_central_material_conflict_entries（24032）看 decision graph，因此其 unresolved-material-alternative 路徑會失去已刪除來源的證據。commit_final_timing_state（24801）又依此產生 audit/review 標記。前置 candidate_disagreement 由 add_alignment_diagnostics 及 report flags 生成的部分可能繼續存在，需看上表實測數字。
提案：保留完整候選供不一致判斷；僅在 select_timing_decision 最終 winner 資格處套用 ctc-first，禁止 Whisper/RAW 決定時間，但保留其 rejected alternatives 和相應證據；透過 build_final_timing_state 及 rerun 路徑傳遞 policy。補上 identity、禁用來源、衝突標記保留及 39 首回歸測試後交 Opus 審查。本輪不修改生產程式。

### N1／N3：Opus 審查結論與來源
直接引用 _review/OPUS_LEAN_N1_N3_REVIEW_20261011.md：N3 s_min=0 的 39 首同位測試通過；僅對旗標歌使用 0.10／0.12／0.15 門檻時 >1s 分別 117／112／113，各新增 2／2／3 行 >1s，未能安全修正壓縮錯位。N1 六規則：聆聽 50.9%、沉默 >1s 6 行、>0.5s 13 行；精簡時間＋A7 原標記：聆聽 41.6%、沉默 >1s 3 行、>0.5s 13 行。39 首屬開發集，無獨立 holdout。
