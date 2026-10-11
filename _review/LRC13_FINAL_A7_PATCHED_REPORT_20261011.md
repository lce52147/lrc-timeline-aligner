# LRC 1.3 A7 — 39 首正式回歸（2026-10-10）

**正式 gate：NOT_PASS**
A6：PASS_PATCH7_AB，14/14；繼承倒退 []

## 全池比較
PCO@0.2：1268 → 1295 / 1487
>1s：117 → 84；新增 >1s 3
可信行 >1s 比例：0.0066445 → 0.0046584
啟用 2/39（5.1%）；退回 37 首
新增倒退 0；INHERITED_BACKTRACK 1；檢核：{'PCO_non_decreasing': True, 'gt1_strictly_decreasing': True, 'no_new_gt1s': False, 'no_new_backtracks': True, 'trusted_wrong_ratio_decreasing': True}
重複歌詞：{'rows': 414, 'main_gt1': 50, 'segment_gt1': 37}；非重複：{'rows': 1073, 'main_gt1': 67, 'segment_gt1': 47}
舊 A7：PCO@0.2 1241/1487；>1s 127
啟用歌曲錯行標記召回：1/1
啟用卻變差：05.遠雷

## 39 首逐首
| 歌 | 方案／原因 | PCO main→A7 | >1s main→A7 | 新 >1s | 新倒退／繼承 | DP margin |
|---|---|---:|---:|---:|---:|---:|
| 01. The Day the Starlight Arrived 星辉抵达之日 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 15→15/24 | 4→4 | 0 | 0/0 | 未提供 |
| 01.エガクミライ | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 27→27/27 | 0→0 | 0 | 0/0 | 未提供 |
| 01.感情グラス | FALLBACK_CTC / STRUCTURAL_DP_AMBIGUITY | 40→40/40 | 0→0 | 0 | 0/0 | 未提供 |
| 03. Choir ‘S’ Choir | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 17→17/44 | 21→21 | 0 | 0/0 | 未提供 |
| 03. 過去に囚われている | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 65→65/68 | 1→0 | 0 | 0/0 | 未提供 |
| 04 影色舞 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 20→45/52 | 28→0 | 0 | 0/0 | 0.099418 |
| 06.ビビッド | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 37→38/55 | 2→0 | 0 | 0/0 | 未提供 |
| 10.方舟 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 33→33/37 | 0→0 | 0 | 0/0 | 未提供 |
| 11.ユートピア | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 32→32/32 | 0→0 | 0 | 0/0 | 未提供 |
| Ever be my love | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 16→15/24 | 3→2 | 0 | 0/0 | 未提供 |
| 01 - フロム | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 32→33/35 | 0→0 | 0 | 0/0 | 未提供 |
| 01. Elegy Dedicated With Love | FALLBACK_CTC / STRUCTURAL_DP_AMBIGUITY | 36→36/37 | 1→1 | 0 | 0/0 | 未提供 |
| 01.おやすみモノクローム | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 35→34/40 | 2→2 | 0 | 0/0 | 未提供 |
| 01. それでも雨は降るんだね | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 54→54/55 | 0→0 | 0 | 0/0 | 未提供 |
| 01. ふり | FALLBACK_CTC / TOO_FEW_PASSAGES | 55→55/57 | 0→0 | 0 | 0/0 | 未提供 |
| 01. やっぱり雨は降るんだね | FALLBACK_CTC / TOO_FEW_PASSAGES | 67→67/67 | 0→0 | 0 | 0/0 | 未提供 |
| 01.ミレニアの水槽 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 27→28/45 | 12→12 | 0 | 0/1 | 未提供 |
| 01.最大幸福度 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 40→40/45 | 0→0 | 0 | 0/0 | 未提供 |
| 01. 芽吹くとき | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 25→25/25 | 0→0 | 0 | 0/0 | 未提供 |
| 01. 雨模様 | FALLBACK_CTC / TOO_FEW_PASSAGES | 44→44/49 | 0→0 | 0 | 0/0 | 未提供 |
| 02. Georgette Me, Georgette You | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 13→13/18 | 2→2 | 0 | 0/0 | 未提供 |
| 02. JANE DOE | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 22→21/24 | 1→1 | 0 | 0/0 | 未提供 |
| 03.春を待つ (feat. 倚水) | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 39→39/39 | 0→0 | 0 | 0/0 | 未提供 |
| 04. Scarborough Fair | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 12→12/14 | 2→2 | 0 | 0/0 | 未提供 |
| 04.可惜夜 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 23→23/24 | 0→0 | 0 | 0/0 | 未提供 |
| 05.遠雷 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 27→26/28 | 0→1 | 1 | 0/0 | 0.09216 |
| 06. あの世行きのバスに乗ってさらば。 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 34→34/40 | 2→3 | 1 | 0/0 | 未提供 |
| 08.月の唄 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 35→35/38 | 1→1 | 0 | 0/0 | 未提供 |
| 08 春日影 (MyGO!!!!! Ver.) | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 36→36/39 | 1→1 | 0 | 0/0 | 未提供 |
| 09. ツユ - レインフォール | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 72→70/74 | 1→2 | 1 | 0/0 | 未提供 |
| 10.深夜特急 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 41→41/41 | 0→0 | 0 | 0/0 | 未提供 |
| 12. ツユ - 朧月夜物語 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 21→21/23 | 1→1 | 0 | 0/0 | 未提供 |
| 12.琥珀の国 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 31→31/32 | 0→0 | 0 | 0/0 | 未提供 |
| 12. 終点の先が在るとするならば。 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 27→27/32 | 3→3 | 0 | 0/0 | 未提供 |
| 13.薔薇の下で | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 29→30/30 | 0→0 | 0 | 0/0 | 未提供 |
| 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 26→26/27 | 1→1 | 0 | 0/0 | 未提供 |
| 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 22→22/33 | 0→0 | 0 | 0/0 | 未提供 |
| 09.夢遊 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 22→24/30 | 6→4 | 0 | 0/0 | 未提供 |
| 01 My Dearest | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 19→21/43 | 22→20 | 0 | 0/0 | 未提供 |

## 新增 >1 秒逐行（交使用者判定）
- 05.遠雷 #20：BK 135.46，main 135.49，segment 132.68；SEGMENT_CURRENT；**待使用者判定**
- 06. あの世行きのバスに乗ってさらば。 #32：BK 151.95，main 151.92，segment 147.00；FALLBACK_CTC；**待使用者判定**
- 09. ツユ - レインフォール #52：BK 144.63，main 144.62，segment 142.42；FALLBACK_CTC；**待使用者判定**

## INHERITED_BACKTRACK
- 01.ミレニアの水槽：相鄰行起點 [16]；同一對 pinned main 已有倒退。

## 最差三首逐行
### 03. Choir ‘S’ Choir（21 行 >1s）
| 行 | BK | main | segment | main 誤差 | segment 誤差 |
|---:|---:|---:|---:|---:|---:|
| 2 | 8.31 | 25.58 | 25.58 | 17.27 | 17.27 |
| 3 | 25.42 | 32.68 | 32.68 | 7.26 | 7.26 |
| 5 | 42.06 | 44.42 | 44.42 | 2.36 | 2.36 |
| 7 | 48.60 | 47.02 | 47.02 | 1.58 | 1.58 |
| 11 | 59.50 | 65.78 | 65.78 | 6.28 | 6.28 |
| 12 | 64.64 | 67.40 | 67.40 | 2.76 | 2.76 |
| 20 | 119.90 | 122.18 | 122.18 | 2.28 | 2.28 |
| 25 | 135.10 | 138.71 | 138.71 | 3.61 | 3.61 |
| 26 | 142.70 | 144.00 | 144.00 | 1.30 | 1.30 |
| 30 | 167.97 | 165.19 | 165.19 | 2.78 | 2.78 |
| 31 | 174.40 | 166.33 | 166.33 | 8.07 | 8.07 |
| 32 | 181.35 | 167.97 | 167.97 | 13.38 | 13.38 |
| 33 | 185.00 | 171.33 | 171.33 | 13.67 | 13.67 |
| 34 | 188.50 | 173.35 | 173.35 | 15.15 | 15.15 |
| 35 | 196.50 | 177.37 | 177.37 | 19.13 | 19.13 |
| 36 | 215.50 | 179.63 | 179.63 | 35.87 | 35.87 |
| 40 | 241.50 | 237.93 | 237.93 | 3.57 | 3.57 |
| 41 | 244.80 | 238.85 | 238.85 | 5.95 | 5.95 |
| 42 | 248.00 | 239.83 | 239.83 | 8.17 | 8.17 |
| 43 | 250.90 | 262.53 | 262.53 | 11.63 | 11.63 |
| 44 | 255.90 | 269.05 | 269.05 | 13.15 | 13.15 |
### 01 My Dearest（20 行 >1s）
| 行 | BK | main | segment | main 誤差 | segment 誤差 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.20 | 11.90 | 11.90 | 7.70 | 7.70 |
| 3 | 11.82 | 33.34 | 33.34 | 21.52 | 21.52 |
| 4 | 35.90 | 37.98 | 37.98 | 2.08 | 2.08 |
| 17 | 121.40 | 123.88 | 123.88 | 2.48 | 2.48 |
| 21 | 143.90 | 169.28 | 169.28 | 25.38 | 25.38 |
| 22 | 146.90 | 171.92 | 171.92 | 25.02 | 25.02 |
| 23 | 154.80 | 174.46 | 174.46 | 19.66 | 19.66 |
| 24 | 161.64 | 174.64 | 174.64 | 13.00 | 13.00 |
| 25 | 169.60 | 174.76 | 174.76 | 5.16 | 5.16 |
| 29 | 195.66 | 193.92 | 195.66 | 1.74 | 0.00 |
| 31 | 207.20 | 216.38 | 216.38 | 9.18 | 9.18 |
| 32 | 212.70 | 231.93 | 231.93 | 19.23 | 19.23 |
| 33 | 218.20 | 263.81 | 263.81 | 45.61 | 45.61 |
| 34 | 221.00 | 265.35 | 265.35 | 44.35 | 44.35 |
| 35 | 229.05 | 266.05 | 266.05 | 37.00 | 37.00 |
| 36 | 264.32 | 266.32 | 266.32 | 2.00 | 2.00 |
| 37 | 270.00 | 268.89 | 269.96 | 1.11 | 0.04 |
| 38 | 275.69 | 270.69 | 270.69 | 5.00 | 5.00 |
| 39 | 281.23 | 272.83 | 272.83 | 8.40 | 8.40 |
| 40 | 287.07 | 275.07 | 275.07 | 12.00 | 12.00 |
| 41 | 292.80 | 277.47 | 277.47 | 15.33 | 15.33 |
| 42 | 298.50 | 281.21 | 281.21 | 17.29 | 17.29 |
### 01.ミレニアの水槽（12 行 >1s）
| 行 | BK | main | segment | main 誤差 | segment 誤差 |
|---:|---:|---:|---:|---:|---:|
| 9 | 52.52 | 51.36 | 51.36 | 1.16 | 1.16 |
| 10 | 57.04 | 54.15 | 54.15 | 2.89 | 2.89 |
| 11 | 60.58 | 57.25 | 57.25 | 3.33 | 3.33 |
| 13 | 72.26 | 70.98 | 70.98 | 1.28 | 1.28 |
| 14 | 81.78 | 75.88 | 75.88 | 5.90 | 5.90 |
| 17 | 96.08 | 91.10 | 91.10 | 4.98 | 4.98 |
| 18 | 97.88 | 91.62 | 91.62 | 6.26 | 6.26 |
| 19 | 101.00 | 107.81 | 107.81 | 6.81 | 6.81 |
| 20 | 103.82 | 108.77 | 108.77 | 4.95 | 4.95 |
| 21 | 107.60 | 110.22 | 110.22 | 2.62 | 2.62 |
| 35 | 188.45 | 187.31 | 187.31 | 1.14 | 1.14 |
| 45 | 243.72 | 238.51 | 238.51 | 5.21 | 5.21 |

## 涵蓋率分析（僅分析，未修改分段演算法）
| 歌 | 退回條件 | DP margin | passage/rows | CTC 增益 | main >1s |
|---|---|---:|---:|---:|---:|
| 01. The Day the Starlight Arrived 星辉抵达之日 | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/24 = 未提供 | 未提供 | 4 |
| 03. Choir ‘S’ Choir | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/44 = 未提供 | 未提供 | 21 |
| Ever be my love | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/24 = 未提供 | 未提供 | 3 |
| 01.ミレニアの水槽 | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/45 = 未提供 | 未提供 | 12 |
| 12. 終点の先が在るとするならば。 | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/32 = 未提供 | 未提供 | 3 |
| 09.夢遊 | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/30 = 未提供 | 未提供 | 6 |
| 01 My Dearest | ASR_UNCORROBORATED_STRUCTURE | 未提供 | None/43 = 未提供 | 未提供 | 22 |

退回條件分布：{'ASR_UNCORROBORATED_STRUCTURE': 7}
共通方向：先量測低 passage/rows 比率與 DP margin 的分布，再在所有歌曲上統一校準 VAD 分段及 DP 的聲學支持；不可針對單曲調參。
風險：副歌錯配、間奏合併、放寬信心門檻後新增 >1s；需在完整 39 首重新驗證。

## 本地 commit 列表（報告寫入前）

- 4c37e8c3 chore(lrc13): keep 7-song A/B monitoring and scorers on actual output path
- de8f8514 fix(lrc13): withhold distant fused onset when CTC and raw ASR acoustics corroborate
- 9ad34883 chore(lrc13): add live full-validation status and honest release report caveats
- 269dc8e6 test(lrc13): stage guarded full Central AB and final A7 validation pipeline
- 48f8513d wip(lrc13): integrate P2/P3/P6 guards and pass 39-song skeleton safety regression
- 9a3ca1f8 wip(lrc13): P1 raw-ASR content DP 39-song failed-gate evidence
- c9518716 refactor(lrc13): P4 share global CTC confidence and verify 39 song parity
- 252a3990 test(lrc13): score all 39 P5 skeleton-layer runs before continuing
- 62b7e2b4 fix(lrc13): P5 preserve global CTC and flag repeated skeleton disagreements
- 7ce01ba2 test(lrc13): document 39-song A7 pinned-main regression
- de9bcff6 fix(lrc13): compare A7 backtracks against pinned main and retain inherited warnings
- 947ded9f A6 gate: distinguish inherited from new backtracks
- e7899e3d fix(lrc13): enforce atomic postcentral rollback and strict A6 regression gate
- 45e86547 fix(lrc13): retain Option A atomic fallback diagnostics after legacy backend selection
- 6af6eca0 feat(lrc13): implement full-song Option A vocal passage skeleton with atomic fallback
- 360e532b perf(lrc13): bound fresh Whisper evidence in repeated Central replay
- 567f2caa fix(lrc13): permit nonmonotonic main baselines in S5 scoring
- 3a3b45d8 test(lrc13): enforce corpus-wide S5 timing and trust regression gates
- 565a4f81 test(lrc13): audit Japanese ASR phonetic ambiguity without authority
- 067c343c feat(lrc13): acquire S1-centered independent Whisper occurrence probes
- 14580f6c feat(lrc13): calibrate reviewers on separate human BK training rows
- bb38e620 feat(lrc13): derive reviewer relative clocks without baseline
- ba2da292 fix(lrc13): exclude legacy provenance row from timed lyrics
- 44eba9db feat(lrc13): honor manually corrected BK in explicit checked mode
- 206fa568 feat(lrc13): research fail-closed single-line vocal recurrence
- 6fe40695 feat(lrc13): audit three-plus vocal repeats without trust promotion
- d32cfce2 perf(lrc13): reuse validated structural CUDA CTC in Central previews
- aff79da0 feat(lrc13): wire fail-closed structural CUDA CTC into Central preview
- 139f99f2 feat(lrc13): stage atomic segment CTC hypotheses for Central
- 750e2c96 feat(lrc13): reject unsafe CTC passage-boundary swaps
- d201bf03 feat(lrc13): infer repeated chorus anchors from vocal acoustics
- 33b01d2c feat(lrc13): allow fail-closed acoustic passage anchors in DP
- afa4b3d0 feat(lrc13): add fail-closed vocal acoustic recurrence research
- e394213f fix(lrc13): avoid English refrain overriding Japanese verses
- f6648d1a fix(lrc13): reject partial Central passage acceptance atomically
- 6df5c251 perf(lrc13): reduce MMS CTC GPU peak without audio windowing
- a1ebc413 fix(lrc13): close persistent helper subprocess pipes
- e0b1319f fix(lrc13): audit all lyric onsets on isolated vocal evidence
- 3893fc3b fix(lrc13): calibrate reviewer clocks from independent vocal onsets
- 3a151cf5 fix(lrc13): infer Whisper language from full sung lyrics
- c2096343 feat(lrc13): add fail-closed vocal-passage DP and atomic bundle guards
- 1c5e5522 chore(lrc13): carry forward local R1-R4 and 1.2 hardening

不 push、不合併 main。LRC bk 與音樂唯讀；BK 只用於凍結後評分。
