# P5 skeleton-layer 39-song posthoc score

Frozen pinned-main CTC evidence as seed; true local skeleton inference without Central/WhisperX/reviewer.
BK used only after inference complete. Full A7 scores are a different stage.
Status: P5_SKELETON_LAYER_39_POSTHOC_SCORED; hard gate: False
Counts: {'rows': 1487, 'pco_old': 1285, 'pco_new': 1215, 'gt1_old': 115, 'gt1_new': 167, 'new_gt1': 92, 'new_backtracks': 0, 'active': 16}; active worse songs: ['01. The Day the Starlight Arrived 星辉抵达之日', '03. 過去に囚われている', '06.ビビッド', '01 - フロム', '01.おやすみモノクローム', '01. それでも雨は降るんだね', '01.最大幸福度', '04. Scarborough Fair', '06. あの世行きのバスに乗ってさらば。', '08.月の唄', '10.深夜特急', '12. ツユ - 朧月夜物語', '12. 終点の先が在るとするならば。', '远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki', '09.夢遊']

| Song | mode | PCO@0.2 CTC→skeleton | >1s CTC→skeleton | newly >1s | new backtrack |
|---|---|---:|---:|---:|---:|
| 01. The Day the Starlight Arrived 星辉抵达之日 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 18→15/24 | 2→4 | 4 | 0 |
| 01.エガクミライ | FALLBACK_CTC / AMBIGUOUS_DP | 27→27/27 | 0→0 | 0 | 0 |
| 01.感情グラス | FALLBACK_CTC / LOCAL_WEAK_CTC_SUPPORT | 40→40/40 | 0→0 | 0 | 0 |
| 03. Choir ‘S’ Choir | FALLBACK_CTC / AMBIGUOUS_DP | 16→16/44 | 22→22 | 0 | 0 |
| 03. 過去に囚われている | SEGMENT_CURRENT / VALID_LOCAL_CTC | 65→56/68 | 0→9 | 9 | 0 |
| 04 影色舞 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 18→47/52 | 32→0 | 0 | 0 |
| 06.ビビッド | SEGMENT_CURRENT / VALID_LOCAL_CTC | 34→20/55 | 3→18 | 17 | 0 |
| 10.方舟 | FALLBACK_CTC / AMBIGUOUS_DP | 34→34/37 | 0→0 | 0 | 0 |
| 11.ユートピア | FALLBACK_CTC / AMBIGUOUS_DP | 32→32/32 | 0→0 | 0 | 0 |
| Ever be my love | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 18→18/24 | 1→1 | 0 | 0 |
| 01 - フロム | SEGMENT_CURRENT / VALID_LOCAL_CTC | 34→25/35 | 0→8 | 8 | 0 |
| 01. Elegy Dedicated With Love | FALLBACK_CTC / AMBIGUOUS_DP | 35→35/37 | 1→1 | 0 | 0 |
| 01.おやすみモノクローム | SEGMENT_CURRENT / VALID_LOCAL_CTC | 39→31/40 | 0→7 | 7 | 0 |
| 01. それでも雨は降るんだね | SEGMENT_CURRENT / VALID_LOCAL_CTC | 55→35/55 | 0→11 | 11 | 0 |
| 01. ふり | FALLBACK_CTC / TOO_FEW_PASSAGES | 54→54/57 | 0→0 | 0 | 0 |
| 01. やっぱり雨は降るんだね | FALLBACK_CTC / TOO_FEW_PASSAGES | 67→67/67 | 0→0 | 0 | 0 |
| 01.ミレニアの水槽 | FALLBACK_CTC / AMBIGUOUS_DP | 32→32/45 | 10→10 | 0 | 0 |
| 01.最大幸福度 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 42→25/45 | 0→16 | 16 | 0 |
| 01. 芽吹くとき | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 24→24/25 | 0→0 | 0 | 0 |
| 01. 雨模様 | FALLBACK_CTC / TOO_FEW_PASSAGES | 43→43/49 | 1→1 | 0 | 0 |
| 02. Georgette Me, Georgette You | FALLBACK_CTC / AMBIGUOUS_DP | 13→13/18 | 5→5 | 0 | 0 |
| 02. JANE DOE | FALLBACK_CTC / AMBIGUOUS_DP | 21→21/24 | 1→1 | 0 | 0 |
| 03.春を待つ (feat. 倚水) | FALLBACK_CTC / AMBIGUOUS_DP | 39→39/39 | 0→0 | 0 | 0 |
| 04. Scarborough Fair | SEGMENT_CURRENT / VALID_LOCAL_CTC | 13→11/14 | 1→2 | 1 | 0 |
| 04.可惜夜 | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 23→23/24 | 0→0 | 0 | 0 |
| 05.遠雷 | FALLBACK_CTC / AMBIGUOUS_DP | 27→27/28 | 0→0 | 0 | 0 |
| 06. あの世行きのバスに乗ってさらば。 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 34→30/40 | 2→7 | 5 | 0 |
| 08.月の唄 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 36→36/38 | 0→1 | 1 | 0 |
| 08 春日影 (MyGO!!!!! Ver.) | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 38→38/39 | 0→0 | 0 | 0 |
| 09. ツユ - レインフォール | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 71→71/74 | 0→0 | 0 | 0 |
| 10.深夜特急 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 41→39/41 | 0→2 | 2 | 0 |
| 12. ツユ - 朧月夜物語 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 19→12/23 | 3→10 | 7 | 0 |
| 12.琥珀の国 | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 31→31/32 | 0→0 | 0 | 0 |
| 12. 終点の先が在るとするならば。 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 25→24/32 | 4→4 | 1 | 0 |
| 13.薔薇の下で | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 30→30/30 | 0→0 | 0 | 0 |
| 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 26→26/27 | 0→0 | 0 | 0 |
| 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki | SEGMENT_CURRENT / VALID_LOCAL_CTC | 26→22/33 | 0→2 | 2 | 0 |
| 09.夢遊 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 25→26/30 | 4→2 | 1 | 0 |
| 01 My Dearest | FALLBACK_CTC / INSUFFICIENT_GLOBAL_ACOUSTIC_SUPPORT | 20→20/43 | 23→23 | 0 | 0 |

## New >1s errors
- 01. The Day the Starlight Arrived 星辉抵达之日 #3: BK 16.80, CTC 16.761, skeleton 20.480
- 01. The Day the Starlight Arrived 星辉抵达之日 #5: BK 33.00, CTC 32.881, skeleton 36.200
- 01. The Day the Starlight Arrived 星辉抵达之日 #7: BK 49.00, CTC 49.142, skeleton 47.260
- 01. The Day the Starlight Arrived 星辉抵达之日 #24: BK 237.00, CTC 237.030, skeleton 240.700
- 03. 過去に囚われている #10: BK 26.16, CTC 26.163, skeleton 23.954
- 03. 過去に囚われている #11: BK 28.78, CTC 28.783, skeleton 26.620
- 03. 過去に囚われている #24: BK 67.28, CTC 67.287, skeleton 70.121
- 03. 過去に囚われている #25: BK 70.14, CTC 70.127, skeleton 71.463
- 03. 過去に囚われている #40: BK 118.47, CTC 118.472, skeleton 119.763
- 03. 過去に囚われている #55: BK 172.78, CTC 173.117, skeleton 174.102
- 03. 過去に囚われている #62: BK 201.86, CTC 201.880, skeleton 197.855
- 03. 過去に囚われている #63: BK 204.53, CTC 204.540, skeleton 202.340
- 03. 過去に囚われている #64: BK 207.19, CTC 207.200, skeleton 205.301
- 06.ビビッド #2: BK 19.30, CTC 19.382, skeleton 18.220
- 06.ビビッド #3: BK 22.50, CTC 22.902, skeleton 20.800
- 06.ビビッド #4: BK 25.60, CTC 25.802, skeleton 27.042
- 06.ビビッド #5: BK 30.30, CTC 30.383, skeleton 31.500
- 06.ビビッド #8: BK 39.50, CTC 39.784, skeleton 42.220
- 06.ビビッド #10: BK 47.40, CTC 47.404, skeleton 49.802
- 06.ビビッド #19: BK 83.00, CTC 83.187, skeleton 85.300
- 06.ビビッド #27: BK 109.50, CTC 109.570, skeleton 110.760
- 06.ビビッド #28: BK 114.00, CTC 113.890, skeleton 115.540
- 06.ビビッド #31: BK 124.00, CTC 124.231, skeleton 117.609
- 06.ビビッド #32: BK 127.30, CTC 127.431, skeleton 120.800
- 06.ビビッド #39: BK 151.00, CTC 151.014, skeleton 147.698
- 06.ビビッド #40: BK 152.60, CTC 152.614, skeleton 148.158
- 06.ビビッド #41: BK 157.30, CTC 157.414, skeleton 149.799
- 06.ビビッド #42: BK 159.00, CTC 159.074, skeleton 150.139
- 06.ビビッド #43: BK 161.00, CTC 160.994, skeleton 150.820
- 06.ビビッド #44: BK 162.60, CTC 162.615, skeleton 153.620
- 01 - フロム #1: BK 20.16, CTC 20.161, skeleton 25.860
- 01 - フロム #2: BK 22.94, CTC 22.941, skeleton 35.825
- 01 - フロム #3: BK 28.78, CTC 28.782, skeleton 41.317
- 01 - フロム #4: BK 37.06, CTC 37.062, skeleton 44.560
- 01 - フロム #5: BK 44.90, CTC 44.903, skeleton 47.980
- 01 - フロム #13: BK 106.17, CTC 106.167, skeleton 108.822
- 01 - フロム #14: BK 108.81, CTC 108.827, skeleton 111.701
- 01 - フロム #15: BK 114.63, CTC 114.627, skeleton 119.281
- 01.おやすみモノクローム #4: BK 18.42, CTC 18.441, skeleton 19.940
- 01.おやすみモノクローム #5: BK 28.90, CTC 28.922, skeleton 30.440
- 01.おやすみモノクローム #15: BK 86.29, CTC 86.307, skeleton 89.600
- 01.おやすみモノクローム #18: BK 100.15, CTC 100.168, skeleton 101.620
- 01.おやすみモノクローム #29: BK 162.55, CTC 162.573, skeleton 169.000
- 01.おやすみモノクローム #30: BK 169.31, CTC 169.333, skeleton 172.920
- 01.おやすみモノクローム #33: BK 186.57, CTC 186.614, skeleton 189.080
- 01. それでも雨は降るんだね #3: BK 10.70, CTC 10.721, skeleton 12.140
- 01. それでも雨は降るんだね #5: BK 17.57, CTC 17.601, skeleton 19.100
- 01. それでも雨は降るんだね #11: BK 37.11, CTC 37.162, skeleton 38.140
- 01. それでも雨は降るんだね #24: BK 96.43, CTC 96.446, skeleton 94.541
- 01. それでも雨は降るんだね #25: BK 99.93, CTC 99.947, skeleton 98.041
- 01. それでも雨は降るんだね #26: BK 103.37, CTC 103.387, skeleton 101.481
- 01. それでも雨は降るんだね #27: BK 106.93, CTC 106.927, skeleton 102.831
- 01. それでも雨は降るんだね #29: BK 113.18, CTC 113.187, skeleton 112.079
- 01. それでも雨は降るんだね #39: BK 151.54, CTC 151.570, skeleton 152.580
- 01. それでも雨は降るんだね #42: BK 163.94, CTC 163.971, skeleton 165.000
- 01. それでも雨は降るんだね #53: BK 208.05, CTC 208.154, skeleton 209.140
- 01.最大幸福度 #1: BK 11.61, CTC 11.621, skeleton 22.743
- 01.最大幸福度 #2: BK 14.81, CTC 14.801, skeleton 32.600
- 01.最大幸福度 #3: BK 17.56, CTC 17.561, skeleton 39.900
- 01.最大幸福度 #4: BK 21.16, CTC 21.161, skeleton 42.660
- 01.最大幸福度 #5: BK 22.66, CTC 22.661, skeleton 46.141
- 01.最大幸福度 #6: BK 27.34, CTC 27.641, skeleton 50.160
- 01.最大幸福度 #7: BK 30.87, CTC 30.861, skeleton 56.400
- 01.最大幸福度 #8: BK 32.99, CTC 32.981, skeleton 57.161
- 01.最大幸福度 #9: BK 37.99, CTC 37.982, skeleton 57.762
- 01.最大幸福度 #10: BK 42.76, CTC 42.762, skeleton 60.485
- 01.最大幸福度 #11: BK 46.14, CTC 46.142, skeleton 62.768
- 01.最大幸福度 #12: BK 56.40, CTC 56.402, skeleton 66.412
- 01.最大幸福度 #13: BK 62.60, CTC 62.603, skeleton 70.497
- 01.最大幸福度 #14: BK 67.30, CTC 67.283, skeleton 73.340
- 01.最大幸福度 #15: BK 73.42, CTC 73.423, skeleton 74.422
- 01.最大幸福度 #17: BK 88.51, CTC 88.484, skeleton 90.980
- 04. Scarborough Fair #13: BK 129.80, CTC 129.828, skeleton 126.513
- 06. あの世行きのバスに乗ってさらば。 #8: BK 35.34, CTC 35.344, skeleton 33.611
- 06. あの世行きのバスに乗ってさらば。 #9: BK 38.60, CTC 37.624, skeleton 36.481
- 06. あの世行きのバスに乗ってさらば。 #29: BK 117.87, CTC 117.894, skeleton 143.600
- 06. あの世行きのバスに乗ってさらば。 #30: BK 121.45, CTC 121.354, skeleton 145.127
- 06. あの世行きのバスに乗ってさらば。 #31: BK 143.90, CTC 143.897, skeleton 146.192
- 08.月の唄 #4: BK 38.47, CTC 38.482, skeleton 40.900
- 10.深夜特急 #5: BK 47.38, CTC 47.364, skeleton 50.780
- 10.深夜特急 #21: BK 126.21, CTC 126.211, skeleton 129.540
- 12. ツユ - 朧月夜物語 #1: BK 6.72, CTC 6.741, skeleton 11.641
- 12. ツユ - 朧月夜物語 #2: BK 18.76, CTC 18.802, skeleton 26.380
- 12. ツユ - 朧月夜物語 #3: BK 33.74, CTC 33.764, skeleton 42.260
- 12. ツユ - 朧月夜物語 #16: BK 135.33, CTC 135.376, skeleton 136.781
- 12. ツユ - 朧月夜物語 #21: BK 162.35, CTC 162.399, skeleton 158.261
- 12. ツユ - 朧月夜物語 #22: BK 168.23, CTC 168.259, skeleton 164.774
- 12. ツユ - 朧月夜物語 #23: BK 174.69, CTC 174.280, skeleton 168.660
- 12. 終点の先が在るとするならば。 #19: BK 72.82, CTC 72.764, skeleton 75.502
- 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki #1: BK 23.50, CTC 23.543, skeleton 25.440
- 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki #19: BK 108.50, CTC 108.552, skeleton 111.080
- 09.夢遊 #28: BK 164.65, CTC 164.794, skeleton 160.888
