# P1 skeleton-layer 39-song posthoc score

Frozen pinned-main CTC evidence as seed; true local skeleton inference without Central/WhisperX/reviewer.
BK used only after inference complete. Full A7 scores are a different stage.
Status: ASR_CORR_LAYER_39_POSTHOC_SCORED; hard gate: True
Counts: {'rows': 1487, 'pco_old': 1285, 'pco_new': 1314, 'gt1_old': 115, 'gt1_new': 83, 'new_gt1': 0, 'new_backtracks': 0, 'active': 2}; active worse songs: []

| Song | mode | PCO@0.2 CTC→skeleton | >1s CTC→skeleton | newly >1s | new backtrack |
|---|---|---:|---:|---:|---:|
| 01. The Day the Starlight Arrived 星辉抵达之日 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 18→18/24 | 2→2 | 0 | 0 |
| 01.エガクミライ | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 27→27/27 | 0→0 | 0 | 0 |
| 01.感情グラス | FALLBACK_CTC / STRUCTURAL_DP_AMBIGUITY | 40→40/40 | 0→0 | 0 | 0 |
| 03. Choir ‘S’ Choir | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 16→16/44 | 22→22 | 0 | 0 |
| 03. 過去に囚われている | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 65→65/68 | 0→0 | 0 | 0 |
| 04 影色舞 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 18→47/52 | 32→0 | 0 | 0 |
| 06.ビビッド | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 34→34/55 | 3→3 | 0 | 0 |
| 10.方舟 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 34→34/37 | 0→0 | 0 | 0 |
| 11.ユートピア | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 32→32/32 | 0→0 | 0 | 0 |
| Ever be my love | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 18→18/24 | 1→1 | 0 | 0 |
| 01 - フロム | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 34→34/35 | 0→0 | 0 | 0 |
| 01. Elegy Dedicated With Love | FALLBACK_CTC / STRUCTURAL_DP_AMBIGUITY | 35→35/37 | 1→1 | 0 | 0 |
| 01.おやすみモノクローム | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 39→39/40 | 0→0 | 0 | 0 |
| 01. それでも雨は降るんだね | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 55→55/55 | 0→0 | 0 | 0 |
| 01. ふり | FALLBACK_CTC / TOO_FEW_PASSAGES | 54→54/57 | 0→0 | 0 | 0 |
| 01. やっぱり雨は降るんだね | FALLBACK_CTC / TOO_FEW_PASSAGES | 67→67/67 | 0→0 | 0 | 0 |
| 01.ミレニアの水槽 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 32→32/45 | 10→10 | 0 | 0 |
| 01.最大幸福度 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 42→42/45 | 0→0 | 0 | 0 |
| 01. 芽吹くとき | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 24→24/25 | 0→0 | 0 | 0 |
| 01. 雨模様 | FALLBACK_CTC / TOO_FEW_PASSAGES | 43→43/49 | 1→1 | 0 | 0 |
| 02. Georgette Me, Georgette You | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 13→13/18 | 5→5 | 0 | 0 |
| 02. JANE DOE | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 21→21/24 | 1→1 | 0 | 0 |
| 03.春を待つ (feat. 倚水) | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 39→39/39 | 0→0 | 0 | 0 |
| 04. Scarborough Fair | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 13→13/14 | 1→1 | 0 | 0 |
| 04.可惜夜 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 23→23/24 | 0→0 | 0 | 0 |
| 05.遠雷 | SEGMENT_CURRENT / VALID_LOCAL_CTC | 27→27/28 | 0→0 | 0 | 0 |
| 06. あの世行きのバスに乗ってさらば。 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 34→34/40 | 2→2 | 0 | 0 |
| 08.月の唄 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 36→36/38 | 0→0 | 0 | 0 |
| 08 春日影 (MyGO!!!!! Ver.) | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 38→38/39 | 0→0 | 0 | 0 |
| 09. ツユ - レインフォール | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 71→71/74 | 0→0 | 0 | 0 |
| 10.深夜特急 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 41→41/41 | 0→0 | 0 | 0 |
| 12. ツユ - 朧月夜物語 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 19→19/23 | 3→3 | 0 | 0 |
| 12.琥珀の国 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 31→31/32 | 0→0 | 0 | 0 |
| 12. 終点の先が在るとするならば。 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 25→25/32 | 4→4 | 0 | 0 |
| 13.薔薇の下で | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 30→30/30 | 0→0 | 0 | 0 |
| 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 26→26/27 | 0→0 | 0 | 0 |
| 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 26→26/33 | 0→0 | 0 | 0 |
| 09.夢遊 | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 25→25/30 | 4→4 | 0 | 0 |
| 01 My Dearest | FALLBACK_CTC / ASR_UNCORROBORATED_STRUCTURE | 20→20/43 | 23→23 | 0 | 0 |

## New >1s errors
