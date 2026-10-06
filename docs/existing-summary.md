# Quantitative alignment summary — LRC 1.2

Current published system: **R2 reviewer-validity with the Phase 4B selector**, measured on 2026-10-07.

Source for every numeric measurement, split size, date, confidence interval, threshold, count, percentage, and comparison in this document: **來自內部評估流程，參考答案與逐行資料不公開**. The checked-reference corpus is intentionally excluded from the public repository. Users can reproduce the per-song timing method on their own references with `python .\scripts\evaluate_lrc.py <reference.lrc> <generated.lrc> --json`; the published aggregate values cannot be reproduced from the repository alone.

| Scope | Songs | Entries | PCO@0.2 | Median absolute error | >500 ms | >1 s | <=50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| full analysis pool | 37 | 1414 | 1229 (86.92%) | 20 ms | 115 (8.13%) | 87 (6.15%) | 1089 (77.02%) |
| development partition | 27 | 1011 | 925 (91.49%) | 20 ms | 49 (4.85%) | 30 (2.97%) | 837 (82.79%) |
| former acceptance partition | 10 | 403 | 304 (75.43%) | 30 ms | 66 (16.38%) | 57 (14.14%) | 252 (62.53%) |
| Japanese subset | 30 | 1203 | 1093 (90.86%) | 20 ms | 65 (5.40%) | 45 (3.74%) | 993 (82.54%) |
| English subset | 7 | 211 | 136 (64.46%) | 80 ms | 50 (23.70%) | 42 (19.91%) | 96 (45.50%) |

The former acceptance subset was merged with development on 2026-10-03. The 37-song pool and both historical partitions are descriptive analysis sets, not independent validation. The published selector created no new rows over 500 ms or 1 s relative to its incoming baseline.

Across songs, median PCO@0.2 is 92%; 22 songs reach at least 90%, and 2 songs are below 60% (`04 影色舞`, 38%; `03. Choir ‘S’ Choir`, 39%). R2 reviewer evidence is Japanese-specific and is sparse for English. Trust labeling defaults to `none`; all trusted rows measured in this pool are Japanese, and trust labels are auxiliary metadata rather than correctness guarantees.

## Reviewer trust profiles

Trust labeling is auxiliary metadata and does not guarantee correctness. The
production reviewer set is HUBP, WhisperX (WX), XLSR, and HUB. `strict` remains
`ALL_REVIEW`; `balanced` uses all four reviewers with at least 3 agreeing;
`loose` uses all four with at least 2 agreeing. Default trust labeling remains
`none`.

Fixed-rule / point-estimate results on the 37-song pool:

| Target | Production profile / fixed rule | CV coverage (95% song-bootstrap CI) | CV trusted error (95% song-bootstrap CI) |
|---:|---|---:|---:|
| 2% | `strict = ALL_REVIEW`; closest four-reviewer boundary candidate `HUBP+WX+XLSR/all` was not promoted | 12.8713% [8.7530, 17.0889] | 3.2967% [0.8127, 6.7583] |
| 5% | `balanced = HUBP+WX+XLSR+HUB/>=3` | 27.7935% [19.3035, 36.0784] | 4.3257% [2.4614, 6.5767] |
| 10% | `loose = HUBP+WX+XLSR+HUB/>=2` | 41.7256% [29.6884, 52.8856] | 6.4407% [3.9032, 9.1198] |

`balanced` passes the 5% point-estimate selector but does not pass the
conservative Selector B, which requires the training-fold Wilson 95% upper
bound to be at or below the target. Selector B on the second-round 28-rule
family gives:

| Target | Selector B CV coverage (95% CI) | Selector B trusted error (95% CI) | Fold selection |
|---:|---:|---:|---|
| 2% | 0.0000% [0.0000, 0.0000] | N/A | `ALL_REVIEW` in 37/37 folds |
| 5% | 0.9901% [0.0000, 2.2580] | 42.8571% [33.3333, 50.0000] | `ALL_REVIEW` in 34/37 folds |
| 10% | 41.7256% [28.9923, 53.8987] | 6.4407% [4.0439, 9.5432] | `loose` rule in 37/37 folds |

The second-round rule family was designed after observing the first-round
results. Selector B shows a material family-design effect at 10%: the 28-rule
family changes coverage by -10.4668 percentage points and trusted error by
-8.1934 points relative to the first-round 32-rule family. These CV results
therefore do not substitute for a new independent validation set.

Descriptive partition figures for the fixed production rules:

| Profile | dev coverage / trusted error | former-acceptance coverage / trusted error |
|---|---:|---:|
| `balanced` | 325/1011 = 32.1464% / 14/325 = 4.3077% | 68/403 = 16.8734% / 3/68 = 4.4118% |
| `loose` | 489/1011 = 48.3680% / 34/489 = 6.9530% | 101/403 = 25.0620% / 4/101 = 3.9604% |

All trusted rows from these fixed rules in the analysis pool are Japanese.
Required reviewer evidence that is unsupported or unqualified fails closed to
review. HubertFA was tested but is not shipped and did not establish a
deployable CV-safe profile. VON and REC were tested but are unused because the
measured wrong-confirmation and shifted-time behavior did not establish the
required independence/discrimination.

R2's review/unverified rate is intentionally conservative: dev is 573/1011
(56.68%) and the former-acceptance release replay is 230/403 (57.07%),
while the earlier S03 dev result was 385/1011 (38.08%). Missing or unqualified
reviewer evidence fails closed, and reviewer agreement only relaxes registered
soft evidence gaps. Timing accuracy and trusted/review status are separate
quantities; a higher review rate does not by itself mean worse timing.

## Development split

- 01 - フロム
- 01. Elegy Dedicated With Love
- 01.おやすみモノクローム
- 01. それでも雨は降るんだね
- 01. ふり
- 01. やっぱり雨は降るんだね
- 01.ミレニアの水槽
- 01.最大幸福度
- 01. 芽吹くとき
- 01. 雨模様
- 02. Georgette Me, Georgette You
- 02. JANE DOE
- 03.春を待つ (feat. 倚水)
- 04. Scarborough Fair
- 04.可惜夜
- 05.遠雷
- 06. あの世行きのバスに乗ってさらば。
- 08.月の唄
- 08 春日影 (MyGO!!!!! Ver.)
- 09. ツユ - レインフォール
- 10.深夜特急
- 12. ツユ - 朧月夜物語
- 12.琥珀の国
- 12. 終点の先が在るとするならば。
- 13.薔薇の下で
- 持续瞬间的永恒 - 鸣潮先约电台,jixwang,markmilian
- 远航星的告别 - 鸣潮先约电台,jixwang,Tarokiki

## Former acceptance split (historical/descriptive; not independent)

- 01. The Day the Starlight Arrived 星辉抵达之日
- 01.エガクミライ
- 01.感情グラス
- 03. Choir ‘S’ Choir
- 03. 過去に囚われている
- 04 影色舞
- 06.ビビッド
- 10.方舟
- 11.ユートピア
- Ever be my love

Reference timestamps and lyric text are intentionally not published.
