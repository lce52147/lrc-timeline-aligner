# LRC13 skeleton fix audit — 2026-10-11

Branch: `lrc-1.3`. No push, no main merge, no modifications to BK/audio.
Input vocal stems are cached. All quick runs use frozen pinned-main CTC
(9069dbc8) as original acoustic seed, without Central / WhisperX / reviewer.
BK times are used only for posthoc score after each phase's 39 outputs freeze.

## P5: evidence preservation (commit 62b7e2b4)
- 39/39 skeleton-layer actual CTC; active 16/39.
- PCO@0.2 1285 → 1215 of 1487; >1s errors 115 → 167;
  92 newly >1s; 15 active songs worse; **FAIL**.
- Dream #28 is correctly marked F4+F5 untrusted, but clock still 160.89.
- Fresh full Central Dream completed; all 30 timestamps unchanged.

## P4: confidence parity (commit c9518716)
- Global and local CTC originally shared identical score bins 0.35/0.70/0.90.
- Refactored to one shared function; 39/39 independent skeleton-layer
  re-run exactly same time/status as P5. **PASS parity**, no claimed trust lift.
- 712 tests passed (2 skipped) at P4.

## P1: raw-ASR paragraph lexical DP (experimental, NOT accepted)
- `W_ASR=1.0`; raw Whisper.cpp paragraph temporal overlap,
  NFKC/romaji bigram Dice, no synthetic line starts or BK timings.
- 39/39 real frozen-vocal skeleton-layer completed, no runtime failures.
- Active 9/39, fallback 30/39. Active worse 8/9 — **HARD GATE FAIL**.
- PCO@0.2 1285 → 1246 of 1487; >1s 115 → 139;
  59 newly >1s; 0 newly backtracked.
- Dream is **FALLBACK_CTC** (thus avoids wrong skeleton #28).
- Complete Python unit suite **716 tests OK (2 skipped)**,
  after fixing immutable synthetic args namespace issue.
- DP-only ASR-weight sensitivity on 39 songs, without BK:
  W 0.5 -> DP READY 26; W 1.0 -> 22; W 2.0 -> 29.
  This is NOT local-CTC acceptance at alternative weights; production W remains 1.0.
- Raw ASR 39/39 paragraph events captured independently and archived.
- Content-fit alone is not enough to support local timing:
  even songs with high ASR bigram match worsened.
- **P2/P3/P6 and 7-song full A/B / 39-song final A7 are not yet accepted.**
  Do not mislabel this P1 stage PASS; do not continue release gate until fixed.

Evidence:
- `LRC13_P1_SKELETON_39_20261011/RUN_MANIFEST.json`
- `LRC13_P1_SKELETON_39_SCORE_20261011.json` and `.md`
- `LRC13_P1_DP_SENSITIVITY_20261011.json`
- `LRC13_P1_ASR_CONTENT_AUDIT_20261011.json`
- `LRC13_P1_RAW_ASR_PARAGRAPHS_20261011.json`
- `LRC13_P4_COMPARE_20261011.json`

## P1 remediation A: neighbor-passage group acoustic competition (NOT accepted)
- 39/39 real skeleton-layer rerun, local CTC group checks against both
  preceding/following passages. Unsupported >15% of rows rolls back entire song;
  remaining rows are marked review and untrusted.
- Active 9, 8 worsened. PCO@0.2 1285 → 1286/1487,
  >1s 115 → 107, **31 new >1s**, 0 new backtracks.
- In particular, the Dream skeleton became active again and was still wrong.
- `LRC13_P1_GUARD_39_SCORE_20261011.json/.md`: **FAIL**.
- This implementation partially overlaps the planned P3, but does NOT
  complete P3: the old global CTC gain/displacement thresholds remain active.

## P1 remediation B: strict repeat F5 fallback (NOT accepted)
- 39/39 real skeleton-layer rerun. When a repeated lyric would move >3 s
  away from pinned full-song CTC, entire skeleton falls back atomically.
- Active 6, **6 worsened**. PCO@0.2 1285 → 1263/1487,
  >1s 115 → 134, **23 new >1s**, 0 new backtracks.
- Dream #28: entire song fell back with
  `REPEATED_SKELETON_GLOBAL_CTC_CONFLICT` (no false segment promotion).
- Remaining >1s errors mostly occur on songs with HIGH global CTC acoustic
  scores, e.g. Vivid, JANE DOE, 月の唄, 远航星的告别.
- Unit suite at this revision: **727 tests OK, 2 skipped**.
- `LRC13_P1_F5_39_SCORE_20261011.json/.md`: **FAIL**.
- Do not report the aggregate >1s reduction from neighbor-only run as
  definitive improvement: the repeat guard modifies which songs are active
  and both variants need full Central tests after hard gates.

## P2 min-marginal offline mathematical precheck (NOT production/P2 accepted)
- Exact forward/backward on 39 frozen vocal-passages/ASR/CTC priors completed.
- 4 independent synthetic/brute-force tests passed.
- Adjacent versus nonadjacent ambiguity was distinguishable. 2 songs
  had nonadjacent ambiguity >10% of lyric rows.
- **All six songs still activated and worsened under remediation B had
  zero min-marginal ambiguity rows**, so P2 min-marginals alone cannot be
  used as a basis to claim P1 acceptance.
- `LRC13_P2_MINMARG_PRECHECK_39_20261011.json`: precheck only.
- No official P2/P3/P6 commits, 7-song full A/B, nor final 39 A7 completed.

**Current overall status: NOT_PASS. Release remains blocked by P1 zero-regression hard gate.**


## P2/P3/P6 integrated skeleton layer — 2026-10-11
- Per-line P2 min-sum marginals replace global margin rejection. Nonadjacent ambiguity >10% gives atomic rollback, adjacent ambiguity is sent to boundary competition; unresolved structural rows require review.
- P3 neighboring passage local CTC is used for placement, while old whole-song vs local score gain/displacement is diagnostic-only.
- P6 CTC scores below 0.005 are actionable review/untrusted; hard failure minimum remains 0.0002.
- First integrated 39-song run: active 8/39, worse 8, 31 new >1s. FAIL. See LRC13_P2P6_EXP_39_SCORE_20261011.json.
- 733 unit tests OK (2 skipped).

## ASR text corroboration conservative candidate — 2026-10-11
- BK-blind comparison of written lyric text against raw Whisper.cpp paragraphs at original global-CTC and proposed local-CTC times. This is correlated with DP raw ASR selection, not independent onset proof.
- Material moves >0.2s require mean raw ASR Dice support gain >=0.25 to promote. Repeated F5 disagreements remain review evidence.
- 39/39 skeleton-layer actual runs: 2 active, 37 fallback. Active: 04 影色舞 and 05.遠雷.
- ZERO-REGRESSION QUICK GATE PASS: PCO@0.2 1285 to 1314/1487; >1s 115 to 83; newly >1s 0; new backtracks 0; worse active songs 0.
- Evidence: LRC13_ASR_CORR_39_SCORE_20261011.json/.md.
- These 39 examples were used in prior diagnosis, so not an independent holdout.
- Overall release remains NOT_PASS until the seven-song complete Central A/B and final 39-song A7 are performed.
