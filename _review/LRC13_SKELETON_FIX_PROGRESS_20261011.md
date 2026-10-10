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
