## LRC Timeline Aligner v1.6.0

Review page redesign. Alignment, report format and the save flow are unchanged from v1.5.0; the 39-song validation pool replays byte-identically.

### What's new in v1.6

- The review page shows one line at a time: lyric, a plain-language reason sentence, a marked waveform and at most three timing options (A / B / keep, keys `1` / `2` / `3`). Candidates closer than 0.25 s merge into one option; candidates within 0.15 s of the current time fold into "keep". Raw candidates, internal source names, severity and reason codes move to a collapsible "詳細" fold.
- `row["reason_text"]`: one sentence per row, chosen by priority from the reason codes (`scripts/review_presentation.py`).
- Waveform markers for current / A / B / previous / next line with a 0.5 s ruler; FLAC decoded in the browser when the vocal stem is unavailable; a visible message when no waveform can be shown.
- Song list with remaining counts, auto-start on the first unfinished song, stat tiles (this song's progress, remaining here and overall, unsaved), per-song queue as one-line rows with a verdict chip, floating play bar while audio plays.
- Keys: `Enter` accept and next, `K` keep and next, `U` undo, `← →` or `[ ]` nudge 0.1 s (Shift 0.01 s), `B` A/B compare, `V` FLAC/stem, `Tab` details, `?` help. The UNTRUSTED filter lives in the help panel.
- Visual refresh: glass panels over a warm gradient, indigo accent, centered 1120 px column.

### Install

In the existing v1.5.0 folder, `git fetch` and `git merge --ff-only origin/main`. No cache migration or dependency change. `Review LRC.bat` opens the new page; old servers still serve the old page.
