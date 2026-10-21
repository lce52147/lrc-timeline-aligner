## LRC Timeline Aligner v1.6.1

Review page fix and a cleanup tool. Alignment, report format and the save flow are unchanged from v1.6.0.

### Fixed
- Songs whose LRC no longer exists (`lrc-missing`) were listed with a finished mark and "no rows to review". They are now folded into one line in the song list ("另有 N 首找不到 LRC，無法審核"), shown grey as read-only when expanded, never chosen as the starting song, and the page explains that the LRC was moved, renamed or belongs to an old test run.

### Added
- `scripts/archive_stale_reports.py`: lists reports in the live reports folder whose `output_path` LRC is missing; `--apply` moves each one with its `.review.tsv`, `.auto.lrc` and `.review-edits.json` into `outputs/reports/_archive/stale_<date>`. Nothing is deleted; reports with a live LRC are never touched.
- `LRC_MULTIMODEL_EXTRA_MODELS` (opt-in, off by default): extra pinned CTC models for the multimodel-consensus reviewer, used for the karaoke-model A/B; the extra specs are part of the consensus cache identity.

### Verification
- Windows: `python -m unittest discover -s scripts -p "test_*.py"`, `scripts\doctor.py` reports v1.6.1, `node scripts\test_review_page.js`.
