## LRC Timeline Aligner v1.7.1

Review verdicts are recorded so the review threshold can be tuned on real outcomes. Alignment, report format and the save flow are unchanged from v1.7.0.

### Added
- Pressing 維持 (K) or accepting a line (Enter) on the review page appends one JSON line to `<song>.review-decisions.jsonl` beside the align report: `keep` or `change`, the chosen and current time, the line's review status, reason codes, severity rank, candidate spread and the options that were shown. Read-only songs record nothing.
- `scripts/review_decisions_summary.py`: read-only tally of the last verdict per line, per review status and per reason code (keep / change / change %). `--json` for machine-readable output.

### Changed
- The import panel's queue lists only the newest job and any still running, so there is one 完成 summary and one 開始審核 button.

### Verification
- Windows: `python -m unittest discover -s scripts -p "test_*.py"`, `scripts\doctor.py` reports v1.7.1, `node scripts\test_review_page.js`.
