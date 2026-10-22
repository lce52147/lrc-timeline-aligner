## LRC Timeline Aligner v1.7.0

Import from the review page, review where the files are, and a retuned review threshold. Alignment, report format and the save flow are unchanged from v1.6.1.

### Added
- **＋ 新增歌曲** in the review page: pick FLACs or a folder in the native dialog or paste paths; per-song lyric discovery (`.lyrics.lrc` → `.lyrics.txt` → `.txt`); paste lyrics, saved as `<song>.lyrics.txt` beside the FLAC; existing LRCs skipped unless 覆蓋; 開始對齊 runs `align-lrc.ps1` one song at a time with the `Align LRC.bat` defaults; finished songs land in the song list, failed ones show the last `ERROR` line and log path and can be retried. Endpoints `/api/pick`, `/api/import/inspect`, `/api/import/lyrics`, `/api/jobs`, `/api/jobs/<id>/cancel` (token-protected; hidden with `--read-only` or `LRC_NO_REVIEW=1`). Queue state in `outputs\jobs`.
- Live refresh: the page reloads itself when a FLAC, LRC or report changes (`/api/version`), keeping staged edits.
- Weak-review policy at output time (`apply_weak_review_policy`): REVIEW rows whose reason codes all sit in the B2 ladder's middle cut become unverified. B2 (36 songs): REVIEW 538 → 232, lines moved 0, >1 s lines kept in REVIEW 35/37, TRUSTED >1 s 0. Report field `weak_review_policy`; test `scripts/test_weak_review_policy.py` is public-safe.
- `compare_report_runs.py`: sole-trigger table and reason-code demotion ladder vs BK.

### Changed
- Review binds to the folder being reviewed: a dragged folder/FLAC/LRC matches its report by file name, edits save there, a missing FLAC/LRC is looked up beside its partner, and with no FLAC the page plays the vocal stem.
- `Review LRC.bat` with no arguments opens the last run's songs (`outputs\reports\batch-targets.txt`); only the `LRC bk` folder is excluded from review.
- Batch summary covers only the songs queued in the run (`--paths-file`).
- Review page: empty page lists the skipped reports and why; progress tile counts reviewed lines and the song list shows lines left; Ctrl+S with no changes explains itself; the unreviewable group reads 無法審核（LRC 已搬走或是舊版報告）.
- lyricsee v2 (single pipeline, bounded retries, OOGG failover) is included under `lyricsee\`.

### Fixed
- `audio_track_labels` falls back to the file stem when `ffprobe` is missing (CI on a bare Windows runner).
- Public checks surface the failing tail as an annotation.

### Verification
- Windows: `python -m unittest discover -s scripts -p "test_*.py"`, `scripts\doctor.py` reports v1.7.0, `node scripts\test_review_page.js`.
- PC: full user flow (drag FLAC → align → review page → pick two options → Ctrl+S → only those lines changed) recorded in `F:\_RuntimePrompt\LRC15\e2e_user_20261022\USER_WALKTHROUGH.md`; import panel flow to be recorded the same way before tagging.
