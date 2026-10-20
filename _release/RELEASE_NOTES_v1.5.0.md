## LRC Timeline Aligner v1.5.0

Architecture update: portable, reproducible model cache; reviewer status and timeouts; less review noise. Line timing policy unchanged.

### Timing results (v1.5 baseline B2 vs v1.4.0 B1)

Measured on the same 39-song / 1487-line manually checked validation pool:

| Measure | v1.4.0 (B1) | v1.5.0 (B2) |
|---|---|---|
| PCO@0.2 | 1362 / 1487 (91.6%) | 1360 / 1487 (91.5%) |
| Lines off by more than 0.5 s | 59 | 60 |
| Lines off by more than 1 s | 40 | 40 |
| TRUSTED lines off by more than 1 s | 0 | 0 |
| REVIEW / UNTRUSTED / TRUSTED rows | 584 / 390 / 513 | 574 / 329 / 584 |

B2 is the v1.5 reference run: the r2 reviewers now run and are cached on every song, where v1.4 silently degraded when a reviewer failed. B2 replays byte-identically (39/39 LRC and review.tsv) from the migrated cache.

### What's new in v1.5

- Portable model cache (schema 2) with `scripts/migrate_model_cache.py`; reviewer sidecars cached; cache-only replay reproduces the full run.
- `scripts/runtime_paths.py` with environment overrides (`LRC_VENV_PYTHON`, `LRC_WHISPER_CLI`, `LRC_WHISPER_MODEL`, `LRC_FFMPEG`, `HF_HUB_CACHE`/`HF_HOME`, `LRC_MODEL_CACHE_DIR`, `LRC_REPORT_DIR`); `doctor.py` reports them.
- `report["reviewer_status"]` per song, reviewers OK/total in the batch summary, `--strict-reviewers` (exit 4), `LRC_SUBPROCESS_TIMEOUT_SECONDS` (7200), `LRC_LOCK_TIMEOUT_SECONDS` (1800), helper workers exit with their parent.
- One writer for the report-level review summary; `scripts/report_diff.py` for report regression gates.
- `F4-ctc-final-disagreement` demoted from REVIEW to UNTRUSTED (REVIEW rows 596 → 574 on the pool, TRUSTED rows unchanged).
- `.gitignore` no longer drops `scripts/*.py`.

### Install

In the existing v1.4.0 folder, `git fetch` and `git merge --ff-only origin/main` (or overwrite the repository files with this release). Requirements and `setup.ps1` are unchanged. After updating, run once:

```
.venv\Scripts\python.exe scripts\migrate_model_cache.py --src outputs\cache\model-outputs --in-place
```

This adds schema-2 entries next to the existing ones; where a migrated entry lands on a schema-1 file's own path the original is first copied to `outputs\cache\model-outputs\_schema1_backup\`, so nothing is lost. The next run then replays from the migrated cache.
