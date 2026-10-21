# Lyricsee v2 — verified sources with OOGG fallback

Drag supported audio files onto `Lyricsee.cmd` / `Lyricsee.bat`, or pass audio and `.m3u8` paths to `lyricsee.py`. Lyricsee prefers a verified published translation; when none is available it probes OOGG at `127.0.0.1:6969` and then Sister at `100.87.97.121:6969`. If neither endpoint is online, it writes the verified original only.

## Source rules

- LRCLIB originals must match the recording's title, artist and duration.
- Google is used to discover existing lyric/translation pages. Bahamut/marumaru originals must agree with the independently checked source, or, without LRCLIB, two different allowed sites must agree on at least 85% of unique lyric lines.
- The program joins a vetted web translation to *exactly matched original lyrics*. It does not invent missing translations, copy site menus, or treat arbitrary search snippets as lyrics.
- If no checked human/community translation exists, OOGG may create an `oogg_draft`. The exact endpoint is recorded in provenance. If OOGG is unavailable, output remains `original_only`.
- Original-only lyrics recovered from the DF playlist are staged separately in `rebuild_no_llm_20261008\original_only`. They are not translated and are not automatically sent into LRC tools.
- Bilingual TXT has the format `original  translation` (exactly two ASCII spaces delimiting two fields). Existing TXT remains untrusted until re-audited. LRC outputs and certified BK are not touched.

## Rebuild DF without any model

From the `lyricsee` directory:

```powershell
python .\rebuild_df_nollm_20261008.py --workers 5
```

This reads the 269-track DF audit, skips 29 certified BK references, resumes existing results, and writes **separate staged files and provenance**. Outputs:

- `rebuild_no_llm_20261008\bilingual`: verifiable source plus available published translation.
- `rebuild_no_llm_20261008\original_only`: verified lyrics only; Chinese translation still missing.
- `rebuild_no_llm_20261008\sources`: source/evidence and hashes, no full lyrics.
- `report\DF_NO_LLM_REBUILD_STATE_20261008.jsonl` and `report\DF_NO_LLM_REBUILD_SUMMARY_20261008.json`: every song's outcome.

For an individual audio file:

```powershell
python .\lyricsee.py "D:\Music\Artist - Song.flac"
```

## Translation and endpoint selection

The main CLI uses this order: verified web translation → OOGG draft → verified original only. OOGG does not supply source lyrics; it only translates already verified original rows.

Without an override, OOGG probes these endpoints in order and uses the first one exposing a model:

1. `http://127.0.0.1:6969`
2. `http://100.87.97.121:6969`

When `--endpoint` or `LYRICSEE_OOGG_ENDPOINT` is set, Lyricsee uses only that endpoint and does not fall back to the defaults. `--no-translate` prevents all OOGG calls. The selected endpoint is saved as `oogg_endpoint` in per-track provenance and run reports.

```powershell
python .\lyricsee.py "D:\Music\Artist - Song.flac"
python .\lyricsee.py --endpoint http://100.87.97.121:6969 "D:\Music\Artist - Song.flac"
python .\lyricsee.py --no-translate "D:\Music\Artist - Song.flac"
python .\lyricsee.py --dry-run "D:\Music\playlist.m3u8"
```

## Sister RTX 5080 two-lane translation — live 2026-10-08

Sister at `100.87.97.121:6969` exposes **two inference slots** on a **single loaded Gemma 4 12B**. The user measured **9.7/16 GiB VRAM** during single-lane inference. Two-lane mode has passed a two-song end-to-end test, with both resulting original-named TXT and .org backups hash-validated; this does NOT provide live VRAM readings. Keep **2 maximum concurrent songs** and the existing **4-line inference batches**, never start a second llama-server process.

```powershell
python -u .\\process_remaining_verified_df_20261008.py --sister-only --endpoint http://100.87.97.121:6969 --parallel 2 --run
```

Only the model calls run concurrently; source provenance, state ledger and .org publication are serialized in the primary Python process. Successful songs are checkpointed and skipped on restart after SHA verification. GPU allocation failure or 503 slot pressure reduces concurrency to 1. If Sister GPU memory usage rises beyond a comfortable ceiling (for example 13–14 GiB of 16), switch back to `--parallel 1`; `llama.cpp /slots` does not expose reliable VRAM metrics, so Lyricsee cannot enforce that threshold automatically. Previously certified BK and missing-source songs remain untouched.

## Publishing draft TXT while retaining exact old .org

After source verification and optional OOGG **first-pass translation**, you can publish a hash-checked draft using the original TXT filename and preserve its previous wrong translation, byte-for-byte, at `.org`:

```powershell
# Preview all completed staged OOGG translations
python .\\publish_oogg_drafts_preserve_org_20261008.py --all-staged

# Explicitly publish, with historical source sidecar backed up too
python .\\publish_oogg_drafts_preserve_org_20261008.py --all-staged --run
```

New file: `lyricsee\\txt\\<song>.lyrics.txt`; original: `lyricsee\\txt\\<song>.lyrics.txt.org`. Historical `report\\<song>.lyrics.sources.json` is similarly preserved with `.org` extension. Translation is a **draft only**: provenance marks `translation_semantics_human_checked=false`, `lrc_timing_verified=false`; no approval is entered into the LRC admission ledger.

Duplicate audio filenames in the flat `txt` directory are held back to avoid overwriting unrelated songs. 29 certified BK tracks and 62 tracks with no verified original are never touched by this publishing script. New OOGG batch translation uses 4-row chunks and **defers the second QA pass** until the later verification phase.

**Current first-draft progress (2026-10-08):** 19 verified-original songs have new bilingual TXT under the original filename; 18 historical TXT files were moved byte-for-byte to their `.lyrics.txt.org` backups. One song had no historical TXT. All translations are *unapproved drafts*, and the admission gate remains 0. The detailed per-song audit is `report/DF_OOGG_GPT6_ORG_FINAL_AUDIT_20261008.json`.

GPT-6 assistant-first-pass translations are recorded in `report/DF_GPT6_TRANSLATION_DRAFTS_20261008.json` and promoted by `publish_gpt6_drafts_preserve_org_20261008.py --run`. This is separate from `--provider oogg` translation and does not use the user's GPU. Existing manual-pass legacy bilingual candidates at indices 165/199 are held under `translation_staging_20261008/legacy_review`, not overwritten or auto-approved.

The batch translator reads exact SHA-256-locked original-only staging and writes new bilingual files under `translation_staging_20261008\<provider>` with provenance and `human_translation_qa=false`. Neither source TXT nor user audio/LRC is overwritten. Machine translation remains pending human semantic QA and cannot pass the production admission gate automatically.

For synchronized originals already confirmed against the audio recording, LRCLIB timestamps can be recovered without ASR/CTC or any language model:

```powershell
python .\stage_lrclib_timing_without_llm_20261008.py
```

Results are **review-only LRCs** under `rebuild_no_llm_20261008\lrc_timing_review`, with hash/timing evidence in `report\DF_NO_LLM_TIMING_STATE_20261008.jsonl`. Original-only tracks remain original-only, and never gain invented Chinese. Translation quality, timestamps and music-directory deployment still need acceptance. **No automatic legacy mass replacement or fabricated LRC timing.**
