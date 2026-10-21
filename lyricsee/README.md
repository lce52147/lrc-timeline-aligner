# Lyricsee — Verified sources, no-LLM default (2026-10-08)

Drag FLAC/MP3/M4A and other supported audio files onto `Lyricsee.cmd` or `Lyricsee.bat`. The launcher **automatically uses verified online translation mode**. It does not call OOGG/Gemini and does not query NetEase.

## Source rules

- LRCLIB originals must match the recording's title, artist and duration.
- Google is used to discover existing lyric/translation pages. Bahamut/marumaru originals must agree with the independently checked source, or, without LRCLIB, two different allowed sites must agree on at least 85% of unique lyric lines.
- The program joins a vetted web translation to *exactly matched original lyrics*. It does not invent missing translations, copy site menus, or treat arbitrary search snippets as lyrics.
- Failure to find a checked human/community translation is a normal, explicit failure; **never substitute LLM text silently**. It does not write a misleading bilingual TXT.
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

## Free-first translation (no paid API)

Lyricsee only retrieves/verifies original lyrics. Optional translation is a separate lightweight API client; the model stays inside the existing OOGG service. **No paid API is selected automatically.**

- **OOGG (default for DF batch):** already running local model at port 6969. No API charge; consumes local GPU resources while translating. No model is loaded by Lyricsee.
- **MyMemory (no key):** real live Japanese-to-`zh-TW` API tested, about **5000 anonymous characters/day shared by source IP**. Lyricsee enforces a conservative **4500 chars per UTC day** using `report/mymemory_free_usage.json`, deduplicates repeated chorus lines, and rejects missing, quota-warning or malformed responses. Good for a few songs only; translation accuracy and Traditional Chinese glyphs need review.
- **Azure Translator F0:** Microsoft lists 2,000,000 characters/month free, but requires an account, free-tier resource and credentials. Not currently connected to Lyricsee and **not automatically used**.
- **DeepL:** prior API Free is closed to new sign-ups; the newer Developer API plan gives a limited total free allocation. Lyricsee currently **refuses paid DeepL keys**; only pre-existing keys ending with `:fx` can access the old free endpoint. No paid fallback exists.

For the existing **178 rebuilt and verified original-only DF songs**, dry-run is the default and the user must specify `--run`:

```powershell
# No API charges; relies on pre-existing local OOGG service
python .\translate_rebuilt_df_20261008.py --max-songs 5
python .\translate_rebuilt_df_20261008.py --provider oogg --max-songs 5 --run

# Anonymous free API; small daily shared quota
python .\translate_rebuilt_df_20261008.py --provider mymemory --max-songs 2 --run
```

For an individual song, the no-key web translation remains the drag-and-drop default; optional MyMemory and OOGG can be selected explicitly:

```powershell
python .\lyricsee.py --mode mymemory "D:\Music\Artist - Song.flac"
python .\lyricsee.py --mode llm --endpoint http://127.0.0.1:6969 "D:\Music\Artist - Song.flac"
```

## Sister remote inference preparation (paused; no Sister IP configured yet)

The local bulk translation batch has been stopped on user request. Its per-song checkpoints, canonical bilingual TXT and byte-identical `.org` backups remain intact. On resume, the script first confirms the successful output SHA before skipping previously published tracks; failed or interrupted tracks are retryable.

Use the exact **Sister Tailscale or LAN IP** (not `127.0.0.1`) and its **llama.cpp OpenAI-compatible port**. Keep the Sister inference server listening only on the appropriate private/Tailscale interface and ensure that port is reachable from the primary PC. Do not expose an unauthenticated model server to the public internet.

```powershell
# Read-only network + health + model checks
python .\\preflight_sister_llm_20261008.py --ip <SISTER_IP> --port 6969

# Optional short real inference smoke test
python .\\preflight_sister_llm_20261008.py --ip <SISTER_IP> --port 6969 --smoke-test

# Dry run; explicitly fail if a local or public endpoint was accidentally used
python .\\process_remaining_verified_df_20261008.py --sister-only --endpoint http://<SISTER_IP>:6969

# Only after verified remote reachability, resume translations
python -u .\\process_remaining_verified_df_20261008.py --sister-only --endpoint http://<SISTER_IP>:6969 --run
```

`--sister-only` denies loopback/public addresses and verifies remote health/model before any translation requests. The `--endpoint` value is forwarded to the OOGG translation client; inference is done on Sister while TXT/backup/provenance stay on this local computer. No user-supplied IP is stored in the project until it is actually provided. The `--smoke-test` is optional and sends one short query only.

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