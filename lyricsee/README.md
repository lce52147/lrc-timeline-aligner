# Lyricsee — verified sources with OOGG fallback

Drag supported audio files onto `Lyricsee.cmd` / `Lyricsee.bat`, or pass audio and `.m3u8` paths to `lyricsee.py`. Lyricsee takes the original lyrics from uta-net first (then UtaTen and J-Lyric) and cross-checks them against each other, LRCLIB and lyric pages (see Source rules); it prefers a verified published translation, and when none is available it probes OOGG at `127.0.0.1:6969` and then Sister at `100.87.97.121:6969`. If neither endpoint is online, it writes the verified original only.

## Source rules

Original lyrics are fetched in this order and cross-checked before anything is written:

1. **uta-net (歌ネット) first.** Title search (`/search/?target=so&type=in&keyword=`), then the song page's `div#kashi_area` only; page furniture and credits are dropped. A row is accepted only when its title matches the recording's title (identical, or identical after stripping `feat.`/bracket additions or a `A/B` slash variant) **and** its artist matches the tag, the album or the folder name. `Call you` is never satisfied by `I Call Your Name`, `花がら` never by `花がらもようの便箋に`. A row with the exact title but no artist evidence (covers, romanised tags such as `Remioromen` for レミオロメン) is not guessed: when LRCLIB supplies an independent text, the row whose 歌い出し matches it is fetched and still has to agree in full; without such a reference a single row is kept only as a weak candidate and several rows are rejected.
2. **UtaTen (うたてん) second.** Same rules (`/search?title=`, then the song page's `div.hiragana` with furigana removed); it is also publisher-licensed, so uta-net and UtaTen agreeing counts as two independent sources, and UtaTen alone can verify a song uta-net does not list (for example 革命前線 / ツユ).
3. **J-Lyric third.** Exact-title search (`/search.php?kt=&ct=1`), lyric from `p#Lyric`; same title+artist rule. It lists some songs the other two lack (天球のMúsica / Ave Mujica). Accented Latin letters are folded when titles are compared (Música = Musica).
4. **LRCLIB and lyric/translation pages are backups.** LRCLIB must still match title, artist and duration (±15 s). Web search (`web_search.py`, plain HTTP) discovers Bahamut/marumaru/publisher pages: up to 16 deduped queries (`"<title>" 歌詞`, `<title> <artist> 歌詞`, 翻譯, 中文歌詞, lyrics, then `site:` queries for gamer/mojim/uta-net/utaten/j-lyric/lyrical-nonsense/vocaloidlyrics), each sent to the first engine that answers: Google (one try per run, it answers HTTP clients with a JavaScript wall), DuckDuckGo HTML, Yahoo! JAPAN, Bing HTML. An engine that shows a challenge or answers 429 rests (Google for the run, the others 180 s, doubling on repeats up to 15 min); two timeouts in a row also rest it. About one request per 2 s per host (Yahoo 8 s). Search stops once two full-length pages agree and one of them is a translation post (or the gamer.com.tw query has run). music.163.com is never searched or fetched. Every query can be logged with `SearchSession(log_path=...)`.
5. **Verification.** Every output row and its provenance say `verified` or `unverified`, which sources were used, and how many lines differ:
   - `verified (two_source_agreement)`: two independent sources (different sites) carry the same lyric. Notation differences do not count: 二人/ふたり, katakana/hiragana, full/half width, punctuation, old-form kanji, numerals, romanised fills, ラララ / la la la / ah lines, backing vocals in parentheses. One different word (黒い過去 vs 暗い過去), or a missing lyric line, does count.
   - `verified (trusted_source_sanity)`: a publisher site (uta-net, UtaTen or J-Lyric) alone, title and artist matched, and the content passes sanity checks (≥ 8 lines, ≥ 6 distinct lines, text amount plausible for the audio duration, no page furniture). The differing lines of any other source are listed in the provenance (`cross_sources[].diff_lines`) and counted in the run table (`diff_lines`, `notation_diffs`).
   - `unverified (single_source)`: only one non-publisher source (for example LRCLIB alone). The TXT is written so it can be reviewed, but provenance has `source_verified: false`, and the track is redone on the next run instead of being treated as existing.
   - `no_original`: nothing matched title and artist, or sources disagree and none is trusted. Nothing is written; the run table lists which rows were rejected and why.
6. The program joins a vetted web translation to *exactly matched original lyrics*. It does not invent missing translations, copy site menus, or treat arbitrary search snippets as lyrics.
7. If no checked human/community translation exists, OOGG may create an `oogg_draft`. The exact endpoint is recorded in provenance. If OOGG is unavailable, output remains `original_only`. If uta-net or LRCLIB is temporarily unavailable after retries, the row is `source_unavailable`.
8. Original-only lyrics recovered from the DF playlist are staged separately in `rebuild_no_llm_20261008\original_only`. They are not translated and are not automatically sent into LRC tools.
9. Bilingual TXT has the format `original  translation` (exactly two ASCII spaces delimiting two fields). Existing TXT remains untrusted until re-audited. LRC outputs and certified BK are not touched.

Run table columns added for verification: `verification`, `verified_by`, `sources` (e.g. `uta-net+lrclib`), `diff_lines` (real differences against the best other source), `notation_diffs`. The same values are at the top of each `report\<stem>.sources.json`, with the full evidence under `verification` (every source considered, its identity match, sanity metrics, and per-line differences by line number and short fragment, never whole lyrics).

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

`LYRICSEE_OOGG_ENDPOINTS` can replace that candidate list with semicolon-separated endpoints. When `--endpoint` is set, Lyricsee uses only that endpoint and does not fall back to the candidate list. The older singular `LYRICSEE_OOGG_ENDPOINT` remains accepted as a compatibility override. `--no-translate` prevents all OOGG calls. The selected endpoint is saved as `oogg_endpoint` in per-track provenance and run reports.

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

New file: `lyricsee\\txt\\<song>.lyrics.txt`; original: `lyricsee\\txt\\<song>.lyrics.txt.org`. Historical `report\\<song>.sources.json` is similarly preserved with `.org` extension. Translation is a **draft only**: provenance marks `translation_semantics_human_checked=false`, `lrc_timing_verified=false`; no approval is entered into the LRC admission ledger.

Duplicate audio filenames in the flat `txt` directory are held back to avoid overwriting unrelated songs. 29 certified BK tracks and 62 tracks with no verified original are never touched by this publishing script. New OOGG batch translation uses 4-row chunks and **defers the second QA pass** until the later verification phase.

**Current first-draft progress (2026-10-08):** 19 verified-original songs have new bilingual TXT under the original filename; 18 historical TXT files were moved byte-for-byte to their `.txt.org` backups. One song had no historical TXT. All translations are *unapproved drafts*, and the admission gate remains 0. The detailed per-song audit is `report/DF_OOGG_GPT6_ORG_FINAL_AUDIT_20261008.json`.

GPT-6 assistant-first-pass translations are recorded in `report/DF_GPT6_TRANSLATION_DRAFTS_20261008.json` and promoted by `publish_gpt6_drafts_preserve_org_20261008.py --run`. This is separate from `--provider oogg` translation and does not use the user's GPU. Existing manual-pass legacy bilingual candidates at indices 165/199 are held under `translation_staging_20261008/legacy_review`, not overwritten or auto-approved.

The batch translator reads exact SHA-256-locked original-only staging and writes new bilingual files under `translation_staging_20261008\<provider>` with provenance and `human_translation_qa=false`. Neither source TXT nor user audio/LRC is overwritten. Machine translation remains pending human semantic QA and cannot pass the production admission gate automatically.

For synchronized originals already confirmed against the audio recording, LRCLIB timestamps can be recovered without ASR/CTC or any language model:

```powershell
python .\stage_lrclib_timing_without_llm_20261008.py
```

Results are **review-only LRCs** under `rebuild_no_llm_20261008\lrc_timing_review`, with hash/timing evidence in `report\DF_NO_LLM_TIMING_STATE_20261008.jsonl`. Original-only tracks remain original-only, and never gain invented Chinese. Translation quality, timestamps and music-directory deployment still need acceptance. **No automatic legacy mass replacement or fabricated LRC timing.**

## 歌詞審核頁面（未驗證候選）

### 開啟審核頁面

審核伺服器啟動後，在瀏覽器開啟提供的 URL（通常為 `http://127.0.0.1:PORT`），即可進入歌詞候選審核頁面。

### 頁面說明

### 歌曲卡片
頁面以卡片形式顯示需要審核的歌曲，每張卡片包含：
- **歌曲名稱**：音檔的檔名幹名（不含副檔名）
- **候選數量**：該歌曲有多少個歌詞候選項

### 候選項顯示
每個候選項展示以下資訊：
- **來源**：歌詞的發布來源（例：uta-net、utaten、jlyric 等）
- **主機**：來源網站的域名
- **行數**：歌詞包含的行數
- **標題匹配**：歌詞頁面標題是否與搜尋的歌曲標題相符
- **藝人匹配**：歌詞頁面藝人名稱是否與搜尋的藝人名稱相符

### 審核操作

### 採用（確認此候選）
點擊「採用」按鈕，表示接受該候選歌詞：
- 系統會將該候選歌詞保存為已驗證原文，路徑為 `txt/<stem>.lyrics.txt`
- 同時生成來源驗證檔 `report/<stem>.sources.json`，標記為人工確認（human_approved）
- 狀態會被標記為 `needs_review`，後續執行 lyricsee 時將進行翻譯處理（不需要 --overwrite）

### 都不對（拒絕所有候選）
點擊「都不對」按鈕，表示拒絕該歌曲的所有候選項：
- 系統會記錄該歌曲已被用戶拒絕
- 後續執行 lyricsee 時，除非指定 --overwrite，否則不會再次詢問該歌曲

### 決定記錄

所有審核決定都會記錄在 `report/review_decisions.jsonl` 中，每條決定包含：
- 時間戳記
- 歌曲名稱
- 決定類型（accept 或 reject）
- 若為採用，則記錄被接受的候選編號

### 翻譯已確認的歌曲

已採用的歌詞可透過標準 lyricsee 流程進行翻譯：

```bash
python lyricsee.py <音檔路徑> --output <輸出目錄>
```

系統會檢測到已驗證的原文，自動跳過候選搜尋步驟，直接進行翻譯。如需重新搜尋和審核，使用：

```bash
python lyricsee.py <音檔路徑> --output <輸出目錄> --overwrite
```

### 技術細節

- 審核結果儲存在輸出目錄的 `candidates/` 目錄下
- 接受的歌詞格式與標準 lyricsee 輸出一致
- 所有文字檔案編碼為 UTF-8

## Auto-pick（候選自動選擇）

`autopick.py` 在人工審核之前先自動決定：把每個候選清洗成原文語言（去掉翻譯行、羅馬字、標頭、空行，全形/片假名/標點正規化），用字元 bigram 做相似度（`candidate_text.py`），再與 Whisper 對音檔的轉錄比對（precision = 候選內容真的有被唱到的比例；coverage = 轉錄內容被候選涵蓋的比例；score = 調和平均）。

```powershell
# 轉錄快取放在 --cache-dir，不會寫到音檔旁邊；一次一首，只有這一步用 GPU
python autopick.py run --out <output> --audio-map <stem->音檔.json> [--stem <stem> ...]
python autopick.py calibrate --txt-dir <已驗證txt> --audio-map <map.json> --cache-dir <dir> --result <out.json>
```

自動採用條件（驗證標記 `auto_consensus`，分數寫進 `report/<stem>.sources.json` 的 `autopick`）：
最佳 ASR 分數 ≥ `ASR_FLOOR` 且領先下一個「不同」候選至少 `ASR_MARGIN`（與最佳相似度 ≥ `AGREE_SIM` 的是同一份歌詞，不算對手）；並且有另一個獨立（不同 URL）、ASR 也過底線的候選與它相符，或最佳分數單獨 ≥ `ASR_STRICT_FLOOR`。
太短的候選（清洗後字數低於 `MIN_CHARS_PER_SEC` × 歌曲秒數或 `MIN_CHARS_ABS`）視為片段，不參與。其餘維持 needs_review。門檻皆為 `autopick.py` 頂部的具名常數。

校準（12 首已驗證歌詞）：正確歌詞對自己音檔的 ASR 分數 0.25–0.62、清洗後 2.6–6.4 字/秒；錯誤歌曲最高 0.13。同曲不同版本的候選相似度 0.46–0.98，不同曲 ≤ 0.19。

### 歌詞翻譯貼文（translation post）

標題寫明「歌詞翻譯」（`歌詞翻譯`／`歌词翻译`／`中日歌詞`…）且包含這首歌名的頁面（例如巴哈姆特 【歌詞翻譯】「歌名」- 歌手），內容是逐行「原文＋中文翻譯」。`save_candidates(identity=…)` 把它標成 `translation_post`，並存 `candidate_<n>.pairs.json`（翻譯行外層的 【】「」（） 已去除；翻譯與原文同行時 `original【中文】` 也會拆開）。
- 採用：專屬貼文本身就是完整歌詞，**不套用字數/秒數的片段規則**，只要求原文 ≥ `MIN_POST_LINES`（6）行且 ≥ 70% 有翻譯。ASR 只是加分訊號：轉錄是真的在唱日文（≥ 30 個日文字、沒有重複字幕署名/迴圈）卻與貼文分數 < `WRONG_SONG_ASR`（0.13）時才否決（`translation_post_contradicted_by_asr`）；Whisper 轉錄是幻覺時不當閘門。理由記為 `translation_post_title_match`。
- 翻譯：採用的歌詞若有原文一致（相似度 ≥ `AGREE_SIM`）的翻譯貼文，無論原文來自哪個來源，都寫入人工翻譯（`translation_method = page_human`，`report/<stem>.sources.json` 記 URL、作者、對齊行數）；貼文沒翻到的行才由 OOGG 補（`page_human+oogg_fill`）：`python translation_posts.py apply --out <output> [--stem …] [--oogg]`。
