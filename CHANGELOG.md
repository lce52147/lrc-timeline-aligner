# 更新紀錄

v1.7 以後用中文記錄；v1.6 以前保留原本的英文內容。

## v1.8

### v1.8.0（草稿，版本號待確認）
- **審核句子的「可略過」層多一條規則（R5）。** 一般層的句子如果跟前後兩句銜接得合理（上一句唱得完、這一句塞得進下一句之前、至少一側相鄰句在 2 秒內、CTC 辨識長度不超過預估的兩倍），也歸入可略過。36 首有校正歌詞的歌上多略過 47 句（44 句在 0.2 秒內、3 句在 0.5 秒內、沒有超過 0.5 秒的），要人工看的句子 161 → 114；可略過層漏掉的大錯維持 2 句。每句多了 `risk_basis`（agreement／fit／model-disagreement／many-codes／default），審核紀錄也寫進去。
- **審核頁更快的操作（R5，預設關閉的設定）。** `?` 視窗裡可開「換句時自動播放」（Enter／K／↑↓ 跳到下一句就先播開頭）和可略過層的「全部維持」按鈕；新快捷鍵 `Shift+A`／`Shift+B` 直接採用 A／B 並跳下一句。候選列加入對齊時權衡過但沒選的 alignment hypotheses。
- **審核頁音量滑桿。** 頂欄以 dB 顯示音量（0 ～ −100 dB，與 foobar2000 相同），滾輪每次 ±1 dB，`M` 靜音；音量記在瀏覽器裡。
- **審核句子依風險分組。** 每句需審的句子在載入時分成「高風險」「一般」「可略過」：帶模型不一致碼、或理由碼超過六個且沒有 WhisperX 逐句證據的是高風險；五個以上來源在 0.3 秒內一致、且 0.5 秒外沒有強競爭者的是可略過。可略過的句子預設收起、不算在剩餘句數裡；每筆審核紀錄都帶 `risk` 欄位。36 首驗證集上，誤差超過 0.5 秒的 45 句有 38 句落在高風險、2 句落在可略過。
- **歌曲清單列出所有跑過的歌。** 不帶參數的 `Review LRC.bat` 和批次結束後打開的頁面都開整個報告目錄；左側分「待確認」和可收合的「已確認」。
- **審核頁載入加速。** 每次載入只解析每份報告一次（原本每個 FLAC 目標都重新解析全部報告，O(n²)）。180 首合成報告由 FLAC 路徑載入：250.8 秒 → 3.8 秒，頁面資料 byte-identical。
- **最終 pass 加速。** 檔案摘要依 stat 身分記住（`scripts\file_digest.py`，64 MiB 以上的檔案另外存在 `cache\runtime-identity`），同一個程序不再重讀 3 GB 的 Whisper 模型；WhisperX 對齊模型查詢結果記在磁碟，只要直譯器、helper、語言和 hub cache 沒變就不再重新探測；尾音證據的 context 驗證和候選區段分析依物件身分記住。
- **長歌名修正。** reviewer 暫存路徑不再超過 Windows MAX_PATH。
- **lyricsee。** 輸出改名為 `<歌名>.txt`，舊的 `.lyrics` 檔自動搬移。
- **lyricsee 歌詞來源改為 uta-net 優先並交叉驗證。** 先以標題搜尋 uta-net，只接受標題與歌手都對上的那一列，再取 `div#kashi_area`；LRCLIB 與歌詞頁改為備援。兩個獨立來源同意（假名／漢字、ラララ、標點、全半形等寫法差異不算，不同字才算）或 uta-net 加內容健全性檢查才記為 `verified`；只有單一來源時記為 `unverified` 並在下次執行重抓。run table 與 provenance 新增 `verification`、`verified_by`、`sources`、`diff_lines`、`notation_diffs`，差異只記行號與片段。抓錯歌（例如 `Call you` 對到 `I Call Your Name`）在標題／歌手關卡就被擋下；同名多列但歌手對不上時，用 LRCLIB 的文字挑出歌い出し相符的那一列，仍須全文同意才採用。兩份歌詞的比對是在整段文字上對齊再回推行號，所以 uta-net 與 LRCLIB 斷行方式不同也不會被當成缺行。
- **測試。** 新增 `scripts\test_final_pass_memo.py`（尾音 context 驗證記憶、WhisperX 模型查詢磁碟記憶、hub 指紋）、`file_digest` 的 memo 讀寫測試、審核頁的音量滑桿／待確認已確認分組／風險分組測試、以及審核頁「每份報告只解析一次」的回歸測試。

## v1.7

### v1.7.2
- README 改寫成中文使用說明：安裝、產生 LRC、審核、快捷鍵、產生的檔案、常見問題和已知限制。版本紀錄移到 `CHANGELOG.md`，演算法與評估細節移到 `docs/TECHNICAL.md`。
- 審核卡片下方的「上一句」「下一句」分成兩行顯示。
- 對齊結果和存檔流程與 v1.7.1 相同。

### v1.7.1
- 審核頁會記錄每一句的審核結果：按「維持」（K）或「接受，下一筆」（Enter）時，在報告旁的 `<歌名>.review-decisions.jsonl` 追加一行，內容是維持或修改、選定的時間，以及這句的狀態和理由碼。不需要按儲存。
- 新增 `scripts\review_decisions_summary.py`：統計每種理由碼最後被維持和被修改的比例，用來依真實結果調整審核門檻。
- 匯入面板的佇列只顯示最新一次和正在跑的工作，「開始審核」只會有一顆。

### v1.7.0
- **在審核頁新增歌曲。** 按「＋ 新增歌曲」，用 Windows 選檔視窗選 FLAC 或資料夾（也可以貼路徑）。面板會顯示每首找到的歌詞檔（順序同命令列：`.lyrics.lrc` → `.lyrics.txt` → `.txt`），缺歌詞可直接貼上，存成 FLAC 旁的 `<歌名>.lyrics.txt`。已有 LRC 的歌要勾「覆蓋」才會重跑。按「開始對齊」後一次跑一首，跑完自動出現在歌曲清單，失敗的會顯示原因和紀錄檔，可以重試。佇列狀態存在 `outputs\jobs`。不支援把檔案直接拖進瀏覽器，因為瀏覽器拿不到檔案的真實路徑。
- **哪裡拖進來就在哪裡審。** 把資料夾、FLAC 或 LRC 拖到 `Review LRC.bat`，就審核並存回那個資料夾裡的檔案，不管歌曲原本在哪裡對齊的。FLAC 或 LRC 被搬走時，會在另一個檔旁邊找；沒有 FLAC 時改播人聲 stem。不帶參數執行 `Review LRC.bat` 只開上一次執行的歌。只有 `LRC bk` 資料夾不會被列入。
- **自動重新整理。** 頁面每 3 秒檢查一次，有新的 FLAC、LRC 變動或新的對齊結果時自動更新，已暫存的修改會保留。
- **審核門檻（B2 階梯，中間檔）。** 寫出結果時，理由碼全部屬於弱理由的 REVIEW 句會降成 UNTRUSTED（未驗證），不會升成 TRUSTED。在 36 首的 B2 測試集上，REVIEW 從 538 句降到 232 句，沒有任何一句時間被改動，誤差超過 1 秒的 37 句裡有 35 句仍留在 REVIEW。報告會記錄 `weak_review_policy`。
- **小修正。** 批次摘要只涵蓋這次排入的歌；空白的審核頁會說明哪些報告被略過和原因；進度數字一致；沒有改動時按 Ctrl+S 會說明；缺少 `ffprobe` 不再讓標題處理當掉。
- 內含 **lyricsee v2**（抓歌詞與 OOGG 翻譯，位於 `lyricsee\`）。

## v1.6

v1.6.1: songs whose LRC no longer exists (`lrc-missing`, typically reports left behind by old test runs) are no longer shown as finished; the song list folds them into one "另有 N 首找不到 LRC" line and the page says why they cannot be reviewed. `scripts/archive_stale_reports.py` lists those reports and, with `--apply`, moves them and their sidecars into `outputs/reports/_archive/`. Alignment unchanged.

Review page redesign. Line timing policy, report format and the save flow are unchanged; v1.6 only changes how a review row is presented and operated.

- **One line at a time.** The page shows the current line as one card: lyric, a one-sentence reason in plain language, the waveform, and at most three timing options, A / B / keep (`1` / `2` / `3`). Candidates closer than 0.25 s are merged into one option; candidates within 0.15 s of the current timestamp fold into "keep". The raw candidate list, internal source names, severity and reason codes sit in a collapsible "詳細" fold.
- **Reason as a sentence.** `row["reason_text"]` is derived server-side from the reason codes by priority (vocal absent → order problem → early opening → detector disagreement → silent risk → repeated line → weak evidence); the per-code labels remain in the fold.
- **Waveform that explains itself.** Markers for the current time, options A/B and the previous/next line; a 0.5 s ruler; when the vocal stem cannot be read the page decodes the FLAC in the browser instead, and when neither works it says why instead of showing an empty box.
- **Progress, not tabs.** A song list with remaining counts replaces the tab wall; the page opens on the first song that still has rows to review; stat tiles show this song's progress, remaining rows here and overall, and unsaved edits; the per-song queue is a list of one-line rows with a verdict chip. `Enter` accepts and moves on, `K` keeps, `U` undoes, `← →` nudge, `Tab` opens the fold, `?` lists the keys.
- **New module** `scripts/review_presentation.py` (`present_options`, `reason_sentence`) with tests; the review server adds `options` and `reason_text` to each row and keeps every previous field.

## v1.5

Architecture update: reproducible runs on any install path, visible reviewer health, less review noise. Line timing policy unchanged.

- **Portable model cache (schema 2).** Cache identities no longer embed absolute paths or machine-specific strings, so a cache copied to another folder or drive replays byte-identically. `scripts/migrate_model_cache.py --src --dst` upgrades an existing cache (add-only; the source is never modified). The r2 reviewer sidecars are now cached too, so a cache-only replay reproduces the full reviewer-validity run instead of silently degrading.
- **`scripts/runtime_paths.py`** is the single source of runtime locations, with environment overrides: `LRC_VENV_PYTHON`, `LRC_WHISPER_CLI`, `LRC_WHISPER_MODEL`, `LRC_FFMPEG`, `HF_HUB_CACHE`/`HF_HOME`, `LRC_MODEL_CACHE_DIR`, `LRC_REPORT_DIR`. `doctor.py` reports them.
- **Reviewer status on every report.** `report["reviewer_status"]` lists each reviewer step with its exit code and status (`OK`, `FAILED`, `MISSING_UNSUPPORTED`, `SKIPPED_DEPENDENCY`, `TIMEOUT`, `LAUNCH_ERROR`); the batch summary shows reviewers OK/total. `--strict-reviewers` makes r2 exit 4 when a step is not acceptable. Subprocesses honour `LRC_SUBPROCESS_TIMEOUT_SECONDS` (default 7200) and GPU lock waits honour `LRC_LOCK_TIMEOUT_SECONDS` (default 1800); the persistent helper workers exit when their parent process disappears.
- **Single writer for the review summary.** `review_required`, its count/percent and `verification_counts` are derived in one place and checked against the per-row reason classes; `scripts/report_diff.py` compares two reports key by key for regression gates.
- **Review noise: F4 demoted.** `F4-ctc-final-disagreement` alone no longer forces a row into REVIEW (audit: it never pointed at a line that needed a human). The code is still recorded; the row's status follows its timing evidence. On the validation pool REVIEW rows drop from 596 to 574 with no change to TRUSTED rows.
- **Repository hygiene.** All `scripts/*.py` and docs are tracked (the old `.gitignore` pattern dropped scripts silently).

Measured on the 39-song / 1487-line validation pool (v1.5 baseline B2, cache-only replay byte-identical 39/39):

| Measure | v1.4.0 (B1) | v1.5.0 (B2) |
|---|---|---|
| Lines within 0.2 s (PCO@0.2) | 1362 / 1487 | 1360 / 1487 |
| Lines off by more than 0.5 s | 59 | 60 |
| Lines off by more than 1 s | 40 | 40 |
| TRUSTED lines off by more than 1 s | 0 | 0 |
| REVIEW / UNTRUSTED / TRUSTED rows | 584 / 390 / 513 | 574 / 329 / 584 |

The small timing difference comes from the r2 reviewers now running (and being cached) on every song; in v1.4 a reviewer failure degraded silently. 19 of 39 LRCs differ from v1.4.0 by at least one line; 5 lines move by more than 0.2 s.

## v1.4

- Review page opens automatically after a batch alignment that produced review rows; `LRC_NO_REVIEW=1` disables it; install paths with spaces are supported.
- Candidates can be listened to and applied by click or digit key; each chip shows how many other candidates agree within ±0.15 s; overflow candidates sit in a collapsible list.
- A/B playback alternates the current and the selected candidate timestamps.
- Vocal-stem waveform strip for the selected row with drag adjustment, plus visible ±0.1 s nudges and JIS-safe bracket keys.
- Plain-language reason text for every review row.
- Transactional save: the LRC, the first-save `.auto.lrc` and the edit log are staged and published together; a failed save leaves the files unchanged.
- Timing and alignment algorithm unchanged from v1.3.1 (39-song cache-only replay byte-identical).

## v1.3

v1.3.1 is an installation and packaging update. It does not change the v1.3 timing or alignment policy. The setup path now builds the repository-local ASR environment, pins the validated dependencies and third-party binaries, can prefetch the production models, and finishes with a doctor check.

v1.3 adds a post-Central multimodel consensus layer and stronger review flagging on top of the v1.2 architecture.

- **Multimodel consensus (post-Central):** rows that Central leaves unresolved are re-aligned by MMS, MMS-1B (eng/jpn), three English CTC models and a Japanese CTC model. A new time is accepted only when at least three model families agree within 0.10 s (MMS required) and the occurrence/boundary guards pass.
- **Review flags:** `model-disagreement` and `silent-risk` mark lines where the models disagree or the timing pattern is risky, so large errors are surfaced for review instead of being written silently.
- **Reproducible reruns:** model outputs are cached by execution identity, so cache-only replays reproduce a run exactly.

Measured on the 39-song / 1487-line manually checked validation pool (cache-only replay of the v1.3.0 release):

| Measure | v1.3.0 |
|---|---|
| Lines within 0.2 s (PCO@0.2) | 1362 / 1487 (91.6%) |
| Lines off by more than 0.5 s | 59 |
| Lines off by more than 1 s | 40, all flagged for review |
| TRUSTED lines off by more than 1 s | 0 |
| Test suite | 1031 passed, 5 skipped |

These results describe this corpus and do not predict accuracy on arbitrary songs. The v1.2 sections below are kept for history.

## v1.2

Architecture rewrite around Central arbitration and independent reviewer evidence (tag `v1.2.0`, 2026-10-04).

- Central becomes the single authority that selects the timestamp written to each lyric row; CTC and Whisper-family systems become evidence producers instead of competing final writers.
- R1 `gross-rescue` and R2 `reviewer-validity` arbiter modes are introduced. R2 can validate only registered soft evidence gaps, using independent HUBP / WhisperX / XLSR evidence, and otherwise fails closed.
- User-facing runs acquire reviewer evidence automatically; reviewer trust labels stay diagnostic and separate from timestamp selection.
- Automatic inference is separated from checked-reference timing; checked timestamps remain available only through explicit assisted modes.
- Validation and reporting are expanded around per-row evidence, review status and reproducible evaluation (see "Current architecture", "Arbiter modes" and "Evaluation method" below).

## v1.1

CTC becomes the primary forced-alignment path and is evaluated alongside the original WhisperX-assisted pipeline (June–July 2026, no release tag).

- Added MMS/CTC forced alignment (`torchaudio.pipelines.MMS_FA`) over the known lyric sequence, with Japanese lyrics romanized for the CTC/Viterbi pass.
- `auto` runs the CTC and the whisper.cpp + WhisperX candidates and selects by backend-aware report quality; near-ties favour CTC because it is constrained to the prepared lyric order.
- Added auditable CTC reports, alignment against isolated vocal stems, conservative acoustic onset correction and experimental vocal-onset refinement.
- Added quantitative checked-song regression gates and alignment summaries.
- Hardened repeated-line and late-ASR cases while preserving the prepared lyric order.

## v1.0

Initial public release of the local timeline aligner (2026-06-16).

- Align prepared lyrics to FLAC without generating or rewriting lyric text.
- Use whisper.cpp ASR anchors plus WhisperX forced alignment for untimed lyrics, with CUDA acceleration when available.
- Reuse checked LRC timing when explicitly supplied and the text order matches.
- Write a same-name `.lrc` together with an auditable alignment report.
- Establish fail-loudly / report-first behaviour and private checked-song regression gates.
