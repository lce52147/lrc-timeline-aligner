## LRC Timeline Aligner v1.8.0（草稿，版本號待確認）

審核頁改版與速度改善。對齊結果、報告格式和存檔流程與 v1.7.2 相同（`LRC_ALGORITHM_REVISION` 未變）。

### 新增
- **審核頁音量滑桿**：頂欄以 dB 顯示（0 ～ −100 dB，與 foobar2000 相同），滑桿曲線讓 0 ～ −30 dB 佔大部分行程；滾輪每次 ±1 dB，`M` 靜音；音量記在瀏覽器（`localStorage`）。
- **審核句子依風險分組**（`review_presentation.review_risk`）：每句 REVIEW 在載入報告時分成 `high` / `normal` / `low`。高風險：帶模型不一致碼，或理由碼超過六個且沒有 WhisperX 逐句證據；可略過：五個以上來源在 0.3 秒內一致，且 0.5 秒外沒有強競爭者。可略過的句子預設收起、不算剩餘句數；`<歌名>.review-decisions.jsonl` 每列多了 `risk`。36 首驗證集：誤差 > 0.5 秒的 45 句，高風險含 38 句、可略過含 2 句。
- **歌曲清單列出所有跑過的歌**：不帶參數的 `Review LRC.bat` 與批次結束後打開的頁面都開整個 `outputs\reports`；左側分「待確認」與可收合的「已確認」。
- `scripts\file_digest.py`：以 stat 身分記住檔案 SHA-256，64 MiB 以上的檔案另存 `cache\runtime-identity\file-sha256.json`，供 auto_lrc、multimodel consensus 和 WhisperX reviewer 共用。

- **可略過層新規則（R5 審聽工作量研究）**：`review_presentation.line_fit` 用寫出時間與歌詞長度算「前一句唱得完」「這一句塞得進下一句之前」「至少一側相鄰句在 2 s 內」「CTC token span ≤ 預估的 2 倍」，符合的一般層句子歸入可略過（`risk_basis="fit"`）。36 首 BK：多略過 47 句（44 句 ≤0.2 s、3 句 ≤0.5 s、0 句 >0.5 s；留一驗證同樣 0 句 >0.5 s），要人工看的句子 161 → 114；可略過層 >0.5 s 的漏抓維持 2 句。DF 178 首：要人工看 1096 → 824 句。每句多 `risk_basis` 與 `fit`，`review-decisions.jsonl` 多 `risk_basis`。
- **審核頁操作**：「換句時自動播放」與可略過層「全部維持」按鈕（來源 `batch-keep`）預設開啟，`?` 視窗可關，設定記在瀏覽器；新快捷鍵 `Shift+A`／`Shift+B` 一鍵採用 A／B 並跳下一句；候選列加入 `alignment_hypotheses`（36 首裡大錯行正解出現在 A／B 的 13 → 14 句，對的句子選項數不變）。改前基線：每句中位 5.6 s（維持 3.9 s、修改 10.5 s）。

### 變更
- **審核頁載入加速**：`review_server` 每次載入只列一次報告目錄、每份報告只解析一次（`_ReportIndex`），`reload()` 重用已解析的資料。原本每個 FLAC 目標都重新解析全部報告（O(n²)）。180 首合成報告（每份約 540 KB）由 FLAC 路徑載入：250.8 秒 → 3.8 秒；資料夾目標 6.8 秒 → 3.4 秒；整個報告目錄 4.2 秒 → 3.0 秒。三種情況的頁面資料與排除清單皆 byte-identical。
- **最終 pass 加速**：不再每首重算模型權重摘要；WhisperX 對齊模型查詢（另開直譯器 import whisperx）記在 `cache\runtime-identity\whisperx-align-model.json`，直譯器、helper、語言、hub cache 任一改變即失效，`unresolved` 不記；尾音證據 context 的驗證結果與候選區段分析依物件身分記住。
- 長歌名不再讓 reviewer 暫存路徑超過 Windows MAX_PATH。
- lyricsee 輸出改名為 `<歌名>.txt`，並自動搬移舊的 `.lyrics` 檔。

### 測試
- 新增 `scripts\test_final_pass_memo.py`（7 題）：尾音 context 驗證依物件記住且有上限、WhisperX 模型查詢跨程序記住並在 hub／helper／語言改變時重探、`unresolved` 不寫入、hub 指紋追蹤 snapshots 與 refs。已加入 `check_public.py` 的公開安全測試集。
- `scripts\test_file_digest.py`：memo 讀寫合併、損毀檔案視為空、寫入失敗不拋錯（+2 題）。
- `scripts\test_review_page.js`：音量滑桿（dB 對應、夾限、滾輪、M 靜音、記憶）、歌曲欄待確認／已確認分組與收合、風險分組與可略過句子的收合和計數（+3 組）。
- `scripts\test_lrc14_review_server.py`：多個 FLAC 目標時每份報告只解析一次（+1 題）；每句帶 `risk_basis`／`fit` 且寫進審核紀錄（+1 題）；候選列含 alignment hypotheses（+1 題）。
- `scripts\test_review_presentation.py`：速率估計、`line_fit`、分級 basis（+3 題）。`scripts\test_review_page.js`：設定、自動播放、Shift+A／B、整批維持（+1 組）。

### 驗證
- Linux（開發機）：`python -m unittest discover -s scripts -p "test_*.py"` 1310 題，`discover -s lyricsee` 124 題，`node scripts\test_review_page.js`。Linux 上有 9 題因環境失敗，與本次改動無關、在 origin/main 同樣失敗：`test_align_lrc_wrapper`（3 題，沒有 `powershell`）、`test_core_logic` 的 V12Hubert 四題（沒有 `pykakasi`）、lyricsee 的 `test_playlist_paths_resolve_relative_entries_and_preserve_order`（Windows 反斜線路徑）和 `test_render_lyrics_converts_translation_to_traditional_chinese`（沒有 OpenCC）。
- Windows（發版前）：`python -m unittest discover -s scripts -p "test_*.py"` 全綠、`scripts\doctor.py` 顯示 v1.8.0、`node scripts\test_review_page.js`。
