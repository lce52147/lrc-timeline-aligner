## LRC Timeline Aligner v1.9.0（草稿）

v1.9.0 是審核頁、lyricsee 與內部整理的版本，對齊準確度與 v1.8.0 相同；段落級重對齊以關閉的診斷功能隨附。

段落級重對齊（segment realign）預設關閉。五輪 PC replay（2026-10-28 完成）沒有修到任何一行，預設維持關閉；`--segment-realign auto` 只作診斷用。`LRC_ALGORITHM_REVISION` 改為 `2026-10-25-lrc-v2.0-segment-realign`（字串裡的 v2.0 是開發期代號，不是版號）。這個階段關閉，或歌沒有被它動到時，它不影響 LRC；審核頁新增「整首高風險」標記。

### 新增
- **段落級重對齊**（`scripts\segment_realign.py`、`ctc_window_realign.apply_segment_realign`，`auto_lrc` 仍轉出）：全曲 CTC 與 local window 之後、Central 之前，用歌詞結構重算每一行該在哪裡——同一首副歌不同次出現的對應行間隔必須一樣（驗證集上差的中位數 0.02 s、95 百分位 0.33 s）、兩個起唱點之間的有人聲秒數要對得上這一行的長度（整首中位速率的 0.62–1.6 倍免費）、時間必須單調遞增；有 raw ASR 佐證或 CTC 分數夠高的行是錨點，其他行可以在錨點之間移動。被推開超過 1 秒的連續幾行組成一個 run，在前後沒被動到的行之間的時間窗（最多 6 個 run、每個 1–60 秒）再跑一次 bounded CTC，通過右緣堆積與人聲缺席檢查、且跟全曲 CTC 給這一行的時間差 0.5 秒以上的起唱點寫成 `segment-realign-ctc` 候選交給 Central（差不到 0.5 秒只算確認、不產候選）；候選的 identity 標成 unavailable，不能自己通過 Central 的最小證明、也不能被聯合邊界觀測認證，只有 Raw／WhisperX 佐證或 Whisper 探針能抬起它。不改任何時間、不碰輸出；錨點不到 35% 的歌整首不碰；沒有 run 的歌零模型呼叫。預設關閉，開啟只作診斷用。
- **整首高風險（Choir 類）**（`review_presentation.song_risk`）：結構圖錨點不到 20%，或至少一半（≥ 8 行）的句子需審，審核頁標題旁顯示「整首高風險 N/M 句需審」並在工作區加一行說明。對齊本身不變，只是提醒整首聽過再逐句確認。
- **可信評估集**：45 份校正 LRC 依「和任一工具輸出到百分秒完全相同的行比例」分組：15 份獨立（562 行）；22 份衍生（相同行 > 10%），其中 B2w3 有配對的 21 首、813 行當回歸；8 份對不到任何工具輸出、不進評估（其中远航星的告别、The Day the Starlight Arrived、ビビッド 3 份是 0.1 秒格手標）。評估以 15 份獨立校正為主，衍生校正只當回歸（不能變差），審聽紀錄（37 行）另計。起點：independent bad（> 0.5 秒）16 行、derived 40 行（Choir ‘S’ Choir 17 行）。

### 審核頁
- **匯出到音檔的驗證與歸檔。** 驗證改為直接比對音訊內容的位元組（FLAC 比對音訊幀、M4A 比對 `mdat`，只有無法解析容器時才退回 ffmpeg 解碼比對）；匯出成功後該首 `.lrc` 預設搬到 `checked\`（環境變數 `LRC_EXPORT_ARCHIVE_DIR`，見 README）。
- **整行刪除。** 歌詞裡不該存在的一行，可以在該句的卡片按「刪除」，標成「待刪除」，按 `Ctrl+S` 時才連同其他修改一起從 LRC 拿掉，「復原刪除」可取消。刪除不會影響其他句的「已確認」標記與草稿；刪掉會讓前後兩句併成一句時存檔被拒絕；存檔出錯時 LRC 還原成存檔前的樣子；被刪的行記在編輯紀錄裡。
- **可略過隱藏與免審。** 可略過的句子預設不列出（`?` 視窗可勾選顯示）；只剩可略過句子的歌標「免審」，排在待確認的最後面。
- **間隔過短。** 沒被標為待審、但到下一句的間隔不到這首歌一般間隔中位數三分之一的句子，加上「間隔過短」小標記並歸入「一般」組；只是提醒，不改時間，有這種句子的歌不標「免審」。
- **直接輸入時間。** 選定那句的時間是輸入框，可打 `mm:ss.xx`、`m:ss.x`、純秒數，或 `+0.1`／`-0.05` 的相對值；`Enter` 套用、`Esc` 取消，格式不對會標紅且不改任何東西。
- **羅馬字與讀音覆寫表。** 目前這句下方顯示羅馬字（只供參考，不影響對齊用的讀音），讀音依「手動 > 振假名（UtaTen）> 字典 > 分析器」決定，並標出來源（自動／字典／手動）；點羅馬字那一行就能直接改，結果存在該首旁邊，也可以選擇加進所有歌共用的 `scripts\reading_overrides.tsv`；`scripts\furigana_audit.py` 會拿振假名比對快取的辨識結果，讀音可能不同的句子標徽章並寫出 `romaji_audit.md`。
- **草稿與固定連接埠。** 還沒儲存的修改會自動存成草稿，重新整理或重開後還原（LRC 沒變才還原，舊草稿可選「捨棄」或「仍然套用」）；兩個分頁同時修改同一首歌不會互相蓋掉。審核頁連接埠固定為 58400（環境變數 `LRC_REVIEW_PORT`，被占用時往後找並在頁面上通知）。
- **熱更新與關閉伺服器。** 程式更新時，閒置中的審核伺服器原地換成新版，連接埠不變、草稿保留（`scripts\review_hot_update.py`）；頁面頂欄多「關閉伺服器」按鈕（要確認，存檔、匯出或重掃進行中會拒絕）。審核伺服器一定有可見的視窗。
- **移除唯讀狀態。** 一律以 LRC 檔為準：報告對不上時改用 LRC 本身的時間標記分組，只有找不到、已匯出或無法解析的 LRC 才唯讀；手標（`*`）的歌不再被藏起來。
- **更快、更穩。** 開歌、匯出、播放都不再觸發整個資料夾重掃，儲存只寫目前開著的那一首，伺服器自己的寫入不會讓頁面重新整理；匯出的等待時間縮短，狀態列用黃（匯出中，顯示秒數）、綠（已寫入並驗證）、紅（失敗）三色。
- **括號歌詞標「可能沒唱」。** 歌詞括號裡的內容（和聲、合唱）可能沒被唱出來，審核頁在那句旁加安靜的「可能沒唱（括號）」小標記（由 lyricsee 產生，只是提醒，不改時間）。
- **和聲候選與只聽和聲軌。** 括號歌詞行多一顆「和聲候選」按鈕，按下去跳到和聲軌上找到的起點並試聽，不會改這句的時間；頂欄的「只聽和聲軌」只播放和聲軌（同一條時間軸）。候選由 `scripts\harmony_candidates.py` 事先產生。候選按鈕也固定依 1 2 3（維持排最後）排列，不再依時間排序。
- **審核理由寫實話。** 每句要嘛寫「偵測到問題」（附訊號與數字），要嘛寫「未自動確認」（附缺少什麼證據）；候選全部一致、沒發現問題的句子歸入可略過。
- **試過後移除。** 按 `T` 設定起點、波形上的候選刻度，加入不久就移除了；全域「每句起點提前 100 ms」試過後預設改回 0（`--start-lead-ms`／`LRC_START_LEAD_MS` 仍可開啟）。

### 對齊
- **阿拉伯數字讀成漢數字。** 只用在對齊用的文字，畫面上的歌詞不變；含阿拉伯數字的歌，對齊結果可能與 v1.8.0 不同。
- **接受 `.m4a`。** 對齊、批次流程與審核頁都能處理 M4A。

### lyricsee
- **一律輸出繁體（台灣）中文。** 原文歌詞只轉真正的簡體字，翻譯一律轉。
- **搜尋更廣。** 多個搜尋引擎、多組查詢（Yahoo JP、Bing、`site:`）並用檔案自己的標籤與專輯資料夾名；`--cdp`（或 `LYRICSEE_CDP=1`）才會用執行中的 Chrome 做最後一輪 Google 搜尋。
- **自動挑候選。** 依跨來源一致與辨識結果比對自動選歌詞；審核頁可接受或拒絕候選，沒驗證的候選留給人工看，接受時寫出清理過的原文歌詞。
- **翻譯貼文與翻譯後端。** 接受歌詞翻譯貼文並寫入人工翻譯（要確認歌手）；機器翻譯改為 Gemini 優先、其次 OOGG（`--translator gemini|oogg|auto`），`sources.json` 記錄用了哪個後端與模型。
- **補漏句。** 主要來源漏掉的短副歌或結尾句，若另一個一致的來源有，就補回原本的位置（最多 3 行、不取自部落格頁、有記錄）。
- **歌詞品質閘門。** 偵測並去掉譯者註解與曲目列，檢查翻譯串到隔壁行、順序、日文異體字、未翻譯與長度比；`sources.json` 記錄 `needs_review` 與問題清單。

### 內部
- **`auto_lrc.py` 拆成模組。** 先拆成 15 個模組，再把其中 5 個細分成 16 個更小的模組，全部是純搬移，`auto_lrc` 仍轉出原有名稱；演算法來源 hash 改為涵蓋全部模組檔（任一檔案變動都會讓快取 identity 改變）。
- **測試。** `test_core_logic.py` 拆成依主題命名的 `test_core_*` 檔；新增拆分守門測試（模組清單、名稱集合、import 無循環、patch 落在讀取該名稱的模組等）。
- **文件。** README 重畫演算流程圖，標出每個步驟用的模型。

### 變更
- CLI 新增 `--segment-realign {auto,off}`，**預設 off**（五輪 replay 沒有修到任何一行，維持關閉）；`--segment-realign auto` 或環境變數 `LRC_SEGMENT_REALIGN=auto` 才啟用，只作診斷用，環境變數蓋過 CLI。預設關閉時這個階段不改任何一行，報告只多 `segment_realign: {status: off}`，但skeleton 為 SEGMENT_CURRENT 的歌（第二輪 39 首中的影色舞、遠雷、远航星）不跑 local window 與本階段，報告沒有這個區塊。
- `segment-realign-ctc` 加入 local CTC 候選的 Raw/WhisperX 佐證來源，以及位移 ≥ 3 秒時的 local Whisper 探針（沒有 `segment-constrained-ctc` 候選時才接手；原有行為不變）。
- 報告新增 `segment_realign`（skeleton 為 SEGMENT_CURRENT 的歌（第二輪 39 首中的影色舞、遠雷、远航星）不跑 local window 與本階段，報告沒有這個區塊）：status（off / skipped / weak-anchors / no-runs / no-candidates / candidates / error）、row_status、structure（錨點數、速率、副歌模板對數）、flagged_entries、每個 run 的 entries / proposed / 時間窗 / 候選數 / 拒絕原因、generated、revision。

### 數字
v1.9.0 的對齊準確度與 v1.8.0 相同；「36 首抓到 6 行大錯」是離線結構圖評估的數字，生產路徑這 6 行目前碰不到（My Dearest 錨點不足被跳過、影色舞走 SEGMENT_CURRENT）。

雲端評估（36 首 BK，`F:\_RuntimePrompt\LRC20\reports\V2_P1_P2_STRUCTURE.md`）：
- 結構圖 flagged：independent 2 行（1 真錯／1 誤報）、derived 不含 Choir 7 行（6 真錯／1 誤報）、Choir 7 行（4／3）；run 時間窗包住 BK 15/16。
- 抓到的 ≥ 1 秒大錯 6 行（My Dearest e23/e24/e36/e39/e40、影色舞 e2）；抓不到的是 2–5 秒整段平移、最後一行提早、沒有副歌模板的中段錯行。
- PC replay（auto vs off，39 首 B2w_input 其中 36 首有 BK，加 20 首 DF 抽樣；`F:\_RuntimePrompt\LRC20\reports\V2_P5_REPLAY.md`）：第二輪（`2b3bac5c`）bk **39/39 byte-identical**、independent ok/mid/bad 528/18/16、derived 750/23/40，與 off 和 B2w3 基線完全相同、fixed 0 / new bad 0、pipeline 時間 +1.3%；df 19/20 相同（1 行差 2 cs）。第三輪（`d7524a7b`）bk 38/39、fixed 0 / **new bad 1**（終点 e20 偏 1.03 秒）、pipeline +15%（該首 2 個 run 在 baseline pass 冷啟動 helper，多 158 秒）；df 19/20、new bad 0。第四輪（`ae8e845e`，0.5 秒確認門檻，只重跑 13 首有候選的歌）：bk 仍 38/39、同一行 new bad——門檻擋掉 20 個候選中的 14 個確認，但 e21 的候選離階段輸入 3.0 秒、擋不到；df 20/20。根因：e21 的候選 79.616（跟校正一致）宣告 identity supported，被 previous+current 聯合邊界觀測認證成 `ctc-joint-boundary-consensus`、該行變成 selected，e20 的多模型共識視窗從 32 秒縮成 10 秒，共識把 e20 從 74.50 改成 75.53。第五輪（`2a689a4a`，候選 identity unavailable，只重跑 5 首仍有候選的歌）：bk **39/39 byte-identical**、independent 528/18/16 與 off 相同、fixed 0 / **new bad 0**；df 20/20；終点 e21 的 provisional 候選回到 `ctc-local-window-wide`（identity unavailable、provisional），與 off 一致。驗收：byte-identical ✓、沒有新 bad ✓、independent bad 砍半 ✗（16 → 16，五輪都沒修到任何一行）。

### 測試
- `scripts\test_segment_realign.py`（+6：一致的歌不動、副歌行被拉回且時間窗包住真值、倒退行往前、錨點不遠移、錨點太少 BLOCKED、長 run 收窄）。
- `scripts\test_v2_segment_realign_stage.py`（新，8 題，fake helper：位移副歌行 → 一次 bounded pass、上下文行不產候選、hypothesis 欄位、timestamp 不變；一致的歌零呼叫；人聲缺席拒絕；weak-anchors 不碰；孤立首 token 不當起唱點；確認候選（差 < 0.5 秒）不寫入；off 開關與環境變數；信任代理）。
- `scripts\test_review_presentation.py`（+3：song_risk）。`check_public.py` 納入新測試，Linux 全綠。
- Windows（發版前）：`python -m unittest discover -s scripts -p "test_*.py"`、`scripts\doctor.py`、P5 replay 的 `P5_SUMMARY.md`。

### 已知限制
- independent 組 16 行 bad 有 11 行是 0.5–1 秒的行級偏移，段落級方法不處理；「bad 砍半」要靠行級邊界修正。
- 生產環境的錨點代理（raw 一致 → TRUSTED、CTC 分數 ≥ 0.12 → UNTRUSTED）：第一首 replay 歌用 0.25 只剩 8/35 個錨點，所以降到 0.12；raw 反對不再把高分 CTC 行降級（評估集上這種行 267 對 / 6 錯，降級讓 My Dearest 只剩 13/43 個錨點而被跳過）。誤報多寡以 P5 報告的 `flagged_entries` 為準。
- Choir 類只標記；換 stem／WhisperX 當主來源的實驗排在 P5 之後。
- Central 的證明門檻：唯一真錯且有候選的夢遊 e14（候選 85.70、校正 85.45、現值 83.16）跟該行其他候選一樣被 `previous-tail-ownership-unknown` / `insufficient-minimum-proof` 拒絕——結構佐證不是 Central 認得的證明。要讓這條路真的修到行，得改 Central 的採用規則；那是另一個要單獨 replay 的決定。
- 多模型共識對視窗敏感：終点 e20 在 32 秒視窗下沒有 override，10 秒視窗下七個模型一致落在 75.53（校正 74.50）。這是這個階段加入之前就有的行為，只是被 e21 的候選觸發；候選改成 identity unavailable 後不再符合聯合邊界認證的資格，共識本身沒有改。
- 聯合邊界觀測（`ctc-joint-boundary-consensus`）把同一個 MMS CTC 模型的兩次 bounded pass 當成獨立佐證，只要求 parent 候選宣告 identity；這也是這個階段加入之前就有的規則，值得另外看。
- 成本：有 run 的歌在 baseline pass 冷啟動 bounded CTC helper（終点 2 個 run 多 158 秒 wall）；沒有 run 的歌零成本。pipeline 合計 +1.3%（第二輪）／ +15%（第三輪，含該首）。
