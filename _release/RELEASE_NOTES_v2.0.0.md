## LRC Timeline Aligner v2.0.0（草稿；版本號與發佈由 EvanOG 決定）

段落級重對齊（segment realign），預設關閉、以 `--segment-realign auto` 啟用。`LRC_ALGORITHM_REVISION` 改為 `2026-10-25-lrc-v2.0-segment-realign`。關閉時與沒有被新階段動到的歌，LRC 與 v1.8.0 完全相同；報告多一個 `segment_realign` 區塊。審核頁新增「整首高風險」標記。

### 新增
- **段落級重對齊**（`scripts\segment_realign.py`、`auto_lrc.apply_segment_realign`）：全曲 CTC 與 local window 之後、Central 之前，用歌詞結構重算每一行該在哪裡——同一首副歌不同次出現的對應行間隔必須一樣（驗證集上差的中位數 0.02 s、95 百分位 0.33 s）、兩個起唱點之間的有人聲秒數要對得上這一行的長度（整首中位速率的 0.62–1.6 倍免費）、時間必須單調遞增；有 raw ASR 佐證或 CTC 分數夠高的行是錨點，其他行可以在錨點之間移動。被推開超過 1 秒的連續幾行組成一個 run，在前後沒被動到的行之間的時間窗（最多 6 個 run、每個 1–60 秒）再跑一次 bounded CTC，通過右緣堆積與人聲缺席檢查、且跟全曲 CTC 給這一行的時間差 0.5 秒以上的起唱點寫成 `segment-realign-ctc` 候選交給 Central（差不到 0.5 秒只算確認、不產候選）；候選的 identity 標成 unavailable，不能自己通過 Central 的最小證明、也不能被聯合邊界觀測認證，只有 Raw／WhisperX 佐證或 Whisper 探針能抬起它。不改任何時間、不碰輸出；錨點不到 35% 的歌整首不碰；沒有 run 的歌零模型呼叫。
- **整首高風險（Choir 類）**（`review_presentation.song_risk`）：結構圖錨點不到 20%，或至少一半（≥ 8 行）的句子需審，審核頁標題旁顯示「整首高風險 N/M 句需審」並在工作區加一行說明。對齊本身不變，只是提醒整首聽過再逐句確認。
- **可信評估集**：37 份校正 LRC 裡 22 份是從工具輸出改來的（與某次輸出到百分秒相同的行數比例 > 0.10）。評估改以 15 份獨立校正（562 行）為主，22 份衍生校正（813 行）只當回歸（不能變差），審聽紀錄（37 行）另計。起點：independent bad（> 0.5 s）16 行、derived 40 行（Choir ‘S’ Choir 17 行）。

### 變更
- CLI 新增 `--segment-realign {auto,off}`，**預設 off**（2.0 replay 驗收前先不開，EvanOG 在 `19b4b53` 定的）；`--segment-realign auto` 或環境變數 `LRC_SEGMENT_REALIGN=auto` 才啟用，環境變數蓋過 CLI。預設關閉時每首歌的 LRC 都與 v1.8.0 相同，報告只多 `segment_realign: {status: off}`。
- `segment-realign-ctc` 加入 local CTC 候選的 Raw/WhisperX 佐證來源，以及位移 ≥ 3 秒時的 local Whisper 探針（沒有 `segment-constrained-ctc` 候選時才接手；原有行為不變）。
- 報告新增 `segment_realign`：status（off / skipped / weak-anchors / no-runs / no-candidates / candidates / error）、row_status、structure（錨點數、速率、副歌模板對數）、flagged_entries、每個 run 的 entries / proposed / 時間窗 / 候選數 / 拒絕原因、generated、revision。

### 數字
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
- 多模型共識對視窗敏感：終点 e20 在 32 秒視窗下沒有 override，10 秒視窗下七個模型一致落在 75.53（校正 74.50）。這是 2.0 之前就有的行為，只是被 e21 的候選觸發；候選改成 identity unavailable 後不再符合聯合邊界認證的資格，共識本身沒有改。
- 聯合邊界觀測（`ctc-joint-boundary-consensus`）把同一個 MMS CTC 模型的兩次 bounded pass 當成獨立佐證，只要求 parent 候選宣告 identity；這也是 2.0 之前就有的規則，值得另外看。
- 成本：有 run 的歌在 baseline pass 冷啟動 bounded CTC helper（終点 2 個 run 多 158 秒 wall）；沒有 run 的歌零成本。pipeline 合計 +1.3%（第二輪）／ +15%（第三輪，含該首）。
