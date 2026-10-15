# Wide-region 備援的區域閘門（2026-10-15）

## 結論
- 「wide 時間正確卻沒套用」的主因**不是**逐列順序檢查，而是 Central 的半徑 3 區域旗標。
  - wide 重播的窗口會一路延伸到最近的可靠錨點，所以重播並通過檢查（admitted）的列，常常落在 Central 的半徑 3 旗標之外。
  - 這些列從來沒有進入提升流程。
- 修正：只要列是 provisional_unresolved，而且帶有已通過檢查的 `ctc-local-window-wide` 候選，就可以提升，不再看半徑 3 旗標。
  - 其他條件全部不變：位移要大於 0.5 s、順序要嚴格介於前後列之間、提升後重播驗證、selected_valid 不能被 wide 取代、仍標為未驗證。

## 證據（Central 重播，39 首，輸入與 726A4413 逐位元相同）
- 改動前後只有 6 列改變，全部變好，沒有一列變壞，也沒有順序破壞。

| 列 | 改前誤差 | 改後誤差 |
|---|---|---|
| c04 #5 | +2.36 | −0.10 |
| c17 #32 | −3.80 | −0.01 |
| c17 #34 | −0.58 | +0.03 |
| c17 #35 | −1.14 | +0.02 |
| c34 #7 | +0.75 | +0.02 |
| c34 #21 | −1.92 | +1.11 |

- 39 首中，wide 時間與現行時間差超過 0.5 s 的列共 15 列（`_review/_opus_scratch/opus3_wide_opps_20261015.py`）：
  - wide 錯的 4 列（c05 #63、#64，c09 #24 −20.4 s，c13 #15）在 Central 裡都是 selected_valid。既有規則本來就不准 wide 取代有效列，所以仍然擋住。
  - c34 #20 是 selected_valid，但錯 +1.03 s。依同一條規則不動。
  - c04 #12 的 wide 時間不符合順序（前一列沒有替代時間），所以不動。
  - c39 #24–#26 有旗標，卻被前面 #21–#23 擋住：那三列錯 +19～+25 s，而且附近沒有任何候選。
    - 我試過整段聯合排序（DP），也救不回來：#23 沒有任何早於 161.6 s 的候選。
    - 那一段的 wide 重播把 #21–#23 擠在 157.5–158.7 s 的無人聲區，被 vocal-absent 拒掉。
    - 這是生成問題（第 2 類），不是選擇問題。

## 整段聯合排序為什麼不做
- 在真實 Central 輸入上，聯合排序除了上面 6 列之外，沒有多救任何一列。
- 先前在報告層做的離線模擬顯示它會壞 6 列，c09 #24 −20 s 也在其中；那個模擬把 selected_valid 誤當成未定列。
- 結論：不加複雜度。

## 新工具：Central 重播（不用 GPU，改完程式一秒內看 39 首）
1. `python -X utf8 _review/opus3_central_capture_all_20261015.py _review/_opus_scratch/central_capture 2`
   - cache-only 跑每首的 production 指令，把最後一次 reviewer-validity 的 `build_final_timing_state` 輸入存成 pickle。
   - 約 12 分鐘。輸出的 candidate.lrc 與 726A4413 逐位元相同。
   - 只要 Central 之前的階段沒改，就不用重抓。
2. `python -X utf8 _review/opus3_central_replay_20261015.py _review/_opus_scratch/central_capture`
   - 用目前的 `scripts/auto_lrc.py` 重跑 Central，列出每個變動列，並用 BK 算前後誤差（BK 只用來計分）。
   - 加上 `LRC_REPLAY_SCRIPTS=<另一份 scripts>` 可以測 worktree；加上 `LRC_REPLAY_REGION=1` 會印出區域與 wide 細節。
- 範圍限制：只看 Central 決策。Central 之後的階段（postcentral tail、報告重評）要靠 cache-only 39 首確認。

## 其他
- `test_lrc13_collapsed_local_window.py` 單獨執行時，以前會因為磁碟上的模型快取而有 3 個假失敗（全套件執行時沒事）。
  - 現在在模組頂端關掉快取，和 `test_model_cache_replay.py` 的做法一樣。
