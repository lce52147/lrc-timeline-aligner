# LRC 1.3 — 完整 Central 反例與聲學候選證據衝突修正（2026-10-11）

分支：lrc-1.3。未 push、未合併 main、BK 與音樂檔案唯讀。BK 時間只在輸出凍結後用於評分。

## 為何骨架層通過，但完整 7 首 A/B 失敗
- 原始 7 首 14/14 完整流程已完成；正式 gate NOT_PASS_FINAL7_AB。
- 唯一新增 >1 秒錯行：04 影色舞 #10。BK 46.38、原全曲 CTC 46.36、骨架局部 CTC 46.41，但 Central 最終採用 45.32 的 WhisperX fused 候選，且 timing_trusted=false。
- 全 7 首計分：Segment PCO@0.2 210，CTC 183；Segment >1 秒 58，CTC 85；新增 >1 秒 1，新增倒退 0。
- 初次嘗試的 neutral provisional-only 改動，完整重跑結果逐行未變，已還原，不列入正式程式。

## 本次通用修正
- 位置：scripts/auto_lrc.py 在 candidate_sets 組成前；獨立 pure helper scripts/segment_peer_outlier.py。
- 只處理當前已啟用 segment skeleton 的候選集合。要求「ctc-vocal-onset-confirmed」與「raw-vocal-independent-fusion」這兩個既有候選時間距離不超過 0.5 秒、均具有聲學與直接起音證據，才把落在該兩候選窗口外超過 0.5 秒的 whisperx-vocal-independent-fusion 候選排除出 Central 裁決。
- 排除記錄留在 segment_peer_fusion_exclusions。未製造時間戳、未改 Central 選擇演算法、未把不可信行改成 trusted。
- 原始《影色舞》完整 report 中具備這類證據衝突的行有 5、10、11、19（純報告候選時鐘判定，不讀 BK）。

## 已完成驗證
- 5 個新專項測試 + 全套 742 tests OK、2 skipped。
- 第二次單首**完整 Central**輸出已凍結：
  - BK #10=46.38，舊 Segment=45.32，新 Segment=46.10，CTC=46.36。新候選仍 timing_trusted=false。
  - 新 Segment 全曲 PCO@0.2 45/52，>1 秒 0，新增 >1 秒 0。
  - 對照全曲 CTC：20/52，>1 秒 28；舊 Segment：47/52，>1 秒 1。
  - **新版本相較舊 Segment 少 2 行落在 0.2s 容許差內**，沒有隱瞞這項精度退步。
- 新 gate 採規格的「相對全曲 CTC 不得啟用退步及新增 >1 秒」條件；先前 smoke 工具額外的「不得低於舊 Segment PCO」已修正。
- 第二輪全 7 首 14 個模式目前在序列執行；最終 39 首 A7 尚未驗收。**總驗收 NOT_PASS/PENDING**。
- 所有候選篩選門檻已在實際 A/B 再跑前寫入程式。這組測試案例參與過開發期診斷，不能宣稱獨立 holdout。
