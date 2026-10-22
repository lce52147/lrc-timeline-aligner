## LRC Timeline Aligner v1.7.2

文件與審核頁小改版。對齊結果、報告格式和存檔流程與 v1.7.1 相同。

### 變更
- README 改寫成中文使用說明：需要什麼、安裝、產生 LRC、審核、快捷鍵、產生的檔案、常見問題和已知限制。
- 版本紀錄移到 `CHANGELOG.md`（v1.7 以後用中文）；演算法、評估方法和維護說明移到 `docs/TECHNICAL.md`。
- 審核卡片下方的「上一句」「下一句」分成兩行，各自顯示時間和歌詞。

### 驗證
- Windows：`python -m unittest discover -s scripts -p "test_*.py"`、`scripts\doctor.py` 顯示 v1.7.2、`node scripts\test_review_page.js`。
