# 參數區（照 `_review/GPT_RUN_TEMPLATE.md` 執行）

```
TAG=20261015_BBED2B54
BASE=20261015_6A14D7CB
SONGS=all
ROWS=c4:3,c4:18,c4:21,c17:14,c17:15,c17:20,c17:21,c34:16,c34:27,c38:14,c38:28,c39:4,c39:20
EXPECT=沿用 6A14D7CB：只有 16 首 collapsed CTC path 可能變，其餘 23 首逐列相同；門檻 PCO@0.2>=1322、>0.5 s<=102、>1 s<=80、TRUSTED>1 s=0
LAUNCH=yes
```

本輪先以 GPU 走完整 39 首 production path，啟用 `LRC_MODEL_CACHE=1` 填滿 `outputs/cache/model-outputs/`。必須保留 GPU round 的 commit identity、summary 對帳、report contract、逐列結果與四項分數。

GPU round 完成後，原樣重跑完整 39 首 production path，環境設為 `LRC_MODEL_CACHE=1`、`LRC_MODEL_CACHE_ONLY=1`（或每個 CLI 加 `--model-cache-only`）。這一輪不得啟動任何模型/helper；cache miss、cache identity mismatch 或 cache-only error 都是明確 `SKIP/FAIL`，不得 fallback 到 GPU，也不得算通過。

重跑仍須完成 production commit、summary reconciliation 與 report contract。用：

```
python -X utf8 scripts/cache_replay_39.py --gpu-manifest <GPU_MANIFEST> --replay-manifest <REPLAY_MANIFEST> --cache-dir outputs/cache/model-outputs
```

只有該工具輸出逐列相同且四項數字完全相同，才可宣告 cache trustworthy：`PCO@0.2`、`>0.5 s`、`>1 s`、`TRUSTED>1 s`。回報 GPU/replay 各自的 commit identity、summary/report contract 結果，以及 candidate-only regression；不要把 inherited failure 寫成全套 PASS。
