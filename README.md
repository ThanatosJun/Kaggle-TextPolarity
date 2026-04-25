# 文字極性分類

Kaggle 競賽 — 對混合領域評論（電影、商品、遊戲）進行二元情感分類（正面 / 負面）。

---

## 專案結構

```
.
├── config/
│   └── config.yaml          # 中央參數設定檔（所有實驗皆從此讀取）
├── datasets/
│   ├── train_2022.csv
│   ├── test_2022.csv
│   └── sample_submission.csv
├── docs/                    # 問題描述與流程筆記
├── EDA/                     # 探索性分析筆記本（Method 0–2）
├── experiments/             # 實驗記錄（已加入 .gitignore）
│   ├── results.csv          # 所有實驗的彙總指標
│   └── {exp_id}/
│       ├── metrics.json     # 該次訓練的 Train / Val 指標
│       └── config_snapshot.yaml
├── models/                  # 訓練好的模型檔案（已加入 .gitignore）
│   └── {exp_id}/
│       ├── xgb.json
│       ├── count_vec.pkl    # 僅在 LDA 啟用時存在
│       └── lda.pkl
├── results/                 # 預測輸出（已加入 .gitignore）
│   └── submission_{exp_id}.csv
├── src/
│   ├── preprocess.py        # 文字清理（Track A 供 TF-IDF，Track B 供 BERT）
│   ├── features.py          # 特徵提取（BERT 嵌入、LDA、NLI）
│   └── experiment.py        # 寫入 metrics.json 與 results.csv
├── train.py                 # 訓練流程
├── predict.py               # 推論流程
└── requirements.txt
```

---

## 環境設定

### 前置需求

- Python 3.10+
- 建議使用支援 CUDA 的 GPU（RTX 30xx 以上）
- Conda 或 venv

### 安裝套件

```bash
pip install -r requirements.txt
```

> `requirements.txt` 內含 PyTorch CUDA 12.6 的安裝來源。若 CUDA 版本不同，請自行調整 `--extra-index-url`。

---

## 操作流程

### 1. 設定實驗參數

每次訓練前先編輯 [config/config.yaml](config/config.yaml)：

```yaml
experiment:
  id:   exp_007_my_experiment      # 唯一 ID，決定模型與指標的儲存位置
  name: 實驗描述

features:
  lda:
    enabled: false   # 是否加入 LDA 主題特徵
  nli:
    enabled: true    # 是否加入 Zero-shot NLI 領域特徵
```

主要開關說明：

| 開關 | Config 路徑 | 效果 |
|------|------------|------|
| LDA 特徵 | `features.lda.enabled` | 加入 CountVec + LDA 主題向量 |
| NLI 領域特徵 | `features.nli.enabled` | 加入零樣本領域分類機率（電影/商品/遊戲） |
| 運算裝置 | `global.device` | `auto` / `cpu` / `cuda` |

### 2. 訓練模型

```bash
python train.py
```

執行步驟：

1. 讀取 `datasets/train_2022.csv`
2. 按 80 / 20 分層切分 Train / Val
3. 對兩個 split 分別前處理（Track A 供 CountVec，Track B 供 BERT）
4. 在 Train split 上 fit 特徵提取器，Val split 僅 transform
5. 以 Train split 訓練 XGBoost，評估兩個 split 的指標
6. 儲存指標至 `experiments/{exp_id}/metrics.json`
7. 將結果 append 至 `experiments/results.csv`
8. 用全量資料重新訓練，將模型存至 `models/{exp_id}/`

輸出範例：

```
Device: cuda | Val ratio: 0.2
Features — LDA: False | NLI: True

[ Train split ]
  Accuracy     0.9925
  F1           0.9925

[ Val split   ]
  Accuracy     0.7250
  F1           0.7250

✅ Artifacts saved: models/exp_006_NLI
```

### 3. 預測並產出 Submission

```bash
# 使用 config.yaml 中設定的實驗
python predict.py

# 指定特定實驗的模型
python predict.py --exp-id exp_005_ablation_no_lda
```

輸出儲存至 `results/submission_{exp_id}.csv`。

> **注意**：`--exp-id` 只覆蓋模型路徑，特徵開關（LDA、NLI）仍讀取 config.yaml。  
> 請確認開關設定與當時訓練一致，可參考 `experiments/{exp_id}/config_snapshot.yaml`。

---

## 特徵流程

```
原始文字
  │
  ├─ Track A（清理供 CountVec/LDA 使用）
  │    └── X_topics  (n, n_topics) 或 (n, 0) 若停用
  │
  └─ Track B（清理供 BERT 使用，還原縮寫）
       ├── X_bert    (n, 384)  — all-MiniLM-L6-v2
       ├── X_meta    (n, 2)    — ? 數量、! 數量
       └── X_nli     (n, 3)    — 零樣本領域分數，或 (n, 0) 若停用

np.hstack([X_bert, X_topics, X_meta, X_nli])
  └── XGBoost 分類器
```

---

## 實驗紀錄

| 實驗 ID | 描述 | Val Acc | Val F1 | Kaggle |
|---------|------|---------|--------|--------|
| exp_003 | BERT + XGBoost（加入正則化） | 0.7125 | 0.7124 | — |
| exp_004 | + CountVectorizer LDA | 0.7175 | 0.7171 | — |
| exp_005 | 消融：移除 LDA | 0.7375 | 0.7374 | — |
| exp_006 | + NLI 領域特徵 | 0.7250 | 0.7250 | — |

完整紀錄見 [experiments/results.csv](experiments/results.csv)。

---

## 新增實驗流程

1. 修改 `config.yaml`，設定新的 `experiment.id` 並調整參數
2. 執行 `python train.py`
3. 查看 `experiments/results.csv` 確認指標
4. 執行 `python predict.py` 產出 submission
5. 上傳 Kaggle 後手動將 `kaggle_score` 補回 `results.csv`
