# 文字極性分類

Kaggle 競賽 — 對混合領域評論（電影、商品、遊戲）進行二元情感分類（正面 / 負面）。

---

## 專案結構

```
.
├── config/
│   └── config.yaml              # 中央參數設定檔（所有實驗皆從此讀取）
├── datasets/
│   ├── train_2022.csv
│   ├── test_2022.csv
│   └── sample_submission.csv
├── docs/
│   ├── experiment_guide.md      # 詳細操作手冊（新實驗者請先閱讀）
│   └── promblem.md              # 問題分析與策略筆記
├── EDA/                         # 探索性分析筆記本（Method 0–2）
├── experiments/                 # 實驗記錄（已加入 .gitignore）
│   ├── results.csv              # 所有實驗的彙總指標
│   └── {exp_id}/
│       ├── metrics.json
│       └── config_snapshot.yaml
├── models/                      # 訓練好的模型（已加入 .gitignore）
│   └── {exp_id}/
│       ├── xgb.json                      # 單模型實驗
│       ├── xgb_{model_id}.json           # Stacking 的各 base model
│       ├── xgb_{model_id}_full.json      # Stacking 全量版（predict 用）
│       ├── meta_model.pkl                # Stacking meta model（val 評估用）
│       └── meta_model_full.pkl           # Stacking meta model（predict 用）
├── results/                     # Submission 輸出（已加入 .gitignore）
│   └── submission_{exp_id}.csv
├── src/
│   ├── preprocess.py            # 雙軌文字清理（Track A / Track B）
│   ├── features.py              # 特徵提取（BERT 嵌入、LDA、NLI）
│   ├── stacking.py              # OOF 生成與 meta model 訓練
│   ├── experiment.py            # 寫入 metrics.json 與 results.csv
│   └── utils.py                 # 共用工具（set_seed 等）
├── train.py                     # 單模型訓練（快速消融實驗用）
├── train_stacking.py            # Stacking 訓練（正式提交用）
├── predict.py                   # 單模型推論
├── predict_stacking.py          # Stacking 推論
└── requirements.txt
```

---

## 快速開始

### 環境安裝

```bash
conda create -n textpolarity python=3.10 -y
conda activate textpolarity
pip install -r requirements.txt
```

### 正式提交流程（Stacking）

```bash
# 1. 設定實驗 ID
# 編輯 config.yaml → experiment.id

# 2. 訓練
python train_stacking.py

# 3. 預測
python predict_stacking.py

# 4. 上傳 results/submission_{exp_id}.csv 至 Kaggle
```

### 快速消融實驗（單模型）

```bash
# 編輯 config.yaml → experiment.id + features 開關
python train.py
python predict.py --exp-id {exp_id}
```

詳細說明請見 [docs/experiment_guide.md](docs/experiment_guide.md)。

---

## 兩種訓練模式

| 腳本 | 用途 | 目前最佳 Val F1 |
|------|------|----------------|
| `train.py` | 快速驗證單一想法（換模型、加特徵） | 0.8200（exp_007） |
| `train_stacking.py` | 正式提交，多模型互補 | **0.8375**（exp_009） |

**建議工作流程：** 用 `train.py` 先確認新特徵/模型有效，再整合進 `train_stacking.py` 的 base model 清單。

---

## 特徵流程

### 單模型（train.py）

```
原始文字
  ├─ Track A（供 CountVec/LDA）
  │    -lrb-/-rrb- 移除，只保留英文字母
  │    → X_topics (n, n_topics) 或 (n, 0)
  │
  └─ Track B（供 BERT/VADER）
       -lrb-/-rrb-   → ( )
       num_num        → number
       num_extend     → number
       縮寫修復        → doesn't / won't ...
       標點空白修復
       → X_bert (n, 768) + X_meta (n, 2) + X_nli (n, 3) 或 (n, 0)

np.hstack([X_bert, X_topics, X_meta, X_nli]) → XGBoost
```

### Stacking（train_stacking.py）

```
原始文字
  ├─ all-mpnet-base-v2     → (n, 770) → 5-fold XGBoost → OOF (n, 2)
  └─ cardiffnlp/roberta    → (n, 770) → 5-fold XGBoost → OOF (n, 2)
                                                              ↓
                                              hstack → (n, 4) → LogisticRegression
```

---

## 實驗紀錄

| 實驗 ID | 描述 | Val F1 | Kaggle |
|---------|------|--------|--------|
| exp_003 | MiniLM + XGBoost（正則化） | 0.7124 | — |
| exp_004 | + CountVectorizer LDA | 0.7171 | — |
| exp_005 | 消融：移除 LDA | 0.7374 | 0.71404 |
| exp_006 | + NLI 領域特徵 | 0.7250 | — |
| exp_007 | 換 mpnet（768-dim） | 0.8200 | — |
| exp_008 | cardiffnlp 情感模型 | 0.7875 | — |
| **exp_009** | **Stacking（mpnet + cardiffnlp）** | **0.8375** | **0.76749** |
| exp_010 | cardiffnlp + VADER（對照組） | 0.7849 | — |
| exp_011 | mpnet + VADER | 0.8024 | — |
| exp_012 | Stacking + num_num/num_extend → number | 0.8350 | — |
| exp_013 | Stacking baseline（模組化重構後基準） | 0.8375 | — |
| **exp_014** | **Stacking + PCA 32-dim 降維** | **0.8375** | — |

> exp_010 / exp_011：VADER 無額外幫助，mpnet 的 768-dim embedding 已涵蓋情感信號。  
> exp_012：文字清洗改動在誤差範圍內，保留作為後續實驗的標準前處理。  
> exp_013：程式碼模組化重構（src/ 拆分為 5 個獨立模組），功能與 exp_009 相同，確認重構無迴歸。  
> exp_014：PCA 將 770-dim 壓縮至 32-dim（保留 ~47% 變異），Val F1 與基準持平，XGBoost 訓練速度提升。

完整紀錄見 [experiments/results.csv](experiments/results.csv)。

---

## 新增實驗

### 單模型實驗

1. 修改 `config.yaml` — 設定新 `experiment.id`、調整 `sentence_transformer.model_name` 或特徵開關
2. 執行 `python train.py`
3. 查看 `experiments/results.csv` 確認 Val F1

### Stacking 實驗

1. 修改 `config.yaml` — 設定新 `experiment.id`、修改 `stacking.base_models` 清單
2. 執行 `python train_stacking.py`
3. 執行 `python predict_stacking.py` 產出 submission
4. 上傳 Kaggle，將 `kaggle_score` 手動填回 `results.csv`

### 程式碼流程
可以請你先幫我整理程式碼流程跟檔案嗎，主要分成以下，放置於 src/ 內，並用train_stacking.py 銜接流程。：
1. data_preprocess.py：資料前處理，包含 text 文字的處理。
2. text_tansform.py：利用 LLM 模型將資料集文字轉換為 embeeding 向量。
3. dimension_decrease.py：利用 PCA 或其他降維方式，降低 embedding 維度並刪除雜訊。
4. model_train.py：訓練 XGBoost 或其他模型用以預測該筆回應正向與負向的輸出機率。
5. model_stack.py：訓練一個 logisticRegression 或其他模型，將前一步驟各個模型輸出的正向與負向機率維輸入，做出最終對於回應的正向與負向結果判斷。
