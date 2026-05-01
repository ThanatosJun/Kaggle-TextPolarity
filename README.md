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
| `train_stacking.py` | 正式提交，多模型互補 | **0.8850**（exp_018） |

> 資料擴增（exp_019/020）：NLPaug fill-mask 擴增（2000→4000 筆）後 val F1 反而下降至 0.8675，增強樣本引入雜訊，目前最佳仍是 exp_018（無增強）。

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

### Stacking（train_stacking.py）— 目前架構（exp_018）

```mermaid
flowchart TD
    IN([/"原始文字評論"/])
    IN --> PP["data_preprocess.py\nTrack B：還原括號・縮寫修復・標點空白"]

    PP --> ENC1["text_transform.py\nall-mpnet-base-v2\n768-dim + 2 meta = 770-dim"]
    PP --> ENC2["text_transform.py\ncardiffnlp/twitter-roberta\n768-dim + 2 meta = 770-dim"]
    PP --> ENC3["text_transform.py\nBAAI/bge-large-en-v1.5\n1024-dim + 2 meta = 1026-dim"]

    ENC1 --> OOF1["model_train.py　5-fold OOF\nXGBoost  →  n,2\nL2 LogReg  →  n,2\n────────────────\nn,4"]
    ENC2 --> OOF2["model_train.py　5-fold OOF\nXGBoost  →  n,2\nL2 LogReg  →  n,2\n────────────────\nn,4"]
    ENC3 --> OOF3["model_train.py　5-fold OOF\nXGBoost  →  n,2\nL2 LogReg  →  n,2\n────────────────\nn,4"]

    OOF1 --> HS["hstack  →  n,12"]
    OOF2 --> HS
    OOF3 --> HS

    HS --> META["model_stack.py\nLogisticRegression  meta model"]
    META --> THR["Threshold tuning\n掃描 0.05–0.95，最大化 macro F1\n最佳 threshold = 0.59"]
    THR --> OUT([/"正向 / 負向 預測"/])
```

---

## 實驗紀錄

| 實驗 ID | 描述 | Train F1 | Val F1 | Kaggle |
|---------|------|----------|--------|--------|
| exp_003 | MiniLM + XGBoost（正則化） | 0.9937 | 0.7124 | — |
| exp_004 | + CountVectorizer LDA | 0.9900 | 0.7171 | — |
| exp_005 | 消融：移除 LDA | 0.9912 | 0.7374 | 0.71404 |
| exp_006 | + NLI 領域特徵 | 0.9925 | 0.7250 | — |
| exp_007 | 換 mpnet（768-dim） | 0.9988 | 0.8200 | — |
| exp_008 | cardiffnlp 情感模型 | 0.9969 | 0.7875 | — |
| **exp_009** | **Stacking（mpnet + cardiffnlp）** | **0.8344** | **0.8375** | **0.76749** |
| exp_010 | cardiffnlp + VADER（對照組） | 0.9969 | 0.7849 | — |
| exp_011 | mpnet + VADER | 0.9969 | 0.8024 | — |
| exp_012 | Stacking + num_num/num_extend → number | 0.8337 | 0.8350 | — |
| exp_013 | Stacking baseline（模組化重構後基準） | 0.8294 | 0.8375 | — |
| exp_014 | Stacking + PCA 32-dim 降維 | 0.8212 | 0.8400 | — |
| exp_015 | Stacking + PCA 32-dim + Optuna XGBoost | 0.8325 | 0.8400 | — |
| **exp_016** | **Stacking 3 models + XGBoost + L2 LogReg per base** | **0.8569** | **0.8625** | — |
| exp_017 | 3 models + XGB+LR + no DR | 0.8569 | 0.8800 | 0.79256 |
| **exp_018** | **3 models + XGB+LR + no DR + threshold tuning（0.59）** | **0.8562** | **0.8850** | — |
| exp_019 | 3 models + NLPaug 擴增（有 data leakage） | 0.9200 | ~~0.9387~~ | — |
| exp_020 | 3 models + NLPaug + leakage 修正（threshold=0.54） | 0.9428 | 0.8675 | — |

> exp_003–exp_008（Train F1 ≈ 0.99）：單模型直接 fit train split，嚴重 overfit，Train/Val 差距超過 0.25。  
> exp_009 起改用 OOF Stacking，Train F1 降至 ~0.83（OOF 預測值），Train/Val 差距收斂到 0.01 以內，泛化大幅改善。  
> exp_010 / exp_011：VADER 無額外幫助，mpnet 的 768-dim embedding 已涵蓋情感信號。  
> exp_012：文字清洗改動在誤差範圍內，保留作為後續實驗的標準前處理。  
> exp_013：程式碼模組化重構（src/ 拆分為 5 個獨立模組），功能與 exp_009 相同，確認重構無迴歸。  
> exp_014：PCA 將 770-dim 壓縮至 32-dim（保留 ~47% 變異），Val F1 略升。  
> exp_015：Optuna 50 trials 調 XGBoost 超參（inner 5-fold CV），Val F1 與 exp_014 持平，確認超參已近上限。  
> exp_016：消融驗證 — 第三個模型（bge_large）貢獻 +0.0325，L2 LogReg 單獨加入反而 -0.0125；兩者合用 +0.0225。主要增益來自 bge_large 的多樣性。  
> exp_017：移除 PCA，3-model [all] + no DR，Val F1 回升至 0.8800（PCA 壓縮破壞三模型互補性）。  
> exp_018：在 val set 掃描 threshold（0.05–0.95）最佳化 macro F1，最佳 threshold = 0.59，Val F1 再進 +0.005。  
> exp_019：NLPaug contextual fill-mask 擴增（bert-base-uncased，aug_p=0.1）將 train 從 2000 增至 4000 筆，但 data leakage（original/nlpaug 配對跨 split）使 Val F1 虛高至 0.9387。  
> exp_020：修正 leakage（val 僅用 original 400 筆，train 含 original+nlpaug 3200 筆），Val F1 降至真實的 0.8675。NLPaug 增強品質不足，反而低於無增強的 exp_018（0.8850）。

### 完整消融：模型組合 × 分類器（ablation_exp016.py）

14 個條件，固定 PCA 32-dim，train/val split 相同。

| 模型組合 | clf | meta 維度 | Train F1 | Val F1 |
|----------|-----|-----------|----------|--------|
| mpnet | XGB | (n,2) | 0.8050 | 0.8223 |
| mpnet | XGB+LR | (n,4) | 0.8156 | 0.8200 |
| cardiffnlp | XGB | (n,2) | 0.7931 | 0.7800 |
| cardiffnlp | XGB+LR | (n,4) | 0.7956 | 0.7924 |
| bge_large | XGB | (n,2) | 0.8394 | 0.8600 |
| bge_large | XGB+LR | (n,4) | 0.8431 | 0.8700 |
| mpnet + cardiffnlp | XGB | (n,4) | 0.8325 | 0.8400 |
| mpnet + cardiffnlp | XGB+LR | (n,8) | 0.8319 | 0.8275 |
| **mpnet + bge_large** | **XGB+LR** | **(n,8)** | **0.8506** | **0.8800** |
| mpnet + bge_large | XGB | (n,4) | 0.8500 | 0.8775 |
| cardiffnlp + bge_large | XGB | (n,4) | 0.8444 | 0.8599 |
| cardiffnlp + bge_large | XGB+LR | (n,8) | 0.8456 | 0.8675 |
| mpnet + cardiffnlp + bge_large | XGB | (n,6) | 0.8550 | 0.8725 |
| mpnet + cardiffnlp + bge_large | XGB+LR | (n,12) | 0.8569 | 0.8625 |

> **bge_large 是最強單一模型**（0.8600），大幅領先 mpnet（0.8223）和 cardiffnlp（0.7800）。  
> **最佳組合：mpnet + bge_large + XGB+LR（0.8800）**，三模型加入 cardiffnlp 反而下降（領域偏移雜訊）。  
> cardiffnlp 在 Twitter 情感資料訓練，遇到電影/商品/遊戲評論出現領域偏移，embedding 帶入雜訊。

### PCA 維度消融：XGB+LR × 模型數量（ablation_pca_2model.py / ablation_pca_3model.py）

固定 XGB+LR，比較 mpnet+bge_large（2-model）與 mpnet+cardiffnlp+bge_large（3-model）在不同 DR 設定下的表現。

**2-model（mpnet + bge_large）+ XGB+LR**

| 降維設定 | meta 維度 | var% | Train F1 | Val F1 |
|----------|-----------|------|----------|--------|
| no DR | (n,8) | 100% | 0.8537 | 0.8750 |
| PCA 128-dim | (n,8) | 78.6% | 0.8494 | 0.8775 |
| PCA 64-dim | (n,8) | 62.9% | 0.8512 | 0.8725 |
| **PCA 32-dim** | **(n,8)** | **48.2%** | **0.8475** | **0.8875** |
| PCA 16-dim | (n,8) | 35.9% | 0.8525 | 0.8750 |

**3-model（mpnet + cardiffnlp + bge_large）+ XGB+LR**

| 降維設定 | meta 維度 | var% | Train F1 | Val F1 |
|----------|-----------|------|----------|--------|
| **no DR** | **(n,12)** | **100%** | **0.8569** | **0.8800** |
| PCA 128-dim | (n,12) | 83.6% | 0.8512 | 0.8775 |
| PCA 64-dim | (n,12) | 70.8% | 0.8531 | 0.8775 |
| PCA 32-dim | (n,12) | 58.2% | 0.8525 | 0.8675 |
| PCA 16-dim | (n,12) | 47.4% | 0.8506 | 0.8725 |

> 2-model：PCA 32 是甜蜜點（0.8875），比 no DR 多 +0.0125；過度壓縮（16-dim）和不壓縮都會下降。  
> 3-model：PCA 反而有損，no DR 最佳（0.8800）；原因是 PCA 把三個模型互補的 embedding 混合壓縮，破壞了多樣性。  
> 結論：2-model 受益於 PCA 的正則化（資料量少），3-model 資訊量更豐富，PCA 壓縮的損失大於正則化增益。  
> **exp_018 + threshold tuning 以 0.8850 超越本區段所有條件，為目前最高分。**

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

入口腳本 `train_stacking.py` 依序呼叫以下五個模組：

| 模組 | 功能 |
|------|------|
| [src/data_preprocess.py](src/data_preprocess.py) | 文字清理。Track A 只保留英文字母（供 TF-IDF/LDA），Track B 還原括號與標點（供 BERT）|
| [src/text_transform.py](src/text_transform.py) | 以 SentenceTransformer 將文字編碼為 embedding 向量，並拼接 meta 特徵（`?` / `!` 計數）|
| [src/dimension_decrease.py](src/dimension_decrease.py) | PCA / PLS 降維（可設 `method: none` 略過）|
| [src/model_train.py](src/model_train.py) | 5-fold OOF 訓練 XGBoost + L2 LogReg，輸出各 fold 的正 / 負機率作為 meta features |
| [src/model_stack.py](src/model_stack.py) | 訓練 LogisticRegression meta model，掃描 threshold（0.05–0.95）最佳化 macro F1，儲存最佳 threshold |

推論流程由 `predict_stacking.py` 執行，載入各 base model 的 `_full` 版本及 `threshold.json`，輸出最終 submission。
