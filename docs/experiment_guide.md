# 實驗操作手冊

本文件供新實驗者接手使用，說明環境建立、兩種訓練模式的完整操作，以及常見問題排除。

---

## 目錄

1. [環境建立](#環境建立)
2. [Config 設定說明](#config-設定說明)
3. [模式一：單模型（train.py）](#模式一單模型trainpy)
4. [模式二：Stacking（train_stacking.py）](#模式二stackingtrain_stackingpy)
5. [推論與產出 Submission](#推論與產出-submission)
6. [新增實驗 SOP](#新增實驗-sop)
7. [常見錯誤排除](#常見錯誤排除)

---

## 環境建立

### 建立 Conda 虛擬環境

```bash
conda create -n textpolarity python=3.10 -y
conda activate textpolarity
pip install -r requirements.txt
```

`requirements.txt` 包含 PyTorch CUDA 12.6 的安裝來源。若 GPU 驅動版本不同，請先確認：

```bash
nvidia-smi   # 確認 CUDA 版本

# 若需要 CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 確認 GPU 可用

```python
import torch
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX XXXX
```

### 首次下載模型快取

訓練時會自動從 HuggingFace 下載（第一次需要網路，之後從快取讀取）：

| 模型 | 用途 | 大小 |
|------|------|------|
| `all-mpnet-base-v2` | Stacking base model（主力） | ~420 MB |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Stacking base model（情感導向） | ~480 MB |
| `all-MiniLM-L6-v2` | 舊版單模型實驗（備用） | ~90 MB |
| `cross-encoder/nli-deberta-v3-small` | NLI 特徵（實驗性） | ~180 MB |

下載慢時設定鏡像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Config 設定說明

所有參數集中於 [config/config.yaml](../config/config.yaml)。每次新實驗前只需調整以下區塊：

### 必填：實驗 ID

```yaml
experiment:
  id:   exp_010_my_experiment   # 唯一識別碼，不可重複
  name: 描述這次實驗重點         # 寫入 results.csv 的說明
```

> 若使用已存在的 `exp_id`，`results.csv` 會更新該行，模型會被覆蓋。

### Stacking base model 設定

```yaml
stacking:
  cv_folds: 5                  # OOF 的 K-fold 數量
  base_models:
    - id:         mpnet
      model_name: all-mpnet-base-v2
    - id:         cardiffnlp
      model_name: cardiffnlp/twitter-roberta-base-sentiment-latest
  meta_model:
    type:     logistic_regression
    C:        1.0
    max_iter: 1000
```

要新增 base model 只需在 `base_models` 加一筆。

### 單模型特徵開關（train.py 用）

```yaml
features:
  sentence_transformer:
    model_name: all-mpnet-base-v2   # 改這裡切換模型
  lda:
    enabled: false   # LDA 主題特徵（消融實驗用，通常關閉）
  nli:
    enabled: false   # Zero-shot NLI 領域特徵（消融實驗用）
```

### 運算裝置

```yaml
global:
  device: auto   # auto = 有 CUDA 用 GPU，否則 CPU
```

---

## 模式一：單模型（train.py）

**用途：快速驗證新想法（換模型、加特徵），不是正式提交用。**

### 執行

```bash
python train.py
```

### 流程

```
1. 讀取 config.yaml + train_2022.csv（2,000 筆）
2. 80/20 分層切分 → Train（1,600）/ Val（400）
3. 雙軌前處理
   ├─ Track A：純英文字母，供 CountVec/LDA 使用
   └─ Track B：還原括號、修復縮寫，供 BERT 使用
4. Train split 特徵提取（fit）
   ├─ BERT 嵌入（768-dim 或 384-dim，依 model_name）
   ├─ Meta：? 與 ! 計數
   ├─ LDA（若啟用）
   └─ NLI（若啟用）
5. Val split 特徵提取（transform only，防止 leakage）
6. XGBoost 訓練 + 評估 Train / Val
7. 儲存實驗記錄
   ├─ experiments/{exp_id}/metrics.json
   ├─ experiments/{exp_id}/config_snapshot.yaml
   └─ experiments/results.csv
8. 全量資料重新訓練（最終提交用模型）
9. 儲存 models/{exp_id}/xgb.json
```

### 輸出範例

```
Device: cuda | Val ratio: 0.2
Features — LDA: False | NLI: False
Train: (1600, 3) | Val: (400, 3)

[ Train split ]  Accuracy: 0.9988  F1: 0.9988
[ Val split   ]  Accuracy: 0.8200  F1: 0.8200

✅ Artifacts saved: models/exp_007_mpnet
```

### 耗時參考（RTX 5070 Ti）

| 模型 | NLI | 約耗時 |
|------|-----|--------|
| MiniLM（384-dim） | ✗ | ~2 分鐘 |
| mpnet（768-dim） | ✗ | ~3 分鐘 |
| 任何模型 | ✓ | +8 分鐘 |

---

## 模式二：Stacking（train_stacking.py）

**用途：正式提交，多個 base model 互補，目前最高分（Val F1 0.8375）。**

### 執行

```bash
python train_stacking.py
```

### 流程

```
1. 讀取 config.yaml + train_2022.csv
2. 80/20 分層切分 → Train（1,600）/ Val（400）
3. 對每個 base model（現為 mpnet + cardiffnlp）：
   a. 編碼 Train + Val 文字為 embedding
   b. 5-fold OOF：Train split 每筆資料由沒見過它的模型預測
      → OOF probabilities (1600, 2)，無 leakage
   c. 用 Train split 全量訓練最終 base model
   d. 用最終 base model 預測 Val → (400, 2)
4. 疊加各 base model 的 OOF → meta 訓練特徵 (1600, 4)
   疊加各 base model 的 Val 預測 → meta 驗證特徵 (400, 4)
5. LogisticRegression meta model 在 (1600, 4) 上訓練
6. meta model 預測 Val → 評估最終 Accuracy / F1
7. 儲存實驗記錄（同 train.py）
8. 全量資料（2,000 筆）重跑一次步驟 3-5
9. 儲存 models/{exp_id}/
   ├─ xgb_{model_id}_full.json  （各 base model 全量版）
   └─ meta_model_full.pkl        （meta model 全量版）
```

### OOF 說明

OOF（Out-of-Fold）是 Stacking 防止 leakage 的機制：

```
Train 1,600筆分成5份，每份320筆：
Fold 1: 在 2+3+4+5 訓練 → 預測 fold 1（模型沒見過 fold 1）
Fold 2: 在 1+3+4+5 訓練 → 預測 fold 2
...
合併 → 1,600筆預測，每筆都由沒見過它的模型產生 = 誠實的 meta 訓練資料
```

### 輸出範例

```
Base models: ['mpnet', 'cardiffnlp']
CV folds: 5 | Meta: logistic_regression

── Base model: mpnet ──
   OOF (5-fold CV)...
     fold 1/5 ... fold 5/5
   Saved: xgb_mpnet.json

── Base model: cardiffnlp ──
   OOF (5-fold CV)...
     fold 1/5 ... fold 5/5
   Saved: xgb_cardiffnlp.json

── Meta model ──
Meta feature shape — Train: (1600, 4) | Val: (400, 4)

[ Train (OOF) ]  F1: 0.8344
[ Val         ]  F1: 0.8375

✅ All artifacts saved: models/exp_009_stacking
```

### 耗時參考（RTX 5070 Ti）

| base model 數量 | 約耗時 |
|----------------|--------|
| 2（mpnet + cardiffnlp） | ~15 分鐘 |
| 每增加一個 base model | +7 分鐘 |

---

## 推論與產出 Submission

### 單模型

```bash
# 使用目前 config.yaml 的 experiment.id
python predict.py

# 指定特定實驗（自動讀取該實驗的 config snapshot）
python predict.py --exp-id exp_007_mpnet
```

### Stacking

```bash
python predict_stacking.py

python predict_stacking.py --exp-id exp_009_stacking
```

兩個 predict 腳本都會：
1. 若指定 `--exp-id`，自動讀取 `experiments/{exp_id}/config_snapshot.yaml`（確保特徵與訓練時一致）
2. 輸出至 `results/submission_{exp_id}.csv`

---

## 新增實驗 SOP

### 快速驗證新想法

```
1. 修改 config.yaml
   - 設定新 experiment.id（格式：exp_NNN_描述）
   - 調整 sentence_transformer.model_name 或特徵開關
2. python train.py
3. 看 Val F1 是否比 0.8200（exp_007）高
4. 若有效 → 整合進 train_stacking.py 的 base_models
```

### 新增 Stacking base model

```
1. 修改 config.yaml
   - 設定新 experiment.id
   - 在 stacking.base_models 新增一筆
2. python train_stacking.py
3. 看 Val F1 是否比 0.8375（exp_009）高
```

### 記錄 Kaggle 分數

上傳 submission 後，手動更新 `experiments/results.csv` 的 `kaggle_score` 欄位：

```
experiment_id, ..., kaggle_score, note
exp_009_stacking, ..., 0.XXXX, 首次 stacking 提交
```

---

## 常見錯誤排除

### `RuntimeError: operator torchvision::nms does not exist`

torchvision 與 torch 版本不相容，移除即可（本專案不需要）：

```bash
pip uninstall torchvision -y
```

### `FileNotFoundError: Project root not found`

從非專案根目錄執行時發生，切換目錄即可：

```bash
cd /path/to/Kaagle-TextPolarity
python train.py
```

### `KeyError: 'nli'` 或 `ValueError: Feature shape mismatch`

用 `--exp-id` 指定舊實驗（NLI 功能加入前訓練的）時可能出現。

已修正：predict.py 與 predict_stacking.py 都會自動讀取該實驗的 `config_snapshot.yaml`，特徵開關自動對齊，不需手動調整 config。

若仍出現，確認 `experiments/{exp_id}/config_snapshot.yaml` 是否存在。

### `OutOfMemoryError` (GPU)

調小批次大小：

```yaml
features:
  sentence_transformer:
    batch_size: 16   # 預設 32
  nli:
    batch_size: 16
```

### HuggingFace 模型下載慢

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train_stacking.py
```
