# 實驗操作手冊

本文件說明如何從零開始建立環境、進行模型訓練，以及產出 Kaggle submission。

---

## 目錄

1. [環境建立](#環境建立)
2. [設定實驗參數](#設定實驗參數)
3. [train.py — 訓練流程](#trainpy--訓練流程)
4. [predict.py — 推論流程](#predictpy--推論流程)
5. [常見錯誤排除](#常見錯誤排除)

---

## 環境建立

### 建立 Conda 虛擬環境

```bash
conda create -n textpolarity python=3.10 -y
conda activate textpolarity
```

### 安裝套件

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 PyTorch CUDA 12.6 的安裝來源。若 GPU 驅動或 CUDA 版本不同，請先確認相容版本再安裝：

```bash
# 確認 CUDA 版本
nvidia-smi

# 若需要 CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 確認 GPU 可用

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX XXXX
```

### 首次下載模型快取

`train.py` 執行時會自動從 HuggingFace 下載以下模型（僅第一次需要，之後從快取讀取）：

| 模型 | 用途 | 大小 |
|------|------|------|
| `all-MiniLM-L6-v2` | BERT 語意嵌入 | ~90 MB |
| `cross-encoder/nli-deberta-v3-small` | NLI 領域分類（啟用時） | ~180 MB |

若下載速度慢，可設定 HuggingFace 鏡像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 設定實驗參數

所有參數集中在 [config/config.yaml](../config/config.yaml)，每次新實驗前請先調整以下欄位：

### 必填：實驗 ID

```yaml
experiment:
  id:   exp_007_my_experiment   # 唯一識別碼，不可與現有實驗重複
  name: 描述這次實驗的重點       # 自由描述，寫入 results.csv
```

> `exp_id` 決定模型、指標、config 快照的儲存目錄。若重複使用相同 ID，`results.csv` 會更新該行，模型會被覆蓋。

### 特徵開關

```yaml
features:
  lda:
    enabled: false    # true = 加入 LDA 主題特徵（3 個主題機率值）
  nli:
    enabled: true     # true = 加入 Zero-shot NLI 領域特徵（3 個領域分數）
```

**消融實驗建議順序：**

| 實驗目的 | lda.enabled | nli.enabled |
|----------|-------------|-------------|
| 僅 BERT embeddings | false | false |
| BERT + NLI | false | true |
| BERT + LDA | true | false |
| 全部特徵 | true | true |

### 模型超參數

```yaml
models:
  xgboost:
    n_estimators:    300     # 樹的數量，越大越慢但表達力越強
    learning_rate:   0.05    # 學習率，調小需搭配增加 n_estimators
    max_depth:       3       # 單棵樹深度，越深越容易過擬合
    subsample:       0.8     # 每棵樹隨機取樣比例（防過擬合）
    colsample_bytree: 0.8    # 每棵樹隨機取樣特徵比例
    reg_alpha:       0.1     # L1 正則化
    reg_lambda:      1.0     # L2 正則化
```

### 運算裝置

```yaml
global:
  device: auto   # auto = 有 CUDA 自動用 GPU，否則用 CPU
                 # 也可直接填 cpu 或 cuda
```

---

## train.py — 訓練流程

### 基本用法

```bash
python train.py
```

### 執行步驟說明

```
1. 讀取 config.yaml
2. 載入 datasets/train_2022.csv（2,000 筆）
3. 按 80/20 分層切分 → Train（1,600）/ Val（400）
        ↓
4. 前處理（Train 與 Val 分開處理，互不影響）
   ├─ Track A：清理英文字母供 CountVectorizer 使用
   └─ Track B：還原括號、修復縮寫供 BERT 使用

5. 特徵提取（Train split）
   ├─ BERT 嵌入：all-MiniLM-L6-v2 → (1600, 384)
   ├─ Meta 特徵：? 與 ! 計數 → (1600, 2)
   ├─ LDA 主題（若啟用）：CountVec.fit_transform + LDA.fit_transform → (1600, 3)
   └─ NLI 領域（若啟用）：zero-shot 推論 → (1600, 3)

6. 特徵提取（Val split）
   └─ 使用步驟 5 fit 好的 CountVec / LDA 進行 transform only（防止特徵洩漏）

7. 訓練 XGBoost（Train split）

8. 評估 Train / Val 兩個 split 的 Accuracy、Precision、Recall、F1

9. 儲存實驗記錄
   ├─ experiments/{exp_id}/metrics.json
   ├─ experiments/{exp_id}/config_snapshot.yaml
   └─ experiments/results.csv（append / 更新）

10. 全量資料重新 fit 特徵、重新訓練 XGBoost（最終提交用模型）

11. 儲存模型 artifacts
    ├─ models/{exp_id}/xgb.json
    ├─ models/{exp_id}/count_vec.pkl（僅 LDA 啟用時）
    └─ models/{exp_id}/lda.pkl（僅 LDA 啟用時）
```

### 輸出範例

```
Device: cuda | Val ratio: 0.2
Features — LDA: False | NLI: True
Full train set: (2000, 3)
Train split: (1600, 3)  |  Val split: (400, 3)

>> Preprocessing...
>> Building train features (fit)...
Batches: 100%|███████████████| 50/50
>> NLI inference on 1600 texts (device=cuda)...
>> Building val features (transform only)...
Batches: 100%|███████████████| 13/13
>> NLI inference on 400 texts (device=cuda)...
Feature matrix — Train: (1600, 389)  |  Val: (400, 389)

>> Training on train split...

[ Train split ]
  Accuracy     0.9925
  Precision    0.9925
  Recall       0.9925
  F1           0.9925

[ Val split   ]
  Accuracy     0.7250
  Precision    0.7250
  Recall       0.7250
  F1           0.7250

✅ metrics saved: experiments/exp_006_NLI/metrics.json
✅ config snapshot saved: experiments/exp_006_NLI/config_snapshot.yaml
✅ results.csv updated: experiments/results.csv

>> Retraining on full dataset for final model...
✅ Artifacts saved: models/exp_006_NLI
```

### 訓練時間參考（RTX 5070 Ti）

| NLI 開啟 | LDA 開啟 | 約耗時 |
|----------|----------|--------|
| ✗ | ✗ | ~2 分鐘 |
| ✓ | ✗ | ~10 分鐘 |
| ✗ | ✓ | ~3 分鐘 |
| ✓ | ✓ | ~11 分鐘 |

> NLI 推論需跑兩次（Train + Full），是主要耗時來源。

---

## predict.py — 推論流程

### 基本用法

```bash
# 使用 config.yaml 中的 experiment.id
python predict.py

# 指定使用特定實驗的模型
python predict.py --exp-id exp_005_ablation_no_lda
```

### 執行步驟說明

```
1. 讀取 config.yaml（--exp-id 會覆蓋 experiment.id）
2. 載入 datasets/test_2022.csv（11,000 筆）
3. 前處理（與訓練時 Track B 相同邏輯）
4. 載入模型 artifacts
   ├─ models/{exp_id}/xgb.json
   ├─ models/{exp_id}/count_vec.pkl（LDA 啟用時）
   └─ models/{exp_id}/lda.pkl（LDA 啟用時）
5. 特徵提取（transform only，不重新 fit）
6. XGBoost 預測
7. 儲存 results/submission_{exp_id}.csv
```

### 特徵開關自動對齊

`--exp-id` 指定時，predict.py 會自動讀取 `experiments/{exp_id}/config_snapshot.yaml`（訓練當下的 config 快照），確保特徵開關與訓練時完全一致。

不指定 `--exp-id` 時，才讀取目前的 `config/config.yaml`。

### 輸出

```
Experiment: exp_006_NLI | Device: cuda | LDA: False | NLI: True
Test: (11000, 2)
>> Preprocessing...
>> Encoding with SentenceTransformer...
>> NLI inference on 11000 texts (device=cuda)...
Feature matrix: (11000, 389)
Predictions: 5821 positive / 5179 negative
✅ Submission saved: results/submission_exp_006_NLI.csv
```

---

## 常見錯誤排除

### `RuntimeError: operator torchvision::nms does not exist`

torchvision 與 torch 版本不相容，直接移除 torchvision 即可（本專案不需要）：

```bash
pip uninstall torchvision -y
```

### `FileNotFoundError: Project root not found`

從非專案目錄執行腳本時發生。請切換到專案根目錄再執行：

```bash
cd /path/to/Kaagle-TextPolarity
python train.py
```

### `XGBoost feature dimension mismatch`

predict.py 的特徵維度與訓練時不同，原因是 config 的特徵開關與訓練時不一致。  
請對照 `experiments/{exp_id}/config_snapshot.yaml` 還原訓練時的開關設定。

### `OutOfMemoryError` (GPU)

NLI 模型較大，batch_size 設太高會 OOM。調小 config 中的批次大小：

```yaml
features:
  nli:
    batch_size: 16   # 預設 32，降至 16 或 8
  sentence_transformer:
    batch_size: 16
```

### HuggingFace 模型下載失敗 / 超慢

設定鏡像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train.py
```
