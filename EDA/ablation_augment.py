"""
EDA 增強消融實驗

比較四種訓練集組合對 Val F1 的影響：
  A) original only          (2000 → 1600 train)
  B) original + synonym     (+ 2000 → 3600 train)
  C) original + swap        (+ 2000 → 3600 train)
  D) original + delete      (+ 2000 → 3600 train)
  E) original + all three   (+ 6000 → 7600 train)

Val set 固定為原始資料的 20%（400 筆），不混入增強資料。
"""

import sys
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '.')
from src.preprocess import clean_for_bert
from src.utils import set_seed


def load_cfg(path='config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    v = cfg['global']['device']
    if v == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def encode(texts: list, model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True)


def build_xgb(cfg: dict) -> XGBClassifier:
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators=c['n_estimators'], learning_rate=c['learning_rate'],
        max_depth=c['max_depth'], random_state=c['random_state'],
        eval_metric=c['eval_metric'], device=c['device'],
        subsample=c['subsample'], colsample_bytree=c['colsample_bytree'],
        reg_alpha=c['reg_alpha'], reg_lambda=c['reg_lambda'],
    )


def run_condition(label: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                  model: SentenceTransformer, cfg: dict) -> float:
    train_texts = [clean_for_bert(t, cfg) for t in train_df['TEXT']]
    val_texts   = [clean_for_bert(t, cfg) for t in val_df['TEXT']]

    X_train = encode(train_texts, model)
    X_val   = encode(val_texts,   model)

    xgb = build_xgb(cfg)
    xgb.fit(X_train, train_df['LABEL'].values)

    f1 = f1_score(val_df['LABEL'].values, xgb.predict(X_val), average='macro')
    print(f'  {label:<40} train={len(train_df):>5}  Val F1={f1:.4f}')
    return f1


def main():
    cfg    = load_cfg()
    seed   = cfg['global']['random_seed']
    device = resolve_device(cfg)
    set_seed(cfg)

    print(f'Device: {device}')
    print('Loading model: all-mpnet-base-v2 ...')
    model = SentenceTransformer('all-mpnet-base-v2', device=device)

    # ── 原始資料 split ──────────────────────────────────────────────────────────
    orig = pd.read_csv('datasets/train_2022.csv')
    train_orig, val_df = train_test_split(
        orig, test_size=0.2, stratify=orig['LABEL'], random_state=seed
    )

    # ── 增強資料（只取非 original 的列）──────────────────────────────────────────
    aug = pd.read_csv('datasets/train_2022_augmented.csv')
    # 只保留 source != original（增強列），且對應的 row 在 train_orig 裡（避免 val 資料的增強版混入）
    train_ids = set(train_orig['row_id'].astype(str).tolist())
    aug_only  = aug[aug['source'] != 'original'].copy()
    # aug ID 格式為 "{orig_row_id}_eda_xxx"，取前綴比對
    aug_only['orig_id'] = aug_only['ID'].astype(str).str.split('_eda').str[0]
    aug_only = aug_only[aug_only['orig_id'].isin(train_ids)]

    syn_df  = aug_only[aug_only['source'] == 'eda_synonym']
    swap_df = aug_only[aug_only['source'] == 'eda_swap']
    del_df  = aug_only[aug_only['source'] == 'eda_delete']

    def merge(*dfs):
        return pd.concat([train_orig, *dfs], ignore_index=True)

    conditions = [
        ('A) original only',         train_orig),
        ('B) original + synonym',     merge(syn_df)),
        ('C) original + swap',        merge(swap_df)),
        ('D) original + delete',      merge(del_df)),
        ('E) original + all three',   merge(syn_df, swap_df, del_df)),
    ]

    print(f'\nVal set size: {len(val_df)} (original only, fixed)\n')
    print('-' * 65)
    results = {}
    for label, train_df in conditions:
        results[label] = run_condition(label, train_df, val_df, model, cfg)
    print('-' * 65)

    best = max(results, key=results.get)
    print(f'\nBest: {best}  →  Val F1 = {results[best]:.4f}')


if __name__ == '__main__':
    main()
