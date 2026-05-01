"""
PCA 降維消融實驗（Stacking 版）

固定 train/val split，對每個 base model 的 (n, 770) 特徵測試不同 PCA 維度，
觀察 Stacking Val F1 的變化。

PCA fit 在 train split，transform 到 val（無洩漏）。
"""

import sys
import yaml
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

sys.path.insert(0, '.')
from src.preprocess import run as preprocess
from src.utils import set_seed


def load_cfg(path='config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    v = cfg['global']['device']
    if v == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def encode_features(texts, meta, model_name, batch_size, device) -> np.ndarray:
    st = SentenceTransformer(model_name, device=device)
    X_emb = st.encode(texts, batch_size=batch_size, show_progress_bar=True)
    del st
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.hstack([X_emb, meta])


def build_xgb(cfg) -> XGBClassifier:
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators=c['n_estimators'], learning_rate=c['learning_rate'],
        max_depth=c['max_depth'], random_state=c['random_state'],
        eval_metric=c['eval_metric'], device=c['device'],
        subsample=c['subsample'], colsample_bytree=c['colsample_bytree'],
        reg_alpha=c['reg_alpha'], reg_lambda=c['reg_lambda'],
    )


def get_oof_and_val(X_train, y_train, X_val, cfg):
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros((len(y_train), 2))
    for tr_idx, val_idx in kf.split(X_train, y_train):
        m = build_xgb(cfg)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        oof[val_idx] = m.predict_proba(X_train[val_idx])

    final = build_xgb(cfg)
    final.fit(X_train, y_train)
    val_proba = final.predict_proba(X_val)
    return oof, val_proba


def run_stacking(raw_train_list, raw_val_list, y_train, y_val, method, n_dim, cfg):
    """
    method: 'none' | 'pca' | 'pls'
    n_dim:  None (no reduction) or int
    """
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression

    oof_list = []
    val_proba_list = []

    for X_tr_raw, X_val_raw in zip(raw_train_list, raw_val_list):
        if method == 'pca' and n_dim:
            reducer = PCA(n_components=n_dim, random_state=cfg['global']['random_seed'])
            X_tr   = reducer.fit_transform(X_tr_raw)
            X_val_ = reducer.transform(X_val_raw)
        elif method == 'pls' and n_dim:
            reducer = PLSRegression(n_components=n_dim)
            reducer.fit(X_tr_raw, y_train)
            X_tr   = reducer.transform(X_tr_raw)
            X_val_ = reducer.transform(X_val_raw)
        else:
            X_tr   = X_tr_raw
            X_val_ = X_val_raw

        oof, val_proba = get_oof_and_val(X_tr, y_train, X_val_, cfg)
        oof_list.append(oof)
        val_proba_list.append(val_proba)

    meta_train = np.hstack(oof_list)
    meta_val   = np.hstack(val_proba_list)

    mc   = cfg['stacking']['meta_model']
    meta = LogisticRegression(C=mc.get('C', 1.0), max_iter=mc.get('max_iter', 1000),
                              random_state=cfg['global']['random_seed'])
    meta.fit(meta_train, y_train)
    f1 = f1_score(y_val, meta.predict(meta_val), average='macro')
    return f1


def main():
    cfg    = load_cfg()
    seed   = cfg['global']['random_seed']
    device = resolve_device(cfg)
    set_seed(cfg)
    batch_size = cfg['features']['sentence_transformer']['batch_size']

    full_df = pd.read_csv(cfg['paths']['train'])
    train_df, val_df = train_test_split(
        full_df, test_size=cfg['evaluation']['val_ratio'],
        stratify=full_df['LABEL'], random_state=seed,
    )
    y_train = train_df['LABEL'].values
    y_val   = val_df['LABEL'].values

    print('>> Preprocessing...')
    proc_train = preprocess(train_df, cfg)
    proc_val   = preprocess(val_df,   cfg)

    base_cfgs = cfg['stacking']['base_models']
    raw_train_list, raw_val_list = [], []

    for bm in base_cfgs:
        print(f'\n>> Encoding {bm["id"]} ({bm["model_name"]})...')
        X_tr  = encode_features(proc_train['bert_texts'], proc_train['meta'],
                                bm['model_name'], batch_size, device)
        X_val = encode_features(proc_val['bert_texts'],   proc_val['meta'],
                                bm['model_name'], batch_size, device)
        print(f'   Shape: {X_tr.shape}')
        raw_train_list.append(X_tr)
        raw_val_list.append(X_val)

    # ── 消融：PCA vs PLS vs 無降維 ───────────────────────────────────────
    conditions = [
        ('none',  None,  'no DR (770-dim)'),
        ('pca',   256,   'PCA 256-dim'),
        ('pca',   128,   'PCA 128-dim'),
        ('pca',    32,   'PCA  32-dim'),
        ('pls',    32,   'PLS  32-dim'),
        ('pls',    16,   'PLS  16-dim'),
        ('pls',     8,   'PLS   8-dim'),
    ]

    orig_dim = raw_train_list[0].shape[1]
    print(f'\n{"="*50}')
    print(f'{"方法":<20} {"Val F1":>10}')
    print(f'{"-"*50}')

    results = {}
    for method, n_dim, label in conditions:
        f1 = run_stacking(raw_train_list, raw_val_list, y_train, y_val, method, n_dim, cfg)
        results[label] = f1
        print(f'{label:<20} {f1:>10.4f}')

    print(f'{"="*50}')
    best = max(results, key=results.get)
    print(f'\nBest: {best}  →  Val F1 = {results[best]:.4f}')


if __name__ == '__main__':
    main()
