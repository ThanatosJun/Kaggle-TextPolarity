"""
消融：3 model [all] + XGB+LR — PCA 維度影響

固定 train/val split，encode 一次，測試不同 PCA 設定：
  - no DR
  - PCA 128-dim
  - PCA 64-dim
  - PCA 32-dim
  - PCA 16-dim

各條件均使用 XGB+LR（ablation_exp016 的最佳 classifier 設定）。
"""

import sys
import yaml
import numpy as np
import cupy as cp
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, '.')
from src.data_preprocess import run as preprocess
from src.text_transform import build_features
from src.utils import set_seed


def load_cfg(path='config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    v = cfg['global']['device']
    if v == 'auto':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def build_xgb(cfg):
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators=c['n_estimators'], learning_rate=c['learning_rate'],
        max_depth=c['max_depth'], random_state=c['random_state'],
        eval_metric=c['eval_metric'], device='cuda',
        subsample=c['subsample'], colsample_bytree=c['colsample_bytree'],
        reg_alpha=c['reg_alpha'], reg_lambda=c['reg_lambda'],
    )


def build_lr(cfg):
    c = cfg.get('models', {}).get('logreg_base', {})
    return LogisticRegression(
        C=c.get('C', 1.0),
        max_iter=c.get('max_iter', 1000),
        random_state=cfg['global']['random_seed'],
    )


def apply_pca(X_train_raw, X_val_raw, n_components, seed):
    if n_components is None:
        return X_train_raw, X_val_raw
    reducer = PCA(n_components=n_components, random_state=seed)
    X_tr  = reducer.fit_transform(X_train_raw)
    X_val = reducer.transform(X_val_raw)
    var_explained = reducer.explained_variance_ratio_.sum()
    return X_tr, X_val, var_explained


def run_condition(raw_train_list, raw_val_list, y_train, y_val, n_components, cfg):
    seed    = cfg['global']['random_seed']
    n_folds = cfg['stacking']['cv_folds']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_parts, val_parts = [], []
    total_var = []

    for X_tr_raw, X_vl_raw in zip(raw_train_list, raw_val_list):
        if n_components is not None:
            result = apply_pca(X_tr_raw, X_vl_raw, n_components, seed)
            X_tr, X_vl, var = result
            total_var.append(var)
        else:
            X_tr, X_vl = X_tr_raw, X_vl_raw

        X_tr_gpu = cp.asarray(X_tr)
        X_vl_gpu = cp.asarray(X_vl)

        oof_xgb = np.zeros((len(y_train), 2))
        oof_lr  = np.zeros((len(y_train), 2))

        for tr_idx, val_idx in kf.split(X_tr, y_train):
            xgb = build_xgb(cfg)
            xgb.fit(X_tr_gpu[tr_idx], y_train[tr_idx])
            oof_xgb[val_idx] = np.asarray(xgb.predict_proba(X_tr_gpu[val_idx]))

            lr = build_lr(cfg)
            lr.fit(X_tr[tr_idx], y_train[tr_idx])
            oof_lr[val_idx] = lr.predict_proba(X_tr[val_idx])

        final_xgb = build_xgb(cfg)
        final_xgb.fit(X_tr_gpu, y_train)
        final_lr = build_lr(cfg)
        final_lr.fit(X_tr, y_train)

        oof_parts.append(np.hstack([oof_xgb, oof_lr]))
        val_parts.append(np.hstack([
            np.asarray(final_xgb.predict_proba(X_vl_gpu)),
            final_lr.predict_proba(X_vl),
        ]))

    meta_train = np.hstack(oof_parts)
    meta_val   = np.hstack(val_parts)

    mc   = cfg['stacking']['meta_model']
    meta = LogisticRegression(
        C=mc.get('C', 1.0), max_iter=mc.get('max_iter', 1000),
        random_state=seed,
    )
    meta.fit(meta_train, y_train)

    train_f1 = f1_score(y_train, meta.predict(meta_train), average='macro')
    val_f1   = f1_score(y_val,   meta.predict(meta_val),   average='macro')
    avg_var  = np.mean(total_var) if total_var else None
    return train_f1, val_f1, meta_train.shape[1], avg_var


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

    # encode 全部 3 個 base model（只跑一次）
    bm_cfgs = cfg['stacking']['base_models']
    raw_train_list, raw_val_list = [], []
    for bm in bm_cfgs:
        print(f'>> Encoding {bm["id"]} ({bm["model_name"]})...')
        X_tr = build_features(
            proc_train['bert_texts'], proc_train['meta'],
            bm['model_name'], batch_size, device,
        )
        X_vl = build_features(
            proc_val['bert_texts'], proc_val['meta'],
            bm['model_name'], batch_size, device,
        )
        raw_train_list.append(X_tr)
        raw_val_list.append(X_vl)
        print(f'   {X_tr.shape}')

    # 測試條件
    conditions = [
        (None,  'no DR'),
        (128,   'PCA 128-dim'),
        (64,    'PCA  64-dim'),
        (32,    'PCA  32-dim'),
        (16,    'PCA  16-dim'),
    ]

    print(f'\n{"="*75}')
    print(f'{"條件":<16} {"meta dim":>10} {"var%":>8} {"Train F1":>10} {"Val F1":>10}')
    print(f'{"-"*75}')

    for n_components, label in conditions:
        tr_f1, vl_f1, meta_dim, avg_var = run_condition(
            raw_train_list, raw_val_list, y_train, y_val, n_components, cfg,
        )
        var_str = f'{avg_var*100:.1f}%' if avg_var is not None else '100%'
        print(f'{label:<16} {f"(n,{meta_dim})":>10} {var_str:>8} {tr_f1:>10.4f} {vl_f1:>10.4f}')

    print(f'{"="*75}')


if __name__ == '__main__':
    main()
