"""
完整消融實驗：模型數量 × 分類器組合

3 個 LLM base model：
  0: mpnet      (all-mpnet-base-v2)
  1: cardiffnlp (cardiffnlp/twitter-roberta-base-sentiment-latest)
  2: bge_large  (BAAI/bge-large-en-v1.5)

測試所有子集 × 2 種分類器：
  1 model  × C(3,1)=3 組合  × 2 = 6
  2 models × C(3,2)=3 組合  × 2 = 6
  3 models × C(3,3)=1 組合  × 2 = 2
  共 14 個條件

固定 train/val split，encode 一次，所有條件共用。
"""

import itertools
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, '.')
from src.data_preprocess import run as preprocess
from src.text_transform import build_features
from src.dimension_decrease import fit_transform, transform
from src.utils import set_seed


def load_cfg(path='config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    v = cfg['global']['device']
    if v == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def build_xgb(cfg):
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators=c['n_estimators'], learning_rate=c['learning_rate'],
        max_depth=c['max_depth'], random_state=c['random_state'],
        eval_metric=c['eval_metric'], device=c['device'],
        subsample=c['subsample'], colsample_bytree=c['colsample_bytree'],
        reg_alpha=c['reg_alpha'], reg_lambda=c['reg_lambda'],
    )


def build_lr(cfg):
    c = cfg.get('models', {}).get('logreg_base', {})
    return LogisticRegression(C=c.get('C', 1.0),
                              max_iter=c.get('max_iter', 1000),
                              random_state=cfg['global']['random_seed'])


def run_condition(X_list, y_train, X_val_list, y_val, use_lr, cfg):
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_parts, val_parts = [], []

    for X_tr, X_vl in zip(X_list, X_val_list):
        oof_xgb = np.zeros((len(y_train), 2))
        oof_lr  = np.zeros((len(y_train), 2)) if use_lr else None

        for tr_idx, val_idx in kf.split(X_tr, y_train):
            xgb = build_xgb(cfg)
            xgb.fit(X_tr[tr_idx], y_train[tr_idx])
            oof_xgb[val_idx] = xgb.predict_proba(X_tr[val_idx])
            if use_lr:
                lr = build_lr(cfg)
                lr.fit(X_tr[tr_idx], y_train[tr_idx])
                oof_lr[val_idx] = lr.predict_proba(X_tr[val_idx])

        final_xgb = build_xgb(cfg)
        final_xgb.fit(X_tr, y_train)
        vp = final_xgb.predict_proba(X_vl)

        if use_lr:
            final_lr = build_lr(cfg)
            final_lr.fit(X_tr, y_train)
            oof_parts.append(np.hstack([oof_xgb, oof_lr]))
            val_parts.append(np.hstack([vp, final_lr.predict_proba(X_vl)]))
        else:
            oof_parts.append(oof_xgb)
            val_parts.append(vp)

    meta_train = np.hstack(oof_parts)
    meta_val   = np.hstack(val_parts)

    mc   = cfg['stacking']['meta_model']
    meta = LogisticRegression(C=mc.get('C', 1.0), max_iter=mc.get('max_iter', 1000),
                              random_state=cfg['global']['random_seed'])
    meta.fit(meta_train, y_train)

    train_f1 = f1_score(y_train, meta.predict(meta_train), average='macro')
    val_f1   = f1_score(y_val,   meta.predict(meta_val),   average='macro')
    return train_f1, val_f1, meta_train.shape[1]


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

    # ── encode 全部 3 個 base model（只跑一次）──────────────────────────────
    bm_cfgs = cfg['stacking']['base_models']
    all_X_train, all_X_val, bm_names = [], [], []
    for bm in bm_cfgs:
        print(f'>> Encoding {bm["id"]} ({bm["model_name"]})...')
        X_tr_raw = build_features(proc_train['bert_texts'], proc_train['meta'],
                                  bm['model_name'], batch_size, device)
        X_vl_raw = build_features(proc_val['bert_texts'],   proc_val['meta'],
                                  bm['model_name'], batch_size, device)
        X_tr, reducer = fit_transform(X_tr_raw, cfg, y=y_train)
        X_vl = transform(reducer, X_vl_raw)
        all_X_train.append(X_tr)
        all_X_val.append(X_vl)
        bm_names.append(bm['id'])
        print(f'   {X_tr_raw.shape} → {X_tr.shape}')

    # ── 建立所有條件 ──────────────────────────────────────────────────────────
    conditions = []
    for k in [1, 2, 3]:
        for combo in itertools.combinations(range(len(bm_names)), k):
            label = '+'.join(bm_names[i] for i in combo)
            for use_lr in [False, True]:
                clf_label = 'XGB+LR' if use_lr else 'XGB   '
                conditions.append((f'{k} model  [{label}]', combo, use_lr, clf_label))

    # ── 執行消融 ─────────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'{"條件":<45} {"clf":<8} {"meta":>6} {"Train F1":>10} {"Val F1":>10}')
    print(f'{"-"*80}')

    results = []
    for label, combo, use_lr, clf_label in conditions:
        X_tr_list = [all_X_train[i] for i in combo]
        X_vl_list = [all_X_val[i]   for i in combo]
        tr_f1, vl_f1, meta_dim = run_condition(
            X_tr_list, y_train, X_vl_list, y_val, use_lr, cfg
        )
        results.append((label, clf_label, meta_dim, tr_f1, vl_f1))
        print(f'{label:<45} {clf_label:<8} {f"(n,{meta_dim})":>6} {tr_f1:>10.4f} {vl_f1:>10.4f}')

    print(f'{"="*80}')
    best = max(results, key=lambda x: x[4])
    print(f'\nBest Val F1: {best[4]:.4f}  →  {best[0]}  [{best[1]}]')


if __name__ == '__main__':
    main()
