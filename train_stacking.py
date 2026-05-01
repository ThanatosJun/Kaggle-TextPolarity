import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from src.data_preprocess import run as preprocess
from src.dimension_decrease import fit_transform, transform
from src.experiment import log_to_csv, save_metrics, snapshot_config
from src.model_stack import (
    build_meta, find_threshold,
    predict_with_threshold, save_threshold,
)
from src.model_train import build_xgb, build_logreg, get_oof_multi
from src.text_transform import build_features
from src.utils import set_seed


def find_project_root(marker: str = 'config') -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(f"Project root not found (marker='{marker}')")


def resolve_device(value: str) -> str:
    if value == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return value


def to_xgb(X: np.ndarray, device: str) -> np.ndarray:
    if device == 'cuda':
        import cupy as cp
        return cp.asarray(X)
    return X


def print_metrics(y_true, y_pred, label: str = '') -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f'\n[ {label} ]')
    print(f'  {"Accuracy":<12} {acc:.4f}')
    print(f'  {"Precision":<12} {prec:.4f}')
    print(f'  {"Recall":<12} {rec:.4f}')
    print(f'  {"F1":<12} {f1:.4f}')
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    return {'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4)}


def main() -> None:
    ROOT = find_project_root()
    config_path = ROOT / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device     = resolve_device(cfg['global']['device'])
    seed       = cfg['global']['random_seed']
    set_seed(cfg)
    val_ratio  = cfg['evaluation']['val_ratio']
    batch_size = cfg['features']['sentence_transformer']['batch_size']
    st_cfg     = cfg['stacking']
    n_folds    = st_cfg['cv_folds']
    base_cfgs  = st_cfg['base_models']
    exp        = cfg['experiment']
    dr_method  = cfg.get('dimension_reduction', {}).get('method', 'none')

    print(f'Device: {device}')
    print(f'Base models: {[b["id"] for b in base_cfgs]}')
    print(f'CV folds: {n_folds} | Meta: {st_cfg["meta_model"]["type"]}')
    print(f'Dimension reduction: {dr_method}')

    # ── 1. 載入資料 ───────────────────────────────────────────────────────────
    full_df      = pd.read_csv(ROOT / cfg['paths']['train'])
    is_augmented = 'source' in full_df.columns

    if is_augmented:
        # val = 400 pure original（完全不參與訓練）
        # train_df = 1600 orig + 1600 對應 nlpaug = 3200
        orig_df    = full_df[full_df['source'] == 'original'].reset_index(drop=True)
        train_orig, val_df = train_test_split(
            orig_df, test_size=val_ratio, stratify=orig_df['LABEL'], random_state=seed,
        )
        train_ids = set(train_orig['row_id'].astype(str))
        train_df  = full_df[
            full_df['row_id'].astype(str)
            .str.replace('_nlpaug', '', regex=False)
            .isin(train_ids)
        ].reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        # GroupKFold：配對的 orig + nlpaug 永遠在同一 fold，避免 OOF leakage
        train_groups = train_df['row_id'].astype(str).str.replace('_nlpaug', '', regex=False).values
        print(f'Augmented: train={len(train_df)} (orig+nlpaug) | val={len(val_df)} (orig only)')
    else:
        train_df, val_df = train_test_split(
            full_df, test_size=val_ratio, stratify=full_df['LABEL'], random_state=seed,
        )
        train_groups = None
        print(f'Train: {len(train_df)} | Val: {len(val_df)}')

    y_train = train_df['LABEL'].values
    y_val   = val_df['LABEL'].values

    # ── 2. 前處理 ─────────────────────────────────────────────────────────────
    print('\n>> Preprocessing...')
    proc_train = preprocess(train_df, cfg)
    proc_val   = preprocess(val_df,   cfg)

    # ── 3. Fold 定義 ──────────────────────────────────────────────────────────
    models_dir = ROOT / cfg['paths']['models_dir'] / exp['id']
    models_dir.mkdir(parents=True, exist_ok=True)

    if train_groups is not None:
        kf    = GroupKFold(n_splits=n_folds)
        folds = list(kf.split(np.zeros(len(y_train)), y_train, groups=train_groups))
    else:
        kf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(np.zeros(len(y_train)), y_train))

    # ── 4. Base model OOF ─────────────────────────────────────────────────────
    oof_list       = []   # each: (n_train, 4)
    val_proba_list = []   # each: (n_val, 4)
    base_model_metrics = {}

    for bm in base_cfgs:
        print(f'\n── Base model: {bm["id"]}  ({bm["model_name"]}) ──')

        X_train_raw = build_features(
            proc_train['bert_texts'], proc_train['meta'],
            bm['model_name'], batch_size, device,
        )
        X_val_raw = build_features(
            proc_val['bert_texts'], proc_val['meta'],
            bm['model_name'], batch_size, device,
        )

        X_train, reducer = fit_transform(X_train_raw, cfg, y=y_train)
        X_val             = transform(reducer, X_val_raw)
        if reducer is not None:
            joblib.dump(reducer, models_dir / f'reducer_{bm["id"]}.pkl')
        print(f'   Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]} | Features: {X_train.shape[1]}')

        oof = np.zeros((len(y_train), 4))
        xgb_train_f1s, xgb_val_f1s = [], []
        lr_train_f1s,  lr_val_f1s  = [], []

        print(f'   OOF ({n_folds}-fold)...')
        for fold_i, (tr_idx, val_idx) in enumerate(folds):
            print(f'     fold {fold_i + 1}/{n_folds}  [train={len(tr_idx)} | val={len(val_idx)}]')
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

            xgb = build_xgb(cfg);    xgb.fit(to_xgb(X_tr, device), y_tr)
            lr  = build_logreg(cfg); lr.fit(X_tr, y_tr)

            oof[val_idx] = np.hstack([
                xgb.predict_proba(to_xgb(X_train[val_idx], device)),
                lr.predict_proba(X_train[val_idx]),
            ])

            xgb_val_f1s.append(f1_score(
                y_train[val_idx],
                (xgb.predict_proba(to_xgb(X_train[val_idx], device))[:, 1] >= 0.5).astype(int),
                average='macro', zero_division=0,
            ))
            xgb_train_f1s.append(f1_score(
                y_tr,
                (xgb.predict_proba(to_xgb(X_tr, device))[:, 1] >= 0.5).astype(int),
                average='macro', zero_division=0,
            ))
            lr_val_f1s.append(f1_score(
                y_train[val_idx],
                (lr.predict_proba(X_train[val_idx])[:, 1] >= 0.5).astype(int),
                average='macro', zero_division=0,
            ))
            lr_train_f1s.append(f1_score(
                y_tr,
                (lr.predict_proba(X_tr)[:, 1] >= 0.5).astype(int),
                average='macro', zero_division=0,
            ))

        # final base model（在全量 train_df 上訓練）
        final_xgb = build_xgb(cfg);    final_xgb.fit(to_xgb(X_train, device), y_train)
        final_lr  = build_logreg(cfg); final_lr.fit(X_train, y_train)
        final_xgb.save_model(str(models_dir / f'xgb_{bm["id"]}.json'))
        joblib.dump(final_lr, models_dir / f'lr_{bm["id"]}.pkl')

        # holdout val 預測（用 final model，不用 OOF model）
        val_proba = np.hstack([
            final_xgb.predict_proba(to_xgb(X_val, device)),
            final_lr.predict_proba(X_val),
        ])

        oof_list.append(oof)
        val_proba_list.append(val_proba)
        base_model_metrics[bm['id']] = {
            'xgb': {
                'train_f1': round(float(np.mean(xgb_train_f1s)), 4),
                'val_f1':   round(float(np.mean(xgb_val_f1s)),   4),
            },
            'lr': {
                'train_f1': round(float(np.mean(lr_train_f1s)), 4),
                'val_f1':   round(float(np.mean(lr_val_f1s)),   4),
            },
        }
        print(f'   Saved: xgb_{bm["id"]}.json + lr_{bm["id"]}.pkl')

    # ── 5. Meta model ─────────────────────────────────────────────────────────
    print('\n── Meta model ──')
    meta_train_X = np.hstack(oof_list)        # (n_train, 12)
    meta_val_X   = np.hstack(val_proba_list)  # (n_val, 12)
    print(f'Meta feature shape — Train: {meta_train_X.shape} | Val: {meta_val_X.shape}')

    meta_model = build_meta(cfg)
    meta_model.fit(meta_train_X, y_train)

    # ── 6. Threshold（在 400 holdout val 上搜尋，完全乾淨）─────────────────
    threshold, thr_f1 = find_threshold(meta_model, meta_val_X, y_val)
    print(f'\n   Best threshold: {threshold:.4f}  (Val F1 = {thr_f1:.4f})')

    # ── 7. 評估 ───────────────────────────────────────────────────────────────
    train_metrics = print_metrics(y_train, predict_with_threshold(meta_model, meta_train_X, threshold), 'Train (OOF)')
    val_metrics   = print_metrics(y_val,   predict_with_threshold(meta_model, meta_val_X,   threshold), 'Val (holdout)')

    # ── 8. 儲存 experiment metrics ────────────────────────────────────────────
    exp_dir = ROOT / 'experiments' / exp['id']
    save_metrics(exp_dir, {
        'experiment_id': exp['id'],
        'val_ratio':     val_ratio,
        'train_metrics': train_metrics,
        'val_metrics':   val_metrics,
        'threshold':     threshold,
        'kaggle_score':  None,
        'note':          f'Stacking: {[b["id"] for b in base_cfgs]} | DR: {dr_method}',
    })
    snapshot_config(exp_dir, config_path)
    log_to_csv(
        experiments_dir = ROOT / 'experiments',
        exp_id          = exp['id'],
        exp_name        = exp['name'],
        train_metrics   = train_metrics,
        val_metrics     = val_metrics,
    )

    # ── 9. 全量資料重新訓練（最終提交用）────────────────────────────────────
    print('\n── Retrain on full data ──')
    proc_full = preprocess(full_df, cfg)
    y_full    = full_df['LABEL'].values

    full_groups = full_df['row_id'].astype(str).str.replace('_nlpaug', '', regex=False).values if is_augmented else None

    oof_full_list = []
    for bm in base_cfgs:
        print(f'\n   {bm["id"]}...')
        X_full_raw = build_features(
            proc_full['bert_texts'], proc_full['meta'],
            bm['model_name'], batch_size, device,
        )
        X_full_feat, reducer_full = fit_transform(X_full_raw, cfg, y=y_full)
        if reducer_full is not None:
            joblib.dump(reducer_full, models_dir / f'reducer_{bm["id"]}_full.pkl')

        print(f'   Train samples: {X_full_feat.shape[0]} | Features: {X_full_feat.shape[1]}')
        oof_full, final_xgb_full, final_lr_full = get_oof_multi(X_full_feat, y_full, cfg, groups=full_groups, device=device)
        oof_full_list.append(oof_full)
        final_xgb_full.save_model(str(models_dir / f'xgb_{bm["id"]}_full.json'))
        joblib.dump(final_lr_full, models_dir / f'lr_{bm["id"]}_full.pkl')
        print(f'   Saved: xgb_{bm["id"]}_full.json + lr_{bm["id"]}_full.pkl')

    meta_full_X    = np.hstack(oof_full_list)
    print(f'\n── Meta model (full) ──')
    print(f'   Train samples: {meta_full_X.shape[0]} | Features: {meta_full_X.shape[1]}')
    meta_model_full = build_meta(cfg)
    meta_model_full.fit(meta_full_X, y_full)
    print(f'   Saved: meta_model_full.pkl')

    joblib.dump(meta_model_full, models_dir / 'meta_model_full.pkl')
    joblib.dump(meta_model,      models_dir / 'meta_model.pkl')
    save_threshold(models_dir / 'threshold.json', threshold)
    print(f'   Threshold: {threshold:.4f}')
    print(f'\n✅ All artifacts saved: {models_dir}')


if __name__ == '__main__':
    main()
