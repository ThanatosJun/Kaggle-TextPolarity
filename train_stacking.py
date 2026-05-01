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
from sklearn.model_selection import train_test_split

from src.data_preprocess import run as preprocess
from src.dimension_decrease import fit_transform, transform
from src.experiment import log_to_csv, save_metrics, snapshot_config
from src.model_stack import train as train_meta
from src.model_train import get_oof_and_final
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
    base_cfgs  = st_cfg['base_models']
    exp        = cfg['experiment']
    dr_method  = cfg.get('dimension_reduction', {}).get('method', 'none')

    print(f'Device: {device}')
    print(f'Base models: {[b["id"] for b in base_cfgs]}')
    print(f'CV folds: {st_cfg["cv_folds"]} | Meta: {st_cfg["meta_model"]["type"]}')
    print(f'Dimension reduction: {dr_method}')

    # ── 1. 載入資料 ───────────────────────────────────────────────────────────
    full_df = pd.read_csv(ROOT / cfg['paths']['train'])
    train_df, val_df = train_test_split(
        full_df, test_size=val_ratio, stratify=full_df['LABEL'], random_state=seed,
    )
    print(f'\nFull: {full_df.shape} | Train: {train_df.shape} | Val: {val_df.shape}')

    # ── 2. 前處理（data_preprocess）───────────────────────────────────────────
    print('\n>> Preprocessing...')
    proc_train = preprocess(train_df, cfg)
    proc_val   = preprocess(val_df,   cfg)
    y_train    = train_df['LABEL'].values
    y_val      = val_df['LABEL'].values

    # ── 3. 每個 base model：encode → 降維 → OOF ─────────────────────────────
    models_dir = ROOT / cfg['paths']['models_dir'] / exp['id']
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_list       = []
    val_proba_list = []

    for bm in base_cfgs:
        print(f'\n── Base model: {bm["id"]}  ({bm["model_name"]}) ──')

        # text_transform：encode + meta 拼接
        X_train_raw = build_features(
            proc_train['bert_texts'], proc_train['meta'],
            bm['model_name'], batch_size, device,
        )
        X_val_raw = build_features(
            proc_val['bert_texts'], proc_val['meta'],
            bm['model_name'], batch_size, device,
        )
        print(f'   Raw feature shape: {X_train_raw.shape}')

        # dimension_decrease：PCA fit on train, transform val
        X_train, reducer = fit_transform(X_train_raw, cfg)
        X_val_            = transform(reducer, X_val_raw)
        if reducer is not None:
            joblib.dump(reducer, models_dir / f'reducer_{bm["id"]}.pkl')
        print(f'   Reduced shape: {X_train.shape}')

        # model_train：5-fold OOF + final model
        print(f'   OOF ({st_cfg["cv_folds"]}-fold CV)...')
        oof, final_model = get_oof_and_final(X_train, y_train, cfg)
        oof_list.append(oof)
        val_proba_list.append(final_model.predict_proba(X_val_))

        final_model.save_model(str(models_dir / f'xgb_{bm["id"]}.json'))
        print(f'   Saved: xgb_{bm["id"]}.json')

    # ── 4. model_stack：Meta model（Train split）─────────────────────────────
    print('\n── Meta model ──')
    meta_train_X = np.hstack(oof_list)
    meta_val_X   = np.hstack(val_proba_list)
    print(f'Meta feature shape — Train: {meta_train_X.shape} | Val: {meta_val_X.shape}')

    meta_model = train_meta(meta_train_X, y_train, cfg)

    # ── 5. 評估 ───────────────────────────────────────────────────────────────
    train_metrics = print_metrics(y_train, meta_model.predict(meta_train_X), 'Train (OOF)')
    val_metrics   = print_metrics(y_val,   meta_model.predict(meta_val_X),   'Val')

    # ── 6. 儲存 experiment metrics ────────────────────────────────────────────
    exp_dir = ROOT / 'experiments' / exp['id']
    save_metrics(exp_dir, {
        'experiment_id': exp['id'],
        'val_ratio':     val_ratio,
        'train_metrics': train_metrics,
        'val_metrics':   val_metrics,
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

    # ── 7. 全量資料重新訓練（最終提交用）────────────────────────────────────
    print('\n── Retrain on full data ──')
    proc_full = preprocess(full_df, cfg)
    y_full    = full_df['LABEL'].values

    oof_full_list = []
    for bm in base_cfgs:
        print(f'\n   {bm["id"]}...')
        X_full_raw = build_features(
            proc_full['bert_texts'], proc_full['meta'],
            bm['model_name'], batch_size, device,
        )
        X_full, reducer_full = fit_transform(X_full_raw, cfg)
        if reducer_full is not None:
            joblib.dump(reducer_full, models_dir / f'reducer_{bm["id"]}_full.pkl')

        oof_full, final_full = get_oof_and_final(X_full, y_full, cfg)
        oof_full_list.append(oof_full)
        final_full.save_model(str(models_dir / f'xgb_{bm["id"]}_full.json'))
        print(f'   Saved: xgb_{bm["id"]}_full.json')

    meta_full_X     = np.hstack(oof_full_list)
    meta_model_full = train_meta(meta_full_X, y_full, cfg)
    joblib.dump(meta_model_full, models_dir / 'meta_model_full.pkl')
    joblib.dump(meta_model,      models_dir / 'meta_model.pkl')
    print(f'\n✅ All artifacts saved: {models_dir}')


if __name__ == '__main__':
    main()
