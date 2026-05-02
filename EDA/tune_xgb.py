"""
XGBoost 超參數調整（Optuna）

流程：
  1. 用 train split（1600 筆）做 inner 5-fold CV 評估每組參數
  2. Optuna 以 TPE 演算法搜尋 n_trials 次
  3. 找到最佳參數後印出，並提示更新 config.yaml

val set（400 筆）完全不參與調參，確保評估無洩漏。

用法：
    python tune_xgb.py               # 預設 n_trials=50
    python tune_xgb.py --trials 100
"""

import argparse
import sys
import warnings
import yaml
import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.data_preprocess import run as preprocess
from src.text_transform import build_features
from src.dimension_decrease import fit_transform
from src.utils import set_seed


def load_cfg(path='config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    v = cfg['global']['device']
    if v == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def build_xgb(trial: optuna.Trial, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = trial.suggest_int('n_estimators',    100, 800),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3,  log=True),
        max_depth         = trial.suggest_int('max_depth',       2,   7),
        subsample         = trial.suggest_float('subsample',     0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha',     1e-3, 5.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda',    1e-3, 5.0, log=True),
        random_state      = seed,
        eval_metric       = 'logloss',
        device            = 'cuda' if torch.cuda.is_available() else 'cpu',
    )


def make_objective(X_list, y, cfg):
    """回傳 Optuna objective function，對多個 base model 的 X 做平均 CV F1。"""
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    def objective(trial):
        scores = []
        for X in X_list:
            fold_f1s = []
            for tr_idx, val_idx in kf.split(X, y):
                m = build_xgb(trial, seed)
                m.fit(X[tr_idx], y[tr_idx])
                pred = m.predict(X[val_idx])
                fold_f1s.append(f1_score(y[val_idx], pred, average='macro'))
            scores.append(np.mean(fold_f1s))
        return np.mean(scores)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    cfg    = load_cfg()
    seed   = cfg['global']['random_seed']
    device = resolve_device(cfg)
    set_seed(cfg)
    batch_size = cfg['features']['sentence_transformer']['batch_size']
    base_cfgs  = cfg['stacking']['base_models']

    # ── 資料：只用 train split，val 不碰 ─────────────────────────────────────
    full_df = pd.read_csv(cfg['paths']['train'])
    train_df, _ = train_test_split(
        full_df, test_size=cfg['evaluation']['val_ratio'],
        stratify=full_df['LABEL'], random_state=seed,
    )
    y_train = train_df['LABEL'].values
    print(f'Train split: {train_df.shape}  (val set not used in tuning)')

    # ── 前處理 + encode + 降維（與 train_stacking 一致）────────────────────
    proc = preprocess(train_df, cfg)
    X_list = []
    for bm in base_cfgs:
        print(f'>> Encoding {bm["id"]}...')
        X_raw = build_features(proc['bert_texts'], proc['meta'],
                               bm['model_name'], batch_size, device)
        X, _ = fit_transform(X_raw, cfg, y=y_train)
        print(f'   Shape after DR: {X.shape}')
        X_list.append(X)

    # ── Optuna 搜尋 ───────────────────────────────────────────────────────
    print(f'\n>> Optuna: {args.trials} trials  (inner {cfg["stacking"]["cv_folds"]}-fold CV)')
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(make_objective(X_list, y_train, cfg),
                   n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    best_f1 = study.best_value
    print(f'\n{"="*55}')
    print(f'Best inner CV F1: {best_f1:.4f}')
    print(f'{"="*55}')
    print('\n最佳參數（貼入 config.yaml → models.xgboost）：\n')
    print(f'  n_estimators:    {best["n_estimators"]}')
    print(f'  learning_rate:   {best["learning_rate"]:.4f}')
    print(f'  max_depth:       {best["max_depth"]}')
    print(f'  subsample:       {best["subsample"]:.3f}')
    print(f'  colsample_bytree:{best["colsample_bytree"]:.3f}')
    print(f'  reg_alpha:       {best["reg_alpha"]:.4f}')
    print(f'  reg_lambda:      {best["reg_lambda"]:.4f}')


if __name__ == '__main__':
    main()
