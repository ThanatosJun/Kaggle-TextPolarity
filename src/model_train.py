import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from xgboost import XGBClassifier


def build_xgb(cfg: dict) -> XGBClassifier:
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators     = c['n_estimators'],
        learning_rate    = c['learning_rate'],
        max_depth        = c['max_depth'],
        random_state     = c['random_state'],
        eval_metric      = c['eval_metric'],
        device           = c['device'],
        subsample        = c['subsample'],
        colsample_bytree = c['colsample_bytree'],
        reg_alpha        = c['reg_alpha'],
        reg_lambda       = c['reg_lambda'],
    )


def build_logreg(cfg: dict) -> LogisticRegression:
    seed = cfg['global']['random_seed']
    c    = cfg.get('models', {}).get('logreg_base', {})
    return LogisticRegression(
        C            = c.get('C', 1.0),
        max_iter     = c.get('max_iter', 1000),
        random_state = seed,
    )


def get_oof_multi(X: np.ndarray, y: np.ndarray, cfg: dict, groups: np.ndarray = None, device: str = 'cpu') -> tuple:
    """
    同一組 K-fold 同時跑 XGBoost 和 L2 LogReg OOF。
    groups: 若提供，改用 GroupKFold 確保配對樣本（original/nlpaug）不跨 fold。
    Returns:
        oof     : (n, 4)  hstack([xgb_oof, lr_oof])
        final_xgb : XGBClassifier trained on full X
        final_lr  : LogisticRegression trained on full X
    """
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']

    if groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(X, y, groups=groups)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(X, y)

    oof_xgb = np.zeros((len(y), 2))
    oof_lr  = np.zeros((len(y), 2))

    def _to_xgb(arr):
        if device == 'cuda':
            import cupy as cp
            return cp.asarray(arr)
        return arr

    for fold, (tr_idx, val_idx) in enumerate(split_iter):
        print(f'     fold {fold + 1}/{n_folds}')
        xgb = build_xgb(cfg)
        xgb.fit(_to_xgb(X[tr_idx]), y[tr_idx])
        oof_xgb[val_idx] = xgb.predict_proba(_to_xgb(X[val_idx]))

        lr = build_logreg(cfg)
        lr.fit(X[tr_idx], y[tr_idx])
        oof_lr[val_idx] = lr.predict_proba(X[val_idx])

    final_xgb = build_xgb(cfg)
    final_xgb.fit(_to_xgb(X), y)

    final_lr = build_logreg(cfg)
    final_lr.fit(X, y)

    oof = np.hstack([oof_xgb, oof_lr])   # (n, 4)
    return oof, final_xgb, final_lr


def get_oof_multi_aug(X_orig: np.ndarray, X_aug: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:  # kept for compat
    """
    OOF on all original + aug samples. Each fold:
      - train: orig[tr_idx] + aug[tr_idx]
      - val  : orig[val_idx] + aug[val_idx]（兩者皆未進 training，無 leakage）
    Returns oof shape (2*n, 4)：前 n 列為 orig OOF，後 n 列為 aug OOF。
    Final model trained on all original + all nlpaug.
    """
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n = len(y)
    oof_xgb_orig = np.zeros((n, 2))
    oof_lr_orig  = np.zeros((n, 2))
    oof_xgb_aug  = np.zeros((n, 2))
    oof_lr_aug   = np.zeros((n, 2))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_orig, y)):
        print(f'     fold {fold + 1}/{n_folds}')
        X_tr = np.vstack([X_orig[tr_idx], X_aug[tr_idx]])
        y_tr = np.concatenate([y[tr_idx], y[tr_idx]])

        xgb = build_xgb(cfg)
        xgb.fit(X_tr, y_tr)
        oof_xgb_orig[val_idx] = xgb.predict_proba(X_orig[val_idx])
        oof_xgb_aug[val_idx]  = xgb.predict_proba(X_aug[val_idx])

        lr = build_logreg(cfg)
        lr.fit(X_tr, y_tr)
        oof_lr_orig[val_idx] = lr.predict_proba(X_orig[val_idx])
        oof_lr_aug[val_idx]  = lr.predict_proba(X_aug[val_idx])

    X_full = np.vstack([X_orig, X_aug])
    y_full = np.concatenate([y, y])

    final_xgb = build_xgb(cfg)
    final_xgb.fit(X_full, y_full)

    final_lr = build_logreg(cfg)
    final_lr.fit(X_full, y_full)

    oof_orig = np.hstack([oof_xgb_orig, oof_lr_orig])  # (n, 4)
    oof_aug  = np.hstack([oof_xgb_aug,  oof_lr_aug])   # (n, 4)
    oof      = np.vstack([oof_orig, oof_aug])           # (2n, 4)
    return oof, final_xgb, final_lr


# backward-compatible alias
def build(cfg: dict) -> XGBClassifier:
    return build_xgb(cfg)


def get_oof_and_final(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """XGBoost-only OOF（單模型實驗用）。Returns (oof (n,2), final_model)."""
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros((len(y), 2))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'     fold {fold + 1}/{n_folds}')
        m = build_xgb(cfg)
        m.fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict_proba(X[val_idx])

    final = build_xgb(cfg)
    final.fit(X, y)
    return oof, final
