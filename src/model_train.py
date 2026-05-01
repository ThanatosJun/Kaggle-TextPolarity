import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
        penalty      = 'l2',
        max_iter     = c.get('max_iter', 1000),
        random_state = seed,
    )


def get_oof_multi(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """
    同一組 K-fold 同時跑 XGBoost 和 L2 LogReg OOF。
    Returns:
        oof     : (n, 4)  hstack([xgb_oof, lr_oof])
        final_xgb : XGBClassifier trained on full X
        final_lr  : LogisticRegression trained on full X
    """
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_xgb = np.zeros((len(y), 2))
    oof_lr  = np.zeros((len(y), 2))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'     fold {fold + 1}/{n_folds}')
        xgb = build_xgb(cfg)
        xgb.fit(X[tr_idx], y[tr_idx])
        oof_xgb[val_idx] = xgb.predict_proba(X[val_idx])

        lr = build_logreg(cfg)
        lr.fit(X[tr_idx], y[tr_idx])
        oof_lr[val_idx] = lr.predict_proba(X[val_idx])

    final_xgb = build_xgb(cfg)
    final_xgb.fit(X, y)

    final_lr = build_logreg(cfg)
    final_lr.fit(X, y)

    oof = np.hstack([oof_xgb, oof_lr])   # (n, 4)
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
