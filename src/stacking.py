import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def _build_xgb(cfg: dict) -> XGBClassifier:
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


def get_oof_and_final(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """
    K-fold OOF probabilities + final model trained on all X/y.
    Returns (oof_proba (n, 2), final_model).
    OOF predictions avoid leakage: each val fold is predicted by a model
    that never saw it during training.
    """
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros((len(y), 2))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        m = _build_xgb(cfg)
        m.fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict_proba(X[val_idx])
        print(f'     fold {fold + 1}/{n_folds}')

    final = _build_xgb(cfg)
    final.fit(X, y)
    return oof, final


def train_meta(meta_X: np.ndarray, y: np.ndarray, cfg: dict) -> LogisticRegression:
    mc   = cfg['stacking']['meta_model']
    seed = cfg['global']['random_seed']
    meta = LogisticRegression(
        C        = mc.get('C', 1.0),
        max_iter = mc.get('max_iter', 1000),
        random_state = seed,
    )
    meta.fit(meta_X, y)
    return meta
