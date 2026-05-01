import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def build(cfg: dict) -> XGBClassifier:
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
    Stratified K-fold OOF prediction + final model trained on all data.
    Returns (oof_proba (n, 2), final_model).
    Each val fold is predicted by a model that never saw it (no leakage).
    """
    n_folds = cfg['stacking']['cv_folds']
    seed    = cfg['global']['random_seed']
    kf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros((len(y), 2))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f'     fold {fold + 1}/{n_folds}')
        m = build(cfg)
        m.fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict_proba(X[val_idx])

    final = build(cfg)
    final.fit(X, y)
    return oof, final
