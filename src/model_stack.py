import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def train(meta_X: np.ndarray, y: np.ndarray, cfg: dict) -> LogisticRegression:
    """Train meta model on stacked OOF probabilities."""
    mc = cfg['stacking']['meta_model']
    model = LogisticRegression(
        C            = mc.get('C', 1.0),
        max_iter     = mc.get('max_iter', 1000),
        random_state = cfg['global']['random_seed'],
    )
    model.fit(meta_X, y)
    return model


def find_threshold(model: LogisticRegression, meta_X: np.ndarray, y: np.ndarray,
                   step: float = 0.01) -> tuple[float, float]:
    """Sweep thresholds on the given split; return (best_threshold, best_f1)."""
    proba = model.predict_proba(meta_X)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, step):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 4), round(best_f1, 4)


def predict_with_threshold(model: LogisticRegression, meta_X: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
    proba = model.predict_proba(meta_X)[:, 1]
    return (proba >= threshold).astype(int)


def save_threshold(path: Path, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'threshold': threshold}, f)


def load_threshold(path: Path, default: float = 0.5) -> float:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)['threshold']
