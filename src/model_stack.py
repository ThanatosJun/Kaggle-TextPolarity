import numpy as np
from sklearn.linear_model import LogisticRegression


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


def predict(model: LogisticRegression, meta_X: np.ndarray) -> np.ndarray:
    return model.predict(meta_X)
