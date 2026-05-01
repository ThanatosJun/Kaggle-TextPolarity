import numpy as np
from sklearn.decomposition import PCA


def fit(X: np.ndarray, cfg: dict):
    """Fit a reducer on X. Returns None if dimension_reduction.method == 'none'."""
    dr = cfg.get('dimension_reduction', {})
    method = dr.get('method', 'none')
    if method == 'pca':
        reducer = PCA(n_components=dr['n_components'],
                      random_state=cfg['global']['random_seed'])
        reducer.fit(X)
        var = reducer.explained_variance_ratio_.sum()
        print(f'   PCA {dr["n_components"]}-dim  →  cumulative variance {var*100:.1f}%')
        return reducer
    return None


def transform(reducer, X: np.ndarray) -> np.ndarray:
    """Apply reducer. If None (no reduction), return X unchanged."""
    if reducer is None:
        return X
    return reducer.transform(X)


def fit_transform(X: np.ndarray, cfg: dict) -> tuple:
    """Fit on X, return (X_reduced, reducer). reducer is None when disabled."""
    reducer = fit(X, cfg)
    return transform(reducer, X), reducer
