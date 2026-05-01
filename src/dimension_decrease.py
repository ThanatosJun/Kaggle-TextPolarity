import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


def fit(X: np.ndarray, cfg: dict, y: np.ndarray = None):
    """
    Fit a reducer on X (and y for PLS).
    Returns None if dimension_reduction.method == 'none'.
    """
    dr = cfg.get('dimension_reduction', {})
    method = dr.get('method', 'none')
    n = dr.get('n_components', 32)

    if method == 'pca':
        reducer = PCA(n_components=n, random_state=cfg['global']['random_seed'])
        reducer.fit(X)
        var = reducer.explained_variance_ratio_.sum()
        print(f'   PCA {n}-dim  →  cumulative variance {var*100:.1f}%')
        return reducer

    if method == 'pls':
        if y is None:
            raise ValueError('PLS requires y (labels) for fitting.')
        reducer = PLSRegression(n_components=n)
        reducer.fit(X, y)
        print(f'   PLS {n}-dim  →  supervised, fit with labels')
        return reducer

    return None


def transform(reducer, X: np.ndarray) -> np.ndarray:
    """Apply reducer. If None (no reduction), return X unchanged."""
    if reducer is None:
        return X
    # PLSRegression.transform returns (X_scores, Y_scores); we only need X_scores
    result = reducer.transform(X)
    if isinstance(result, tuple):
        return result[0]
    return result


def fit_transform(X: np.ndarray, cfg: dict, y: np.ndarray = None) -> tuple:
    """Fit on X (+ y for PLS), return (X_reduced, reducer). reducer is None when disabled."""
    reducer = fit(X, cfg, y=y)
    return transform(reducer, X), reducer
