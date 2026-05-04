import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


def _n_meta(cfg: dict) -> int:
    meta_cfg = cfg.get('features', {}).get('meta', {})
    n = 0
    if meta_cfg.get('use_question_mark', False): n += 1
    if meta_cfg.get('use_exclaim_mark',  False): n += 1
    if meta_cfg.get('use_vader',         False): n += 4
    return n


def fit(X: np.ndarray, cfg: dict, y: np.ndarray = None):
    """
    Fit a reducer on the encoder embedding only (meta columns are excluded).
    Returns None if dimension_reduction.method == 'none'.
    Stores n_meta_ on the reducer so transform() can re-attach meta columns.
    """
    dr = cfg.get('dimension_reduction', {})
    method = dr.get('method', 'none')
    n = dr.get('n_components', 32)
    n_meta = _n_meta(cfg)

    X_emb = X[:, :-n_meta] if n_meta > 0 else X

    if method == 'pca':
        reducer = PCA(n_components=n, random_state=cfg['global']['random_seed'])
        reducer.fit(X_emb)
        reducer.n_meta_ = n_meta
        var = reducer.explained_variance_ratio_.sum()
        print(f'   PCA {n}-dim  →  cumulative variance {var*100:.1f}%')
        return reducer

    if method == 'pls':
        if y is None:
            raise ValueError('PLS requires y (labels) for fitting.')
        reducer = PLSRegression(n_components=n)
        reducer.fit(X_emb, y)
        reducer.n_meta_ = n_meta
        print(f'   PLS {n}-dim  →  supervised, fit with labels')
        return reducer

    return None


def transform(reducer, X: np.ndarray) -> np.ndarray:
    """Apply reducer to embedding only; re-attach meta columns afterwards."""
    if reducer is None:
        return X
    n_meta = getattr(reducer, 'n_meta_', 0)
    X_emb = X[:, :-n_meta] if n_meta > 0 else X
    X_meta = X[:, -n_meta:] if n_meta > 0 else None

    result = reducer.transform(X_emb)
    if isinstance(result, tuple):
        result = result[0]

    return np.hstack([result, X_meta]) if X_meta is not None else result


def fit_transform(X: np.ndarray, cfg: dict, y: np.ndarray = None) -> tuple:
    """Fit on embedding, return (X_reduced_with_meta, reducer). reducer is None when disabled."""
    reducer = fit(X, cfg, y=y)
    return transform(reducer, X), reducer
