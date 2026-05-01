import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def encode(texts: list, model_name: str, batch_size: int, device: str) -> np.ndarray:
    st = SentenceTransformer(model_name, device=device)
    X = st.encode(texts, batch_size=batch_size, show_progress_bar=True)
    del st
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X


def build_features(texts: list, meta: np.ndarray,
                   model_name: str, batch_size: int, device: str) -> np.ndarray:
    """Encode texts and concatenate meta features → (n, embedding_dim + n_meta)."""
    X_emb = encode(texts, model_name, batch_size, device)
    return np.hstack([X_emb, meta])
