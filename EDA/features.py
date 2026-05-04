import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


def build_topic_features(train_texts: list, cfg: dict) -> dict:
    """
    CountVectorizer + LDA on train_texts.
    若 lda.enabled = false，回傳空矩陣（shape (n, 0)），np.hstack 不受影響。
    """
    fc = cfg['features']
    n  = len(train_texts)

    if not fc['lda']['enabled']:
        return {'count_vec': None, 'lda': None, 'X_train_topics': np.empty((n, 0))}

    count_vec = CountVectorizer(
        max_features = fc['count_vec']['max_features'],
        stop_words   = fc['count_vec']['stop_words'],
        min_df       = fc['count_vec']['min_df'],
    )
    train_mat = count_vec.fit_transform(train_texts)

    lda = LatentDirichletAllocation(
        n_components = fc['lda']['n_topics'],
        random_state = fc['lda']['random_state'],
    )
    return {
        'count_vec':      count_vec,
        'lda':            lda,
        'X_train_topics': lda.fit_transform(train_mat),
    }


def transform_topic_features(count_vec, lda, texts: list, cfg: dict) -> np.ndarray:
    """Val / Test 用。LDA 未啟用時回傳空矩陣。"""
    if not cfg['features']['lda']['enabled']:
        return np.empty((len(texts), 0))
    return lda.transform(count_vec.transform(texts))


def encode_texts(texts: list, cfg: dict, device: str) -> np.ndarray:
    """Encode texts with SentenceTransformer."""
    st = cfg['features']['sentence_transformer']
    model = SentenceTransformer(st['model_name'], device=device)
    return model.encode(texts, batch_size=st['batch_size'], show_progress_bar=True)


def build_nli_features(texts: list, cfg: dict, device: str) -> np.ndarray:
    """
    Zero-shot NLI domain classification（stateless，不需要 fit）。
    輸出 (n, len(candidate_labels)) 的機率矩陣，欄位順序固定與 candidate_labels 一致。
    nli.enabled = false 時回傳空矩陣，np.hstack 不受影響。
    """
    nli_cfg = cfg['features'].get('nli', {})
    if not nli_cfg.get('enabled', False):
        return np.empty((len(texts), 0))

    from transformers import pipeline

    nli_device = device if nli_cfg['device'] == 'auto' else nli_cfg['device']
    labels     = nli_cfg['candidate_labels']

    classifier = pipeline(
        'zero-shot-classification',
        model  = nli_cfg['model_name'],
        device = nli_device,
    )

    print(f'>> NLI inference on {len(texts)} texts (device={nli_device})...')
    raw = classifier(texts, candidate_labels=labels, batch_size=nli_cfg['batch_size'])

    # pipeline 依分數降序排列，需還原成 candidate_labels 的固定順序
    results = []
    for r in raw:
        score_map = dict(zip(r['labels'], r['scores']))
        results.append([score_map[label] for label in labels])

    return np.array(results, dtype=np.float32)
