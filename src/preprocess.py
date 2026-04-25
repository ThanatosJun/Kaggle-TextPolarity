import re
import numpy as np
import pandas as pd


def clean_for_tfidf(text: str, cfg: dict) -> str:
    pre = cfg['preprocessing']
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = text.replace(pre['track_a']['lrb_token'], ' ')
    text = text.replace(pre['track_a']['rrb_token'], ' ')
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_for_bert(text: str, cfg: dict) -> str:
    pre = cfg['preprocessing']
    if not isinstance(text, str):
        return ''
    text = text.replace(pre['track_a']['lrb_token'], pre['track_b']['lrb_replace'])
    text = text.replace(pre['track_a']['rrb_token'], pre['track_b']['rrb_replace'])
    # 修復被斷開的縮寫（e.g., "does n't" → "doesn't"）
    for suffix in ("n't", "'s", "'re", "'ll", "'ve", "'m", "'d"):
        text = re.sub(r"\s+" + re.escape(suffix), suffix, text)
    # 修復標點前多餘空白（e.g., "word ." → "word."）
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_meta(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    meta = {}
    if cfg['features']['meta']['use_question_mark']:
        meta['q_mark'] = df['TEXT'].apply(lambda x: str(x).count('?'))
    if cfg['features']['meta']['use_exclaim_mark']:
        meta['e_mark'] = df['TEXT'].apply(lambda x: str(x).count('!'))
    return pd.DataFrame(meta).values


def run(df: pd.DataFrame, cfg: dict) -> dict:
    return {
        'tfidf_texts': df['TEXT'].apply(lambda t: clean_for_tfidf(t, cfg)).tolist(),
        'bert_texts':  df['TEXT'].apply(lambda t: clean_for_bert(t, cfg)).tolist(),
        'meta':        extract_meta(df, cfg),
    }
