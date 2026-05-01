import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


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
    text = text.replace(pre['special_tokens']['num_token'],    'number')
    text = text.replace(pre['special_tokens']['extend_token'], 'number')
    for suffix in ("n't", "'s", "'re", "'ll", "'ve", "'m", "'d"):
        text = re.sub(r"\s+" + re.escape(suffix), suffix, text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_meta(df: pd.DataFrame, cfg: dict, bert_texts: list) -> np.ndarray:
    meta = {}
    if cfg['features']['meta']['use_question_mark']:
        meta['q_mark'] = df['TEXT'].apply(lambda x: str(x).count('?'))
    if cfg['features']['meta']['use_exclaim_mark']:
        meta['e_mark'] = df['TEXT'].apply(lambda x: str(x).count('!'))
    if cfg['features']['meta'].get('use_vader', False):
        scores = [_vader.polarity_scores(t) for t in bert_texts]
        meta['vader_neg']      = [s['neg']      for s in scores]
        meta['vader_neu']      = [s['neu']      for s in scores]
        meta['vader_pos']      = [s['pos']      for s in scores]
        meta['vader_compound'] = [s['compound'] for s in scores]
    return pd.DataFrame(meta).values


def run(df: pd.DataFrame, cfg: dict) -> dict:
    bert_texts = df['TEXT'].apply(lambda t: clean_for_bert(t, cfg)).tolist()
    return {
        'tfidf_texts': df['TEXT'].apply(lambda t: clean_for_tfidf(t, cfg)).tolist(),
        'bert_texts':  bert_texts,
        'meta':        extract_meta(df, cfg, bert_texts),
    }
