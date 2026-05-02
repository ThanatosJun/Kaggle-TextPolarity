import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from xgboost import XGBClassifier

from src.features import build_nli_features, encode_texts, transform_topic_features
from src.preprocess import run as preprocess


def find_project_root(marker: str = 'config') -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / marker).is_dir():
            return candidate
    raise FileNotFoundError(f"Project root not found (marker='{marker}')")


def resolve_device(value: str) -> str:
    if value == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict with a trained experiment model.')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='Experiment ID to load (overrides config experiment.id)')
    args = parser.parse_args()

    ROOT = find_project_root()

    if args.exp_id:
        snapshot = ROOT / 'experiments' / args.exp_id / 'config_snapshot.yaml'
        if not snapshot.exists():
            raise FileNotFoundError(f"No config snapshot found for '{args.exp_id}': {snapshot}")
        with open(snapshot, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['experiment']['id'] = args.exp_id
    else:
        with open(ROOT / 'config' / 'config.yaml', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    device = resolve_device(cfg['global']['device'])
    lda_on = cfg['features']['lda']['enabled']
    nli_on = cfg['features'].get('nli', {}).get('enabled', False)
    exp_id = cfg['experiment']['id']
    print(f'Experiment: {exp_id} | Device: {device} | LDA: {lda_on} | NLI: {nli_on}')

    # ── 載入測試資料 ──────────────────────────────────────────────────────────
    test_df = pd.read_csv(ROOT / cfg['paths']['test'])
    print(f'Test: {test_df.shape}')

    # ── 前處理 ────────────────────────────────────────────────────────────────
    print('>> Preprocessing...')
    processed = preprocess(test_df, cfg)

    # ── 載入 artifacts ────────────────────────────────────────────────────────
    models_dir = ROOT / cfg['paths']['models_dir'] / exp_id
    if lda_on:
        count_vec = joblib.load(models_dir / 'count_vec.pkl')
        lda       = joblib.load(models_dir / 'lda.pkl')
        print(f'>> Loaded count_vec + lda from {models_dir}')
    else:
        count_vec, lda = None, None

    # ── 特徵工程 ──────────────────────────────────────────────────────────────
    X_topics = transform_topic_features(count_vec, lda, processed['tfidf_texts'], cfg)

    print('>> Encoding with SentenceTransformer...')
    X_bert = encode_texts(processed['bert_texts'], cfg, device=device)

    X_nli = build_nli_features(processed['bert_texts'], cfg, device=device)

    X = np.hstack([X_bert, X_topics, processed['meta'], X_nli])
    print(f'Feature matrix: {X.shape}')

    # ── 載入模型並預測 ────────────────────────────────────────────────────────
    xgb = XGBClassifier()
    xgb.load_model(str(models_dir / 'xgb.json'))
    preds = xgb.predict(X)
    print(f'Predictions: {preds.sum()} positive / {(preds == 0).sum()} negative')

    # ── 輸出 submission ───────────────────────────────────────────────────────
    out_path = ROOT / 'results' / f'submission_{exp_id}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row_id': test_df['row_id'], 'LABEL': preds}).to_csv(out_path, index=False)
    print(f'✅ Submission saved: {out_path}')


if __name__ == '__main__':
    main()
