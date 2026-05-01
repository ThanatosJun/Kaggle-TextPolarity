import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from xgboost import XGBClassifier

from src.data_preprocess import run as preprocess
from src.dimension_decrease import transform
from src.text_transform import build_features


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
    parser = argparse.ArgumentParser(description='Predict with a stacked model.')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='Experiment ID (overrides config experiment.id)')
    args = parser.parse_args()

    ROOT = find_project_root()

    if args.exp_id:
        snapshot = ROOT / 'experiments' / args.exp_id / 'config_snapshot.yaml'
        if not snapshot.exists():
            raise FileNotFoundError(f"No config snapshot for '{args.exp_id}': {snapshot}")
        with open(snapshot, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['experiment']['id'] = args.exp_id
    else:
        with open(ROOT / 'config' / 'config.yaml', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    device     = resolve_device(cfg['global']['device'])
    batch_size = cfg['features']['sentence_transformer']['batch_size']
    exp_id     = cfg['experiment']['id']
    base_cfgs  = cfg['stacking']['base_models']

    print(f'Experiment: {exp_id} | Device: {device}')
    print(f'Base models: {[b["id"] for b in base_cfgs]}')

    # ── 載入測試資料 ──────────────────────────────────────────────────────────
    test_df   = pd.read_csv(ROOT / cfg['paths']['test'])
    print(f'Test: {test_df.shape}')
    proc_test = preprocess(test_df, cfg)

    models_dir = ROOT / cfg['paths']['models_dir'] / exp_id

    # ── 每個 base model：encode → 降維 → predict_proba ───────────────────────
    test_proba_list = []
    for bm in base_cfgs:
        print(f'\n── {bm["id"]}  ({bm["model_name"]}) ──')

        # text_transform
        X_test_raw = build_features(
            proc_test['bert_texts'], proc_test['meta'],
            bm['model_name'], batch_size, device,
        )

        # dimension_decrease：載入 full reducer（若存在）
        reducer_path = models_dir / f'reducer_{bm["id"]}_full.pkl'
        reducer = joblib.load(reducer_path) if reducer_path.exists() else None
        X_test = transform(reducer, X_test_raw)
        if reducer is not None:
            print(f'   Applied reducer: {X_test_raw.shape} → {X_test.shape}')

        # model_train：載入 full XGBoost
        xgb = XGBClassifier()
        xgb.load_model(str(models_dir / f'xgb_{bm["id"]}_full.json'))
        test_proba_list.append(xgb.predict_proba(X_test))

    # ── model_stack：Meta model 預測 ─────────────────────────────────────────
    meta_test_X = np.hstack(test_proba_list)
    meta_model  = joblib.load(models_dir / 'meta_model_full.pkl')
    preds       = meta_model.predict(meta_test_X)
    print(f'\nPredictions: {preds.sum()} positive / {(preds == 0).sum()} negative')

    # ── 輸出 submission ───────────────────────────────────────────────────────
    out_path = ROOT / 'results' / f'submission_{exp_id}.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row_id': test_df['row_id'], 'LABEL': preds}).to_csv(out_path, index=False)
    print(f'✅ Submission saved: {out_path}')


if __name__ == '__main__':
    main()
