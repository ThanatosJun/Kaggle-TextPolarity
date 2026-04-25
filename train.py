import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.experiment import log_to_csv, save_metrics, snapshot_config
from src.features import (
    build_nli_features, build_topic_features,
    encode_texts, transform_topic_features,
)
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


def build_xgb(cfg: dict) -> XGBClassifier:
    c = cfg['models']['xgboost']
    return XGBClassifier(
        n_estimators     = c['n_estimators'],
        learning_rate    = c['learning_rate'],
        max_depth        = c['max_depth'],
        random_state     = c['random_state'],
        eval_metric      = c['eval_metric'],
        device           = c['device'],
        subsample        = c['subsample'],
        colsample_bytree = c['colsample_bytree'],
        reg_alpha        = c['reg_alpha'],
        reg_lambda       = c['reg_lambda'],
    )


def print_metrics(y_true, y_pred, label: str = '') -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f'\n[ {label} ]')
    print(f'  {"Accuracy":<12} {acc:.4f}')
    print(f'  {"Precision":<12} {prec:.4f}')
    print(f'  {"Recall":<12} {rec:.4f}')
    print(f'  {"F1":<12} {f1:.4f}')
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    return {'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4)}


def extract_features(proc: dict, cfg: dict, device: str,
                     count_vec=None, lda=None, fit: bool = False) -> np.ndarray:
    """
    統一特徵提取入口。
    fit=True  → 訓練 CountVec/LDA（train split 與 full data 使用）
    fit=False → transform only（val / test 使用）
    回傳 (X, count_vec, lda)，後兩者在 fit=False 時維持傳入值。
    """
    # LDA / Topic features
    if fit:
        topics    = build_topic_features(proc['tfidf_texts'], cfg)
        count_vec = topics['count_vec']
        lda       = topics['lda']
        X_topics  = topics['X_train_topics']
    else:
        X_topics = transform_topic_features(count_vec, lda, proc['tfidf_texts'], cfg)

    # BERT embeddings
    X_bert = encode_texts(proc['bert_texts'], cfg, device=device)

    # NLI domain features（stateless，train/val/test 都直接 inference）
    X_nli = build_nli_features(proc['bert_texts'], cfg, device=device)

    X = np.hstack([X_bert, X_topics, proc['meta'], X_nli])
    return X, count_vec, lda


def main() -> None:
    ROOT = find_project_root()
    config_path = ROOT / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device    = resolve_device(cfg['global']['device'])
    seed      = cfg['global']['random_seed']
    val_ratio = cfg['evaluation']['val_ratio']

    lda_on = cfg['features']['lda']['enabled']
    nli_on = cfg['features']['nli']['enabled']
    print(f'Device: {device} | Val ratio: {val_ratio}')
    print(f'Features — LDA: {lda_on} | NLI: {nli_on}')

    # ── 1. 載入資料 ───────────────────────────────────────────────────────────
    full_df = pd.read_csv(ROOT / cfg['paths']['train'])
    print(f'Full train set: {full_df.shape}')

    # ── 2. Train / Val split ──────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        full_df, test_size=val_ratio, stratify=full_df['LABEL'], random_state=seed,
    )
    print(f'Train split: {train_df.shape}  |  Val split: {val_df.shape}')

    # ── 3. 前處理 ─────────────────────────────────────────────────────────────
    print('\n>> Preprocessing...')
    proc_train = preprocess(train_df, cfg)
    proc_val   = preprocess(val_df,   cfg)

    # ── 4. 特徵工程 ──────────────────────────────────────────────────────────
    print('>> Building train features (fit)...')
    X_train, count_vec, lda = extract_features(proc_train, cfg, device, fit=True)

    print('>> Building val features (transform only)...')
    X_val, _, _ = extract_features(proc_val, cfg, device, count_vec=count_vec, lda=lda, fit=False)

    y_train = train_df['LABEL'].values
    y_val   = val_df['LABEL'].values
    print(f'Feature matrix — Train: {X_train.shape}  |  Val: {X_val.shape}')

    # ── 5. 訓練模型（train split）────────────────────────────────────────────
    print('\n>> Training on train split...')
    xgb = build_xgb(cfg)
    xgb.fit(X_train, y_train)

    # ── 6. 評估 ───────────────────────────────────────────────────────────────
    train_metrics = print_metrics(y_train, xgb.predict(X_train), 'Train split')
    val_metrics   = print_metrics(y_val,   xgb.predict(X_val),   'Val split  ')

    # ── 7. 儲存 experiment metrics ────────────────────────────────────────────
    exp     = cfg['experiment']
    exp_dir = ROOT / 'experiments' / exp['id']
    save_metrics(exp_dir, {
        'experiment_id': exp['id'],
        'val_ratio':      val_ratio,
        'train_metrics':  train_metrics,
        'val_metrics':    val_metrics,
        'kaggle_score':   None,
        'note':           '',
    })
    snapshot_config(exp_dir, config_path)
    log_to_csv(
        experiments_dir = ROOT / 'experiments',
        exp_id          = exp['id'],
        exp_name        = exp['name'],
        train_metrics   = train_metrics,
        val_metrics     = val_metrics,
    )

    # ── 8. 最終模型：全量資料重新 fit + 訓練 ─────────────────────────────────
    print('\n>> Retraining on full dataset for final model...')
    proc_full = preprocess(full_df, cfg)
    X_full, topics_count_vec, topics_lda = extract_features(proc_full, cfg, device, fit=True)

    xgb_final = build_xgb(cfg)
    xgb_final.fit(X_full, full_df['LABEL'].values)

    # ── 9. 儲存 artifacts ─────────────────────────────────────────────────────
    models_dir = ROOT / cfg['paths']['models_dir'] / exp['id']
    models_dir.mkdir(parents=True, exist_ok=True)
    if lda_on:
        joblib.dump(topics_count_vec, models_dir / 'count_vec.pkl')
        joblib.dump(topics_lda,       models_dir / 'lda.pkl')
        print('   count_vec.pkl + lda.pkl saved')
    xgb_final.save_model(str(models_dir / 'xgb.json'))
    print(f'✅ Artifacts saved: {models_dir}')


if __name__ == '__main__':
    main()
