"""
NLPaug contextual 資料擴增腳本

從 datasets/train_2022.csv 生成 datasets/train_2022_augmented.csv。
每筆原始樣本產生一筆 contextual augmentation（BERT masked prediction），
最終資料集為原本的 2 倍（~4000 筆），以 'source' 欄位標明來源。

用法：
    python augment_data.py
"""

import sys
import torch
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, '.')
from src.augment import build_augmenter, contextual_augment


def load_cfg(path: str = 'config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    v = cfg['global']['device']
    if v == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return v


def augment_df(df: pd.DataFrame, augmenter, seed: int = 42) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Augmenting'):
        text = str(row['TEXT'])
        label = row['LABEL']
        rid = row.get('row_id', _)

        rows.append({
            'row_id': rid,
            'TEXT':   text,
            'LABEL':  label,
            'source': 'original',
        })
        rows.append({
            'row_id': f'{rid}_nlpaug',
            'TEXT':   contextual_augment(text, augmenter),
            'LABEL':  label,
            'source': 'nlpaug',
        })

    return pd.DataFrame(rows)


def main():
    cfg    = load_cfg()
    seed   = cfg['global']['random_seed']
    device = resolve_device(cfg)

    train_path = cfg['paths']['train']
    out_path   = train_path.replace('train_2022.csv', 'train_2022_augmented.csv')

    print(f'讀取原始資料：{train_path}')
    df = pd.read_csv(train_path)
    print(f'原始筆數：{len(df)} | Device: {device}')

    print('\n>> 建立 NLPaug augmenter（bert-base-uncased）...')
    augmenter = build_augmenter(model_path='bert-base-uncased', aug_p=0.1, device=device)

    aug_df = augment_df(df, augmenter, seed=seed)
    aug_df.to_csv(out_path, index=False)

    orig = (aug_df['source'] == 'original').sum()
    aug  = (aug_df['source'] == 'nlpaug').sum()
    print(f'\n擴增後筆數：{len(aug_df)}')
    print(f'  original : {orig}')
    print(f'  nlpaug   : {aug}')
    print(f'\n已儲存至：{out_path}')


if __name__ == '__main__':
    main()
