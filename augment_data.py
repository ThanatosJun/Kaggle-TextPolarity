"""
EDA 資料擴增腳本

從 datasets/train_2022.csv 生成 datasets/train_2022_augmented.csv。
每筆原始樣本會額外產生三筆增強樣本（synonym / swap / delete），
最終資料集約為原本的 4 倍（~8000 筆），以 'source' 欄位標明來源。

用法：
    python augment_data.py
"""

import random
import sys

import nltk
import pandas as pd
import yaml

sys.path.insert(0, '.')
from src.augment import synonym_replacement, random_swap, random_deletion

# 確保 NLTK 資源存在
for resource in ('wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                 'punkt', 'punkt_tab'):
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource
                       else f'taggers/{resource}' if 'tagger' in resource
                       else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


def load_cfg(path: str = 'config/config.yaml') -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def augment_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _, row in df.iterrows():
        text = str(row['TEXT'])
        label = row['LABEL']
        original_id = row.get('row_id', _)

        rows.append({
            'ID':     original_id,
            'TEXT':   text,
            'LABEL':  label,
            'source': 'original',
        })
        rows.append({
            'ID':     f'{original_id}_eda_syn',
            'TEXT':   synonym_replacement(text, n=2),
            'LABEL':  label,
            'source': 'eda_synonym',
        })
        rows.append({
            'ID':     f'{original_id}_eda_swap',
            'TEXT':   random_swap(text, n=2),
            'LABEL':  label,
            'source': 'eda_swap',
        })
        rows.append({
            'ID':     f'{original_id}_eda_del',
            'TEXT':   random_deletion(text, p=0.15),
            'LABEL':  label,
            'source': 'eda_delete',
        })

    return pd.DataFrame(rows)


def main():
    cfg = load_cfg()
    seed = cfg['global']['random_seed']

    train_path = cfg['paths']['train']
    out_path = train_path.replace('train_2022.csv', 'train_2022_augmented.csv')

    print(f"讀取原始資料：{train_path}")
    df = pd.read_csv(train_path)
    print(f"原始筆數：{len(df)}")

    aug_df = augment_df(df, seed=seed)
    aug_df.to_csv(out_path, index=False)

    orig = (aug_df['source'] == 'original').sum()
    print(f"\n擴增後筆數：{len(aug_df)}")
    print(f"  original    : {orig}")
    for src in ('eda_synonym', 'eda_swap', 'eda_delete'):
        print(f"  {src:<14}: {(aug_df['source'] == src).sum()}")
    print(f"\n已儲存至：{out_path}")


if __name__ == '__main__':
    main()
