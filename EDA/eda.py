import pandas as pd
import numpy as np

# Load datasets
train = pd.read_csv('datasets/train_2022.csv')
test = pd.read_csv('datasets/test_2022.csv')
sample_sub = pd.read_csv('datasets/sample_submission.csv')

print("=== 基本資訊 (Shape Info) ===")
print(f"Train set: {train.shape}")
print(f"Test set:  {test.shape}")
print(f"Sample Submission: {sample_sub.shape}")

print("\n=== 缺失值 (Missing Values) ===")
print("Train:")
print(train.isnull().sum())
print("\nTest:")
print(test.isnull().sum())

print("\n=== 標籤分佈 (Label Distribution) ===")
if 'LABEL' in train.columns:
    print(train['LABEL'].value_counts())
    print(train['LABEL'].value_counts(normalize=True))

print("\n=== 文本長度統計 (Text Length Statistics) ===")
train['text_len'] = train['TEXT'].astype(str).apply(len)
train['word_count'] = train['TEXT'].astype(str).apply(lambda x: len(x.split()))
test['text_len'] = test['TEXT'].astype(str).apply(len)
test['word_count'] = test['TEXT'].astype(str).apply(lambda x: len(x.split()))

print("Train Word Count Stats:")
print(train['word_count'].describe())
print("\nTest Word Count Stats:")
print(test['word_count'].describe())

print("\n=== 重複值分析 (Duplicates Analysis) ===")
print(f"Train TEXT 重複數: {train.duplicated(subset=['TEXT']).sum()}")
print(f"Test TEXT 重複數: {test.duplicated(subset=['TEXT']).sum()}")

train_texts = set(train['TEXT'].dropna())
test_texts = set(test['TEXT'].dropna())
overlap = len(train_texts.intersection(test_texts))
print(f"Train 與 Test 之間重複的文本數量: {overlap}")

print("\n=== 潛在異常或特徵字串 (Potential Issues) ===")
print(f"空白或僅含空格的文本 (Train): {(train['TEXT'].astype(str).str.strip() == '').sum()}")
print(f"含有 'num_num' 的文本數量 (Train): {train['TEXT'].astype(str).str.contains('num_num').sum()}")
print(f"含有 'num_extend' 的文本數量 (Train): {train['TEXT'].astype(str).str.contains('num_extend').sum()}")
