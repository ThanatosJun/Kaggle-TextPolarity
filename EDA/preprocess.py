import pandas as pd
import re

def clean_for_tfidf(text):
    """
    Text preprocessing for Traditional ML (TF-IDF).
    Goals: Lowercase, remove punctuations (keeping logic for ?, !), remove placeholders.
    """
    text = str(text).lower()
    
    # 1. 移除特定的字眼 (e.g., -lrb-, -rrb-)
    text = re.sub(r'-lrb-', '', text)
    text = re.sub(r'-rrb-', '', text)
    
    # 2. 處理特定的 placeholder，如果認為沒用也可以移除
    text = re.sub(r'num_num', '', text)
    text = re.sub(r'num_extend', '', text)
    
    # 3. 把標點符號清除 (只保留小寫字母、數字與空白)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. 把多餘的空白換成單一空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_for_llm(text):
    """
    Text preprocessing for LLMs (BERT, RoBERTa).
    Goals: Restore structures, keep punctuations, fix tokenization issues.
    """
    text = str(text)
    
    # 1. 還原括號
    text = text.replace('-lrb-', '(').replace('-rrb-', ')')
    
    # 2. 修復被切開的否定詞或特殊符號 (如 does n't -> doesn't)
    text = re.sub(r'\s+n\'t', "n't", text)
    text = re.sub(r'\s+\'s', "'s", text)
    text = re.sub(r'\s+\'re', "'re", text)
    text = re.sub(r'\s+\'ll', "'ll", text)
    text = re.sub(r'\s+\'ve', "'ve", text)
    text = re.sub(r'\s+\'m', "'m", text)
    text = re.sub(r'\s+\'d', "'d", text)
    
    # 3. 修復與標點符號間的多餘空白 (e.g., "word ." -> "word.")
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # 4. 保留 num_num / num_extend (供後續加入 special_tokens 使用)
    
    # 5. 清理多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_meta_features(df, text_col='TEXT'):
    """
    Extract meta features (punctuation counts, word counts) before deep cleaning.
    """
    df = df.copy()
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df['exclamation_count'] = df[text_col].astype(str).apply(lambda x: x.count('!'))
    df['question_count'] = df[text_col].astype(str).apply(lambda x: x.count('?'))
    return df

if __name__ == "__main__":
    # 測試腳本
    sample_texts = [
        "director dirk shafer and co-writer greg hinton ride the dubious divide where gay porn reaches for serious drama .",
        "the price was good ,  and came quickly though my prime membership .",
        "i was looking forward to this game for a couple of num_extend",
        "-lrb- wendigo is -rrb- why we go to the cinema : to be fed through the eye , the heart , the mind .",
        "it does n't feel like one !"
    ]
    
    df = pd.DataFrame({'TEXT': sample_texts})
    df = extract_meta_features(df)
    df['TEXT_tfidf'] = df['TEXT'].apply(clean_for_tfidf)
    df['TEXT_llm'] = df['TEXT'].apply(clean_for_llm)
    
    for i, row in df.iterrows():
        print(f"--- 行 {i} ---")
        print(f"[Original] {row['TEXT']}")
        print(f"[TF-IDF]   {row['TEXT_tfidf']}")
        print(f"[LLM]      {row['TEXT_llm']}")
        print()