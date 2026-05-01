import random
from transformers import pipeline


def build_augmenter(model_path: str = 'bert-base-uncased',
                    aug_p: float = 0.1,
                    device: str = 'cpu') -> dict:
    """Build a fill-mask augmenter using transformers pipeline."""
    device_id = 0 if device == 'cuda' else -1
    filler = pipeline('fill-mask', model=model_path, device=device_id)
    return {'pipeline': filler, 'aug_p': aug_p}


def contextual_augment(text: str, augmenter: dict) -> str:
    """Replace ~aug_p fraction of words with BERT fill-mask predictions."""
    filler  = augmenter['pipeline']
    aug_p   = augmenter['aug_p']

    words  = text.split()
    n_mask = max(1, round(len(words) * aug_p))
    indices = random.sample(range(len(words)), min(n_mask, len(words)))

    result = words.copy()
    for idx in indices:
        candidate = result.copy()
        candidate[idx] = '[MASK]'
        masked_text = ' '.join(candidate[:400])   # BERT max ~512 tokens
        try:
            preds = filler(masked_text)
            token = preds[0]['token_str'].strip()
            # skip subword pieces (e.g. "##ing")
            if token and not token.startswith('##'):
                result[idx] = token
        except Exception:
            pass   # keep original word on failure

    return ' '.join(result)
