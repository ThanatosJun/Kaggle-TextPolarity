import random
import re
from typing import List

import nltk
from nltk.corpus import wordnet

# Negation words should never be replaced or swapped away — flipping them changes the label
_NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', "n't",
    'nothing', 'nobody', 'nowhere', 'none',
}

_POS_MAP = {
    'NN': wordnet.NOUN,
    'NNS': wordnet.NOUN,
    'VB': wordnet.VERB,
    'VBD': wordnet.VERB,
    'VBG': wordnet.VERB,
    'VBN': wordnet.VERB,
    'VBP': wordnet.VERB,
    'VBZ': wordnet.VERB,
    'JJ': wordnet.ADJ,
    'JJR': wordnet.ADJ,
    'JJS': wordnet.ADJ,
    'RB': wordnet.ADV,
    'RBR': wordnet.ADV,
    'RBS': wordnet.ADV,
}


def _get_synonym(word: str, pos_tag: str) -> str:
    """Return a random synonym from WordNet, or the original word if none found."""
    wn_pos = _POS_MAP.get(pos_tag)
    if wn_pos is None:
        return word
    synsets = wordnet.synsets(word, pos=wn_pos)
    candidates = []
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace('_', ' ')
            if candidate.lower() != word.lower():
                candidates.append(candidate)
    return random.choice(candidates) if candidates else word


def synonym_replacement(text: str, n: int = 2) -> str:
    """Replace up to n non-negation words with WordNet synonyms."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    replaceable = [
        i for i, (w, _) in enumerate(tagged)
        if w.lower() not in _NEGATION_WORDS and re.search(r'[a-zA-Z]', w)
    ]
    if not replaceable:
        return text

    random.shuffle(replaceable)
    for idx in replaceable[:n]:
        word, tag = tagged[idx]
        tagged[idx] = (_get_synonym(word, tag), tag)

    return ' '.join(w for w, _ in tagged)


def random_swap(text: str, n: int = 2) -> str:
    """Randomly swap n pairs of non-negation word positions."""
    tokens = nltk.word_tokenize(text)
    movable = [
        i for i, w in enumerate(tokens)
        if w.lower() not in _NEGATION_WORDS and re.search(r'[a-zA-Z]', w)
    ]
    if len(movable) < 2:
        return text

    for _ in range(n):
        i, j = random.sample(movable, 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]

    return ' '.join(tokens)


def random_deletion(text: str, p: float = 0.15) -> str:
    """Randomly delete each non-negation word with probability p."""
    tokens = nltk.word_tokenize(text)
    if len(tokens) <= 1:
        return text

    result = [
        w for w in tokens
        if w.lower() in _NEGATION_WORDS
        or not re.search(r'[a-zA-Z]', w)
        or random.random() > p
    ]
    return ' '.join(result) if result else tokens[0]
