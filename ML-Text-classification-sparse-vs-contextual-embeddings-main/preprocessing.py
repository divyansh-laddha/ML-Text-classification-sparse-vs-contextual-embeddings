"""
preprocessing.py
Text cleaning and tokenization utilities.
"""

import re
import nltk
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


def preprocess(text: str) -> str:
    """Lowercase → remove non-alpha → tokenize → stopword removal."""
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)


def preprocess_corpus(texts: list[str]) -> list[str]:
    """Apply preprocess() to an iterable of raw texts."""
    return [preprocess(t) for t in texts]


def load_data():
    """
    Load AG News from HuggingFace and return train/test splits.
    Uses the same 8 000 / 2 000 subset defined in the project spec.
    """
    from datasets import load_dataset

    dataset = load_dataset("ag_news")
    print(dataset)

    X_train = list(dataset['train']['text'])[:8000]
    y_train = list(dataset['train']['label'])[:8000]
    X_test  = list(dataset['test']['text'])[:2000]
    y_test  = list(dataset['test']['label'])[:2000]

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    return X_train, y_train, X_test, y_test
