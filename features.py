"""
features.py
Sparse (BoW / TF-IDF) and dense (DistilBERT, GloVe) feature extraction.
"""

import os
import numpy as np
import torch
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
BATCH_SIZE   = 32
MAX_LENGTH   = 128
ARTIFACT_DIR = "artifacts"

# GloVe: download from https://nlp.stanford.edu/data/glove.6B.zip
# then point this path to glove.6B.100d.txt (or any other variant)
GLOVE_PATH   = "glove.6B.100d.txt"
GLOVE_DIM    = 100  # must match the file above

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Sparse Features
# ══════════════════════════════════════════════════════════════════════════════

def build_bow(X_train_clean: list[str], X_test_clean: list[str],
              max_features: int = 5000):
    """Unigram Bag-of-Words (CountVectorizer)."""
    vectorizer    = CountVectorizer(ngram_range=(1, 1), max_features=max_features)
    X_train_bow   = vectorizer.fit_transform(X_train_clean)
    X_test_bow    = vectorizer.transform(X_test_clean)
    print(f"BoW  — train: {X_train_bow.shape}, test: {X_test_bow.shape}")
    return vectorizer, X_train_bow, X_test_bow


def build_tfidf(X_train_clean: list[str], X_test_clean: list[str],
                max_features: int = 5000):
    """Bigram TF-IDF."""
    vectorizer     = TfidfVectorizer(ngram_range=(2, 2), max_features=max_features)
    X_train_tfidf  = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf   = vectorizer.transform(X_test_clean)
    print(f"TF-IDF — train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")
    return vectorizer, X_train_tfidf, X_test_tfidf


# ══════════════════════════════════════════════════════════════════════════════
# Dense Embeddings — GloVe (mean-pooled word vectors)
# ══════════════════════════════════════════════════════════════════════════════

def load_glove_vectors(glove_path: str = GLOVE_PATH) -> dict[str, np.ndarray]:
    """
    Parse a GloVe .txt file into a word → vector dict.
    Download GloVe from: https://nlp.stanford.edu/data/glove.6B.zip
    Recommended file: glove.6B.100d.txt  (400 k words, 100-d vectors, ~347 MB)
    """
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"GloVe file not found at '{glove_path}'.\n"
            "Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip, "
            "unzip it, and set GLOVE_PATH in features.py to the correct path."
        )
    vectors: dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    print(f"Loaded {len(vectors):,} GloVe vectors (dim={next(iter(vectors.values())).shape[0]})")
    return vectors


def text_to_glove(text: str, glove: dict[str, np.ndarray],
                  dim: int = GLOVE_DIM) -> np.ndarray:
    """
    Mean-pool GloVe vectors for all recognised words in `text`.
    Returns a zero vector if no word is found in the vocabulary.
    """
    vecs = [glove[w] for w in text.split() if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)


def get_glove_embeddings(texts: list[str],
                          glove: dict[str, np.ndarray],
                          dim: int = GLOVE_DIM) -> np.ndarray:
    """Convert a list of (pre-cleaned) texts to a GloVe embedding matrix."""
    return np.vstack([text_to_glove(t, glove, dim) for t in texts])


def get_or_cache_glove(X_train_clean: list[str],
                        X_test_clean:  list[str],
                        glove_path:    str = GLOVE_PATH,
                        cache_dir:     str = ARTIFACT_DIR,
                        dim:           int = GLOVE_DIM
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load GloVe embeddings from cache if present, otherwise compute and save.
    Note: pass *preprocessed* (cleaned) texts so stopwords are already removed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_glove.npy")
    test_path  = os.path.join(cache_dir, "test_glove.npy")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading cached GloVe embeddings …")
        return np.load(train_path), np.load(test_path)

    glove = load_glove_vectors(glove_path)

    print("Computing GloVe train embeddings …")
    X_train_glove = get_glove_embeddings(X_train_clean, glove, dim)
    np.save(train_path, X_train_glove)

    print("Computing GloVe test embeddings …")
    X_test_glove = get_glove_embeddings(X_test_clean, glove, dim)
    np.save(test_path, X_test_glove)

    print(f"GloVe embeddings cached → {cache_dir}")
    return X_train_glove, X_test_glove
# sentence
# → tokenizer
# → DistilBERT
# → last hidden states
# → mean pooling
# → sentence embedding

# ══════════════════════════════════════════════════════════════════════════════
# Dense Embeddings — DistilBERT (frozen feature extractor)
# ══════════════════════════════════════════════════════════════════════════════

def load_bert_model():
    """Load DistilBERT tokenizer + model (all params frozen)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    for param in model.parameters():
        param.requires_grad = False          # feature extractor only
    model.to(device)
    model.eval()
    print(f"Loaded {MODEL_NAME} on {device}")
    return tokenizer, model


def get_embeddings(texts: list[str], tokenizer, model,
                   batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings, batched."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        all_embeddings.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} …")
    return np.vstack(all_embeddings)


def get_or_cache_embeddings(X_train: list[str], X_test: list[str],
                             cache_dir: str = ARTIFACT_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings from cache if present, otherwise compute and save."""
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_embed.npy")
    test_path  = os.path.join(cache_dir, "test_embed.npy")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading cached BERT embeddings …")
        return np.load(train_path), np.load(test_path)

    tokenizer, model = load_bert_model()

    print("Computing train embeddings …")
    X_train_embed = get_embeddings(X_train, tokenizer, model)
    np.save(train_path, X_train_embed)

    print("Computing test embeddings …")
    X_test_embed = get_embeddings(X_test, tokenizer, model)
    np.save(test_path, X_test_embed)

    print(f"BERT embeddings cached → {cache_dir}")
    return X_train_embed, X_test_embed
