"""
pca.py
Dimensionality reduction:
  - TruncatedSVD on sparse TF-IDF features
  - PCA on dense BERT embeddings
Also provides a component-sweep utility used by main.py.
"""

import numpy as np

from sklearn.decomposition  import TruncatedSVD, PCA
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import accuracy_score


# ─── Reducers ─────────────────────────────────────────────────────────────────

def reduce_tfidf(X_train, X_test, n_components: int = 100):
    """TruncatedSVD on sparse TF-IDF (PCA equivalent for sparse matrices)."""
    svd      = TruncatedSVD(n_components=n_components, random_state=42)
    X_tr_svd = svd.fit_transform(X_train)
    X_te_svd = svd.transform(X_test)
    cum_var  = svd.explained_variance_ratio_.cumsum()[-1]
    print(f"TruncatedSVD ({n_components}d) — cumulative explained variance: {cum_var:.4f}")
    return svd, X_tr_svd, X_te_svd, svd.explained_variance_ratio_


def reduce_embeddings(X_train: np.ndarray, X_test: np.ndarray,
                      n_components: int = 100):
    """PCA on dense BERT embeddings."""
    pca      = PCA(n_components=n_components, random_state=42)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    cum_var  = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA ({n_components}d) — cumulative explained variance: {cum_var:.4f}")
    return pca, X_tr_pca, X_te_pca, pca.explained_variance_ratio_


# ─── Accuracy vs n_components Sweep ──────────────────────────────────────────

def sweep_components(X_train, X_test, y_train, y_test,
                     reducer_cls,
                     label: str,
                     components_list: list[int] = (10, 25, 50, 100, 200)
                     ) -> tuple[list[int], list[float]]:
    """
    For each value in components_list, reduce then train/eval a LR classifier.
    Returns (components_list, accuracy_list).
    """
    accs = []
    for n in components_list:
        n       = min(n, min(X_train.shape))
        reducer = reducer_cls(n_components=n, random_state=42)
        X_tr    = reducer.fit_transform(X_train)
        X_te    = reducer.transform(X_test)
        lr      = LogisticRegression(max_iter=500, solver="lbfgs",
                                     multi_class="auto")
        lr.fit(X_tr, y_train[:len(X_tr)])
        acc = accuracy_score(y_test[:len(X_te)], lr.predict(X_te))
        accs.append(acc)
        print(f"  {label}  n={n:4d} → acc={acc:.4f}")
    return list(components_list), accs
