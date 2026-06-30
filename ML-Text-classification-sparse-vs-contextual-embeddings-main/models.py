"""
models.py
Model definitions: Logistic Regression, Linear SVM, KNN, KMeans.
Each function accepts (X_train, y_train, X_test, y_test) and returns
the fitted estimator plus predictions.
"""

import numpy as np

from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.cluster       import KMeans


# ─── Supervised Models ────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_test,
                               max_iter: int = 1000):
    """Logistic Regression (lbfgs, multi-class auto)."""
    lr = LogisticRegression(max_iter=max_iter, solver="lbfgs",
                            multi_class="auto")
    lr.fit(X_train, y_train)
    return lr, lr.predict(X_test)


def train_linear_svm(X_train, y_train, X_test):
    """Linear SVM (kernel='linear' — no RBF)."""
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    return svm, svm.predict(X_test)


def train_knn(X_train, y_train, X_test, n_neighbors: int = 5):
    """K-Nearest Neighbours."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn, knn.predict(X_test)


# ─── Unsupervised Baseline ────────────────────────────────────────────────────

def train_kmeans(X_train, y_train, X_test,
                 n_clusters: int = 4):
    """
    KMeans with majority-vote cluster → label mapping.
    Returns (fitted_model, predicted_labels_for_test).
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X_train)

    # Map each cluster to its majority ground-truth label
    cluster_labels = np.zeros(n_clusters, dtype=int)
    train_preds    = km.predict(X_train)
    for c in range(n_clusters):
        mask = train_preds == c
        if mask.sum() > 0:
            cluster_labels[c] = np.bincount(np.array(y_train)[mask]).argmax()

    y_pred = cluster_labels[km.predict(X_test)]
    return km, y_pred
