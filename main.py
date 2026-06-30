"""
main.py
End-to-end orchestrator for the SML Text Classification project.

Run order:
  Part A  — Preprocessing + Sparse Features
  Part B  — BERT Embeddings + GloVe Embeddings (both cached)
  Part C  — Model Training & Evaluation
  Part D  — Dimensionality Reduction & Plots

Usage:
  python main.py

GloVe setup (one-time):
  1. Download https://nlp.stanford.edu/data/glove.6B.zip
  2. Unzip and place glove.6B.100d.txt in the project root
     (or update GLOVE_PATH / GLOVE_DIM in features.py)
"""

import os
import pickle

import numpy as np
import scipy.sparse as sp

# ─── Project modules ──────────────────────────────────────────────────────────
from preprocessing import load_data, preprocess_corpus
from features      import (build_bow, build_tfidf,
                            get_or_cache_embeddings,
                            get_or_cache_glove)
from models        import (train_logistic_regression, train_linear_svm,
                           train_knn, train_kmeans)
from evaluation    import (compute_metrics, plot_model_comparison,
                           plot_explained_variance, plot_acc_vs_components)
from pca           import reduce_tfidf, reduce_embeddings, sweep_components
from sklearn.decomposition import PCA

# ─── Directories ──────────────────────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"
PLOTS_DIR    = "plots"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,    exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Preprocessing + Sparse Features
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART A — Sparse Features")
print("=" * 60)

X_train_raw, y_train, X_test_raw, y_test = load_data()

print("\nPreprocessing …")
X_train_clean = preprocess_corpus(X_train_raw)
X_test_clean  = preprocess_corpus(X_test_raw)

print("\nSample before:", X_train_raw[0][:80])
print("Sample after :", X_train_clean[0][:80])

print("\nBuilding BoW …")
bow_vec, X_train_bow, X_test_bow = build_bow(X_train_clean, X_test_clean)

print("\nBuilding TF-IDF …")
tfidf_vec, X_train_tfidf, X_test_tfidf = build_tfidf(X_train_clean, X_test_clean)

# Persist sparse matrices + labels
sp.save_npz(os.path.join(ARTIFACT_DIR, "X_train_bow.npz"),   X_train_bow)
sp.save_npz(os.path.join(ARTIFACT_DIR, "X_test_bow.npz"),    X_test_bow)
sp.save_npz(os.path.join(ARTIFACT_DIR, "X_train_tfidf.npz"), X_train_tfidf)
sp.save_npz(os.path.join(ARTIFACT_DIR, "X_test_tfidf.npz"),  X_test_tfidf)
with open(os.path.join(ARTIFACT_DIR, "labels.pkl"), "wb") as f:
    pickle.dump({"y_train": y_train, "y_test": y_test}, f)
with open(os.path.join(ARTIFACT_DIR, "raw_texts.pkl"), "wb") as f:
    pickle.dump({"X_train": X_train_raw, "X_test": X_test_raw}, f)

print("\nPart A complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — BERT Embeddings + GloVe Embeddings
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART B — Dense Embeddings (BERT + GloVe)")
print("=" * 60)

# ── BERT ──────────────────────────────────────────────────────────────────────
print("\n── DistilBERT ──")
X_train_embed, X_test_embed = get_or_cache_embeddings(
    X_train_raw, X_test_raw, cache_dir=ARTIFACT_DIR
)
print(f"BERT shapes  — train: {X_train_embed.shape}, test: {X_test_embed.shape}")

# ── GloVe ─────────────────────────────────────────────────────────────────────
# Preprocessed (cleaned) texts are used so stopwords are already removed,
# matching the same vocabulary as the sparse features.
print("\n── GloVe ──")
X_train_glove, X_test_glove = get_or_cache_glove(
    X_train_clean, X_test_clean, cache_dir=ARTIFACT_DIR
)
print(f"GloVe shapes — train: {X_train_glove.shape}, test: {X_test_glove.shape}")

print("\nPart B complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Model Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART C — Model Training")
print("=" * 60)

y_train_embed = y_train[:len(X_train_embed)]
y_test_embed  = y_test[:len(X_test_embed)]
y_train_glove = y_train[:len(X_train_glove)]
y_test_glove  = y_test[:len(X_test_glove)]

all_results = []

def _run(name, X_tr, y_tr, X_te, y_te, fn, **kw):
    _, y_pred = fn(X_tr, y_tr, X_te, **kw)
    metrics   = compute_metrics(name, y_te, y_pred)
    all_results.append(metrics)
    return metrics

# ── TF-IDF (Sparse) ───────────────────────────────────────────────────────────
_run("LR | TF-IDF",     X_train_tfidf, y_train, X_test_tfidf, y_test, train_logistic_regression)
_run("SVM | TF-IDF",    X_train_tfidf, y_train, X_test_tfidf, y_test, train_linear_svm)
_run("KNN | TF-IDF",    X_train_tfidf, y_train, X_test_tfidf, y_test, train_knn)
_run("KMeans | TF-IDF", X_train_tfidf, y_train, X_test_tfidf, y_test, train_kmeans)

# ── BERT Embeddings (Dense) ───────────────────────────────────────────────────
_run("LR | BERT",       X_train_embed, y_train_embed, X_test_embed, y_test_embed, train_logistic_regression)
_run("SVM | BERT",      X_train_embed, y_train_embed, X_test_embed, y_test_embed, train_linear_svm)
_run("KNN | BERT",      X_train_embed, y_train_embed, X_test_embed, y_test_embed, train_knn)
_run("KMeans | BERT",   X_train_embed, y_train_embed, X_test_embed, y_test_embed, train_kmeans)

# ── GloVe Embeddings (Dense) ──────────────────────────────────────────────────
_run("LR | GloVe",      X_train_glove, y_train_glove, X_test_glove, y_test_glove, train_logistic_regression)
_run("SVM | GloVe",     X_train_glove, y_train_glove, X_test_glove, y_test_glove, train_linear_svm)
_run("KNN | GloVe",     X_train_glove, y_train_glove, X_test_glove, y_test_glove, train_knn)
_run("KMeans | GloVe",  X_train_glove, y_train_glove, X_test_glove, y_test_glove, train_kmeans)

with open(os.path.join(ARTIFACT_DIR, "results.pkl"), "wb") as f:
    pickle.dump(all_results, f)

print("\n====== Summary ======")
for r in all_results:
    print(f"  {r['name']:22s}  acc={r['accuracy']:.4f}  f1={r['macro_f1']:.4f}")

print("\nPart C complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Dimensionality Reduction & Plots
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART D — Dimensionality Reduction")
print("=" * 60)

# Reduce
print("\n── Reducing TF-IDF with TruncatedSVD ──")
svd, X_tr_svd, X_te_svd, evr_svd = reduce_tfidf(X_train_tfidf, X_test_tfidf)

print("\n── Reducing BERT embeddings with PCA ──")
pca, X_tr_pca, X_te_pca, evr_pca = reduce_embeddings(X_train_embed, X_test_embed)

# Accuracy vs n_components sweep
comp_list = [10, 25, 50, 100, 200]

print("\n── Sweep: TF-IDF / PCA ──")
_, acc_svd = sweep_components(
    X_train_tfidf.toarray(), X_test_tfidf.toarray(),
    y_train, y_test, PCA, "SVD-TF-IDF", comp_list
)

print("\n── Sweep: BERT / PCA ──")
_, acc_pca = sweep_components(
    X_train_embed, X_test_embed,
    y_train_embed, y_test_embed, PCA, "PCA-BERT", comp_list
)

print("\n── Sweep: GloVe / PCA ──")
_, acc_glove = sweep_components(
    X_train_glove, X_test_glove,
    y_train_glove, y_test_glove, PCA, "PCA-GloVe", comp_list
)

# Plots
print("\n── Generating Plots ──")
plot_explained_variance(evr_svd, evr_pca)
plot_model_comparison(all_results)
plot_acc_vs_components(comp_list, acc_svd, acc_pca)

print("\nPart D complete — plots saved to ./plots/")
print("\n✓ All parts complete.")
