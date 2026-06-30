"""
evaluation.py
Metrics computation and comparison visualisations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report


PLOTS_DIR = "plots"


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(name: str, y_true, y_pred) -> dict:
    """Print and return accuracy + macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n[{name}]  Accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"name": name, "accuracy": acc, "macro_f1": f1}


# ─── Plots ────────────────────────────────────────────────────────────────────

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_model_comparison(results: list[dict]):
    """Grouped bar chart: accuracy & macro-F1 per model × feature space."""
    _ensure_plots_dir()
    labels = [r["name"] for r in results]
    accs   = [r["accuracy"]  for r in results]
    f1s    = [r["macro_f1"]  for r in results]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy",  color="steelblue",  alpha=0.85)
    ax.bar(x + w / 2, f1s,  w, label="Macro-F1",  color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model × Feature Space Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_explained_variance(evr_svd: np.ndarray, evr_pca: np.ndarray):
    """Cumulative explained variance curves for SVD (TF-IDF) and PCA (BERT)."""
    _ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.cumsum(evr_svd), color="steelblue")
    axes[0].set_title("TruncatedSVD — Cumulative Explained Variance (TF-IDF)")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.cumsum(evr_pca), color="darkorange")
    axes[1].set_title("PCA — Cumulative Explained Variance (BERT)")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "explained_variance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_acc_vs_components(components: list[int],
                            acc_svd:    list[float],
                            acc_pca:    list[float]):
    """LR accuracy as a function of the number of reduced components."""
    _ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components, acc_svd, marker="o", label="TruncatedSVD (TF-IDF)", color="steelblue")
    ax.plot(components[:len(acc_pca)], acc_pca, marker="s", label="PCA (BERT)",  color="darkorange")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("LR Accuracy")
    ax.set_title("Accuracy vs Number of Reduced Components")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "acc_vs_components.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")
