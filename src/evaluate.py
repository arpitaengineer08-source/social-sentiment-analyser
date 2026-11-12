"""Plotting helpers for model evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (works inside VS Code, servers, Streamlit)
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], path: Path) -> None:
    names = list(results)
    cv = [results[n]["cv_f1_macro"] for n in names]
    test = [results[n]["test_f1_macro"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(x - 0.2, cv, 0.4, label="Cross-val F1 (macro)", color="#4C78A8")
    ax.bar(x + 0.2, test, 0.4, label="Test F1 (macro)", color="#F58518")
    ax.set_xticks(x, labels=names)
    ax.set_ylim(min(min(cv), min(test)) - 0.1, 1.02)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model Comparison", fontweight="bold")
    for i, (a, b) in enumerate(zip(cv, test)):
        ax.text(i - 0.2, a + 0.004, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, b + 0.004, f"{b:.3f}", ha="center", fontsize=9)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
