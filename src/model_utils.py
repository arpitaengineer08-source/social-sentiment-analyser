"""Helpers shared by training and inference."""
from __future__ import annotations

import numpy as np


def class_weights(clf) -> np.ndarray | None:
    """Per-class feature weights (n_classes x n_features) for explainability.

    Works for linear models (coef_) and Naive Bayes (centred log-probabilities).
    """
    if hasattr(clf, "coef_") and clf.coef_.shape[0] == len(clf.classes_):
        return np.asarray(clf.coef_)
    if hasattr(clf, "feature_log_prob_"):
        flp = np.asarray(clf.feature_log_prob_)
        return flp - flp.mean(axis=0, keepdims=True)
    return None
