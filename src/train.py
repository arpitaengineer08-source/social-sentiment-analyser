"""Train, tune, compare and persist sentiment models.

Usage:
    python -m src.train
    python -m src.train --data tweets.csv --text-col text --label-col sentiment
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.data import load_dataset
from src.model_utils import class_weights
from src.evaluate import plot_confusion_matrix, plot_model_comparison
from src.preprocessing import clean_text


def _pipeline(clf) -> Pipeline:
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "Logistic Regression": (
            _pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"clf__C": [1, 5, 20]},
        ),
        "Linear SVM": (
            _pipeline(LinearSVC(class_weight="balanced", dual=True)),
            {"clf__C": [0.3, 1, 3]},
        ),
        "Complement NB": (
            _pipeline(ComplementNB()),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
    }


def top_features(model: Pipeline, k: int = 15) -> dict[str, list[list]]:
    """Most indicative n-grams per class (linear models only)."""
    clf = model.named_steps["clf"]
    weights = class_weights(clf)
    if weights is None:
        return {}
    names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    out = {}
    for idx, label in enumerate(clf.classes_):
        top = np.argsort(weights[idx])[::-1][:k]
        out[label] = [[names[i], round(float(weights[idx][i]), 3)] for i in top]
    return out


def main(data: str | None = None, text_col: str = "text", label_col: str = "sentiment") -> dict:
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)

    df = load_dataset(data, text_col, label_col)
    print(f"Loaded {len(df):,} posts | class balance:\n{df['sentiment'].value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=config.TEST_SIZE,
        stratify=df["sentiment"], random_state=config.RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, (pipe, grid) in candidate_models().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        preds = search.predict(X_test)
        results[name] = {
            "best_params": search.best_params_,
            "cv_f1_macro": round(float(search.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "test_f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
            "train_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = search.best_estimator_
        r = results[name]
        print(f"{name:<20} CV F1={r['cv_f1_macro']:.4f}  Test Acc={r['test_accuracy']:.4f}  "
              f"Test F1={r['test_f1_macro']:.4f}  ({r['train_seconds']}s)")

    best_name = max(results, key=lambda n: results[n]["cv_f1_macro"])  # select on CV, never on test
    best = fitted[best_name]
    preds = best.predict(X_test)
    labels = list(best.classes_)
    cm = confusion_matrix(y_test, preds, labels=labels)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(best, config.MODEL_PATH)
    plot_confusion_matrix(cm, labels, config.CONFUSION_PATH, f"{best_name} - Confusion Matrix")
    plot_model_comparison(results, config.COMPARISON_PATH)

    metrics = {
        "best_model": best_name,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": labels,
        "comparison": results,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    config.FEATURES_PATH.write_text(json.dumps(top_features(best), indent=2))
    print(f"Saved model   -> {config.MODEL_PATH}")
    print(f"Saved reports -> {config.REPORT_DIR}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the sentiment model")
    ap.add_argument("--data", help="Path to a CSV file (default: bundled dataset)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="sentiment")
    args = ap.parse_args()
    main(args.data, args.text_col, args.label_col)
