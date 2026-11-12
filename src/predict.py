"""Inference API + CLI.

    python -m src.predict "I love this phone!"
    python -m src.predict --file posts.csv --column text --out results.csv
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import config
from src.model_utils import class_weights


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SentimentPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            print("No trained model found - training one now (takes a few seconds)...")
            from src.train import main as train
            train()
        self.model = joblib.load(model_path)
        self.classes = list(self.model.classes_)

    def _proba(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(texts)
        return _softmax(self.model.decision_function(texts))  # e.g. LinearSVC

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        probs = self._proba(list(texts))
        idx = probs.argmax(axis=1)
        out = pd.DataFrame(probs, columns=[f"p_{c}" for c in self.classes])
        out.insert(0, "text", list(texts))
        out.insert(1, "sentiment", [self.classes[i] for i in idx])
        out.insert(2, "confidence", probs.max(axis=1).round(4))
        return out

    def predict(self, text: str) -> dict:
        row = self.predict_batch([text]).iloc[0]
        return {
            "text": text,
            "sentiment": row["sentiment"],
            "confidence": float(row["confidence"]),
            "probabilities": {c: float(row[f"p_{c}"]) for c in self.classes},
        }

    def explain(self, text: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Words/phrases that pushed the prediction towards its predicted class."""
        weights = class_weights(self.model.named_steps["clf"])
        if weights is None:
            return []
        vec = self.model.named_steps["tfidf"]
        x = vec.transform([text])
        label = self.predict(text)["sentiment"]
        coef = weights[self.classes.index(label)]
        names = vec.get_feature_names_out()
        contrib = x.multiply(coef).tocoo()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: p[1], reverse=True)
        return [(names[c], round(float(v), 3)) for c, v in pairs[:top_k] if v > 0]


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict sentiment of social media posts")
    ap.add_argument("texts", nargs="*", help="One or more posts (wrap each in quotes)")
    ap.add_argument("--file", help="CSV file with posts")
    ap.add_argument("--column", default="text", help="Text column in the CSV")
    ap.add_argument("--out", help="Where to save CSV results")
    args = ap.parse_args()

    predictor = get_predictor()
    if args.file:
        df = pd.read_csv(args.file)
        result = predictor.predict_batch(df[args.column].astype(str).tolist())
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Saved {len(result)} predictions -> {args.out}")
        else:
            print(result.to_string(index=False))
        print("\n" + result["sentiment"].value_counts().to_string())
    elif args.texts:
        for t in args.texts:
            r = predictor.predict(t)
            probs = " | ".join(f"{k}: {v:.0%}" for k, v in r["probabilities"].items())
            print(f"\n\"{t}\"\n  -> {r['sentiment'].upper()} ({r['confidence']:.0%})   [{probs}]")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
