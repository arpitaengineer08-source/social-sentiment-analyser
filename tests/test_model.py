import pytest

import config
from src.data import generate_dataset
from src.predict import SentimentPredictor


@pytest.fixture(scope="session")
def predictor():
    return SentimentPredictor()  # trains automatically if no model exists


def test_dataset_is_balanced_and_valid():
    df = generate_dataset(n_per_class=100)
    assert set(df["sentiment"]) <= set(config.LABELS)
    assert df["sentiment"].value_counts().min() > 80


@pytest.mark.parametrize("text,label", [
    ("I absolutely love this app, it's fantastic!", "positive"),
    ("Worst service ever, totally useless", "negative"),
    ("The update will be released on Monday", "neutral"),
    ("this phone is not good", "negative"),
])
def test_predictions(predictor, text, label):
    assert predictor.predict(text)["sentiment"] == label


def test_probabilities_sum_to_one(predictor):
    res = predictor.predict("great stuff")
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-6


def test_batch_shape(predictor):
    out = predictor.predict_batch(["good", "bad", "ok"])
    assert len(out) == 3 and {"sentiment", "confidence"} <= set(out.columns)
