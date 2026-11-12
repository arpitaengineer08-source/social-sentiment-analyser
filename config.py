"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_PATH = DATA_DIR / "social_media_posts.csv"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
FEATURES_PATH = REPORT_DIR / "top_features.json"
CONFUSION_PATH = REPORT_DIR / "confusion_matrix.png"
COMPARISON_PATH = REPORT_DIR / "model_comparison.png"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
