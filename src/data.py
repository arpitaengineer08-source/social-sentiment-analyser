"""Dataset utilities: synthetic social-media corpus generator + loader.

The bundled dataset is synthetic (template based, with noise) so the project
runs fully offline. To train on real data (e.g. Sentiment140 or a Kaggle
Twitter dataset) use `python -m src.train --data your.csv --text-col text
--label-col sentiment`.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

import config

SUBJECTS = [
    "this phone", "the new update", "the movie", "customer service", "this app",
    "the delivery", "the food", "the match", "the new iPhone", "this show",
    "the airline", "the concert", "the new policy", "this laptop", "the game",
    "the hotel", "this song", "the cab ride", "the new season", "this brand",
]
POS_ADJ = ["amazing", "fantastic", "brilliant", "superb", "awesome", "excellent",
           "lovely", "incredible", "impressive", "fabulous", "outstanding", "perfect"]
NEG_ADJ = ["terrible", "awful", "horrible", "disappointing", "garbage", "pathetic",
           "useless", "overpriced", "annoying", "broken", "trash", "unacceptable"]
MILD_NEG = ["bad", "terrible", "that bad", "half bad", "awful"]
MILD_POS = ["good", "great", "impressive", "worth it", "helpful"]
HI_POS = ["mast", "zabardast", "badhiya", "kamaal", "ekdum jhakaas", "shaandar"]
HI_NEG = ["bakwas", "bekar", "ghatiya", "faltu", "kharab", "waste"]

POS_T = [
    "I absolutely love {s}!", "{s} is {a}, totally worth it", "Just tried {s} and it's {a}",
    "{s} exceeded my expectations", "Can't believe how {a} {s} is", "Honestly {s} made my day",
    "10/10 would recommend {s}", "{s} is so {a} right now", "Big thanks, {s} was {a}",
    "Loving {s} so much, {a} stuff", "Wow, {s} is {a}. Well done team",
]
NEG_T = [
    "I absolutely hate {s}", "{s} is {a}, total waste of money", "Just tried {s} and it's {a}",
    "{s} ruined my day", "Can't believe how {a} {s} is", "Never using {s} again",
    "0/10 would not recommend {s}", "{s} is so {a} right now", "Really disappointed, {s} was {a}",
    "So frustrated with {s}, {a} experience", "Ugh, {s} is {a}. Fix it already",
]
NEU_T = [
    "{s} will be available from Monday", "Just saw {s} at the store", "Anyone know when {s} is releasing?",
    "Reading a review of {s}", "{s} was announced today", "Waiting for {s} to arrive",
    "{s} comes in three colors", "Thread about {s}, link below", "Is {s} available in my city?",
    "Watching the livestream about {s}", "{s} price list was shared today", "Comparing {s} with last year's model",
]
NEG_OF_POS_T = ["{s} is not {m}", "honestly {s} isn't {m} at all", "{s} was never {m}"]
HI_POS_T = ["{s} ekdum {h} hai yaar", "{s} bilkul {h} laga", "Bhai {s} toh {h} nikla", "{s} {h} hai, loved it"]
HI_NEG_T = ["{s} ekdum {h} hai yaar", "{s} bilkul {h} laga", "Bhai {s} toh {h} nikla", "{s} {h} hai, paisa barbaad"]
HI_NEU_T = ["{s} kal launch ho raha hai", "{s} ke baare me kisi ko pata hai?", "{s} ka review dekh raha hu"]

POS_EMOJI = ["😍", "🔥", "👏", "😊", "🎉", "❤️", "💯", ""]
NEG_EMOJI = ["😡", "😤", "👎", "😞", "🤬", "💔", ""]
NEU_EMOJI = ["", "", "🤔", "👀", ""]
POS_TAGS = ["#blessed", "#win", "#loveit", "#happy", ""]
NEG_TAGS = ["#fail", "#disappointed", "#worst", "#nevergain", ""]
NEU_TAGS = ["#news", "#update", "#tech", "#daily", ""]
BRANDS = ["@SupportTeam", "@BrandOfficial", "@CareDesk", "@TechDaily", ""]


def _decorate(text: str, label: str, rng: random.Random) -> str:
    emoji = rng.choice({"positive": POS_EMOJI, "negative": NEG_EMOJI, "neutral": NEU_EMOJI}[label])
    tag = rng.choice({"positive": POS_TAGS, "negative": NEG_TAGS, "neutral": NEU_TAGS}[label])
    brand = rng.choice(BRANDS)
    url = rng.choice(["", "", "", "https://t.co/xYz12Ab"])
    r = rng.random()
    if r < 0.12:
        text = text.upper()
    elif r < 0.30:
        text = text.lower()
    if rng.random() < 0.15:  # elongation
        text = text.replace("so ", "sooo ", 1).replace("love", "looove", 1)
    if rng.random() < 0.10:  # exclamation burst
        text += "!!!"
    parts = [brand, text, emoji, tag, url]
    return " ".join(p for p in parts if p).strip()


def generate_dataset(n_per_class: int = 1500, seed: int = config.RANDOM_STATE,
                     label_noise: float = 0.03) -> pd.DataFrame:
    """Create a balanced, noisy social-media style dataset."""
    rng = random.Random(seed)
    rows, seen = [], set()
    labels = ["positive", "negative", "neutral"]
    attempts = 0
    while any(sum(1 for r in rows if r[1] == lb) < n_per_class for lb in labels) and attempts < n_per_class * 60:
        attempts += 1
        lb = rng.choice(labels)
        if sum(1 for r in rows if r[1] == lb) >= n_per_class:
            continue
        s = rng.choice(SUBJECTS)
        kind = rng.random()
        if lb == "positive":
            if kind < 0.70:
                t = rng.choice(POS_T).format(s=s, a=rng.choice(POS_ADJ))
            elif kind < 0.85:
                t = rng.choice(NEG_OF_POS_T).format(s=s, m=rng.choice(MILD_NEG))  # "not bad"
            else:
                t = rng.choice(HI_POS_T).format(s=s, h=rng.choice(HI_POS))
        elif lb == "negative":
            if kind < 0.70:
                t = rng.choice(NEG_T).format(s=s, a=rng.choice(NEG_ADJ))
            elif kind < 0.85:
                t = rng.choice(["{s} is not {m}", "{s} isn't {m}, sadly", "{s} was never {m}"]).format(
                    s=s, m=rng.choice(MILD_POS))  # "not good"
            else:
                t = rng.choice(HI_NEG_T).format(s=s, h=rng.choice(HI_NEG))
        else:
            pool = HI_NEU_T if kind > 0.88 else NEU_T
            t = rng.choice(pool).format(s=s)
        t = t[0].upper() + t[1:]
        post = _decorate(t, lb, rng)
        if post in seen:
            continue
        seen.add(post)
        rows.append((post, lb))

    df = pd.DataFrame(rows, columns=["text", "sentiment"])
    # realistic annotation noise
    flip = rng.sample(range(len(df)), int(len(df) * label_noise))
    for i in flip:
        df.at[i, "sentiment"] = rng.choice([lb for lb in labels if lb != df.at[i, "sentiment"]])
    return df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)


def ensure_dataset(path: Path = config.DATA_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(path, index=False)
    return path


def load_dataset(path: Path | str | None = None, text_col: str = "text",
                 label_col: str = "sentiment") -> pd.DataFrame:
    """Load a CSV and normalise labels to negative / neutral / positive."""
    path = Path(path) if path else ensure_dataset()
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    if text_col not in df or label_col not in df:
        raise ValueError(f"CSV must contain columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"}).dropna()
    mapping = {0: "negative", 2: "neutral", 4: "positive", -1: "negative", 1: "positive",
               "0": "negative", "2": "neutral", "4": "positive", "-1": "negative", "1": "positive"}
    df["sentiment"] = df["sentiment"].map(lambda v: mapping.get(v, str(v).strip().lower()))
    df = df[df["sentiment"].isin(config.LABELS)]
    if df.empty:
        raise ValueError(f"No rows with labels in {config.LABELS} after cleaning.")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = config.DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset().to_csv(out, index=False)
    print(f"Dataset written to {out}")
