"""Streamlit dashboard for the Social Media Sentiment Analyzer.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

import config
from src.predict import SentimentPredictor

st.set_page_config(page_title="Social Sentiment Analyzer", page_icon="💬", layout="wide")

EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😡"}
EXAMPLES = {
    "Happy customer": "Just tried the new update and it's absolutely amazing! Smooth, fast, 10/10 😍 #loveit",
    "Angry customer": "@SupportTeam worst delivery ever, 3 weeks late and nobody replies 😡 #fail",
    "Negation": "Honestly this phone is not bad at all, pleasantly surprised",
    "Hinglish": "Bhai ye movie toh ekdum zabardast thi, paisa vasool!",
    "News / neutral": "The new policy will be announced on Monday, link below https://t.co/abc",
}


@st.cache_resource(show_spinner="Preparing model (first run trains it, takes a few seconds)...")
def load_predictor() -> SentimentPredictor:
    return SentimentPredictor()


@st.cache_data
def load_metrics() -> dict:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return {}


predictor = load_predictor()
metrics = load_metrics()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("💬 Sentiment Analyzer")
    st.caption("TF-IDF + classical ML for noisy social media text")
    if metrics:
        best = metrics["best_model"]
        st.metric("Active model", best)
        st.metric("Test accuracy", f"{metrics['comparison'][best]['test_accuracy']:.1%}")
        st.metric("Test macro-F1", f"{metrics['comparison'][best]['test_f1_macro']:.3f}")
        st.caption(f"Trained on {metrics['n_train']:,} posts, tested on {metrics['n_test']:,}.")
    st.divider()
    st.caption("Handles emojis, hashtags, @mentions, URLs, slang, negation and Hinglish.")

st.title("Social Media Sentiment Analyzer")
tab_single, tab_batch, tab_insights = st.tabs(["🔍 Analyze a post", "📂 Batch analysis", "📊 Model insights"])

# ---------------------------------------------------------------- single post
with tab_single:
    st.write("Pick an example or type your own post:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, width="stretch", on_click=lambda t=text: st.session_state.update(post=t))

    post = st.text_area("Post text", key="post", height=120, placeholder="Type or paste a tweet / comment / review...")

    if st.button("Analyze", type="primary") or post:
        if not post.strip():
            st.info("Enter some text to analyze.")
        else:
            res = predictor.predict(post)
            left, right = st.columns([1, 1.4])
            with left:
                st.subheader(f"{EMOJI[res['sentiment']]} {res['sentiment'].title()}")
                st.metric("Confidence", f"{res['confidence']:.1%}")
                words = predictor.explain(post)
                if words:
                    st.caption("Words driving this prediction")
                    st.dataframe(pd.DataFrame(words, columns=["token", "influence"]),
                                 hide_index=True, width="stretch")
            with right:
                st.caption("Class probabilities")
                probs = pd.Series(res["probabilities"]).reindex(config.LABELS)
                st.bar_chart(probs)

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV of posts to score every row, then download the results.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up, encoding_errors="replace")
        text_col = st.selectbox("Column containing the post text", df.columns,
                                index=list(df.columns).index("text") if "text" in df.columns else 0)
        if st.button("Run batch analysis", type="primary"):
            with st.spinner(f"Scoring {len(df):,} posts..."):
                out = predictor.predict_batch(df[text_col].astype(str).tolist())
            counts = out["sentiment"].value_counts().reindex(config.LABELS, fill_value=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Posts", f"{len(out):,}")
            for m, lab in zip((m2, m3, m4), ("positive", "neutral", "negative")):
                m.metric(f"{EMOJI[lab]} {lab.title()}", f"{counts[lab]:,}", f"{counts[lab] / len(out):.0%}", delta_color="off")
            st.bar_chart(counts)
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")

# ---------------------------------------------------------------- insights
with tab_insights:
    if not metrics:
        st.warning("Run `python -m src.train` to generate evaluation reports.")
    else:
        st.subheader("Model comparison (model chosen by cross-validation, not test score)")
        comp = pd.DataFrame(metrics["comparison"]).T.drop(columns=["best_params"])
        st.dataframe(comp, width="stretch")
        c1, c2 = st.columns(2)
        if config.COMPARISON_PATH.exists():
            c1.image(str(config.COMPARISON_PATH))
        if config.CONFUSION_PATH.exists():
            c2.image(str(config.CONFUSION_PATH))
        if config.FEATURES_PATH.exists():
            feats = json.loads(config.FEATURES_PATH.read_text())
            if feats:
                st.subheader("Most indicative words per class")
                fcols = st.columns(len(feats))
                for col, (label, items) in zip(fcols, feats.items()):
                    col.markdown(f"**{EMOJI.get(label, '')} {label.title()}**")
                    col.dataframe(pd.DataFrame(items, columns=["token", "weight"]), hide_index=True)
        with st.expander("Full classification report"):
            st.dataframe(pd.DataFrame(metrics["classification_report"]).T.round(3))
