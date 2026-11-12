"""Text cleaning tailored for noisy social media posts.

Handles URLs, @mentions, hashtags, emojis, elongated words ("soooo"),
contractions, Hinglish negations and explicit negation scope marking
("not good" -> "not NOT_good"), which matters a lot for sentiment.
No external downloads (NLTK etc.) are required.
"""
from __future__ import annotations

import html
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"[a-z_']+|[!?.,;:]")

POSITIVE_EMOJI = "😀😃😄😁😊😍🥰😘😎🤩👍👏🙌🎉❤💖💯🔥✨😂🤣😻"
NEGATIVE_EMOJI = "😡😠🤬😤😞😢😭😩😫👎💔🤮😒🙄😖😣"

CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'s\b"), ""),
]

NEGATIONS = {"not", "no", "never", "nothing", "neither", "nor", "hardly",
             "nahi", "nahin", "nhi", "na", "mat", "cannot"}
CLAUSE_BREAKS = {".", ",", ";", ":", "!", "?"}
NEGATION_WINDOW = 3

# Deliberately small: negations and sentiment-bearing words are NOT removed.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "to", "from", "in", "on", "up", "is", "are", "was", "were", "be", "been",
    "am", "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "they", "them", "their", "so", "as",
    "do", "does", "did", "have", "has", "had", "just", "rt", "ye", "yeh", "hai",
    "hain", "ka", "ki", "ke", "ko", "se", "aur", "toh",
}


def _replace_emojis(text: str) -> str:
    out = []
    for ch in text:
        if ch in POSITIVE_EMOJI:
            out.append(" emojipos ")
        elif ch in NEGATIVE_EMOJI:
            out.append(" emojineg ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str) -> str:
    """Normalise a raw post into a space-separated string of tokens."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _replace_emojis(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = REPEAT_RE.sub(r"\1\1", text)  # soooo -> soo
    text = re.sub(r"\bcan'?t believe\b", "wow", text)  # idiom, not a negation
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)

    tokens, negate_left = [], 0
    for tok in TOKEN_RE.findall(text):
        if tok in CLAUSE_BREAKS:
            negate_left = 0
            continue
        tok = tok.strip("'")
        if not tok:
            continue
        if tok in NEGATIONS:
            negate_left = NEGATION_WINDOW
            tokens.append(tok)
            continue
        if tok in STOPWORDS:
            continue
        if negate_left > 0 and not tok.startswith("emoji"):
            tokens.append(f"NOT_{tok}")
            negate_left -= 1
        else:
            tokens.append(tok)
    return " ".join(tokens)
