"""Row-wise handcrafted features.

Each value depends only on its own row and the fixed VADER lexicon, so the features are computed once in
the preprocess stage. They follow the EDA: length and long documents (section 5), lexicon sentiment on
the drug's sentences (section 9), plus punctuation, negation and first-person cues in the drug window.
Drug-mention features (count, position, other drugs) come from DrugContext.
"""

from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from drug_sentiment.preprocessing.cleaning import NEGATIONS
from drug_sentiment.preprocessing.drug_context import split_sentences

_TOKEN_RE = re.compile(r"[a-z']+")
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "ive", "i'd", "i'll"})


def mean_sentence_compound(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """VADER compound score averaged over sentences. Scoring a whole long document in one call is very slow
    (about 50 s for the longest, 197k-character page); averaging is linear and weighs sentences equally."""
    scores = [analyzer.polarity_scores(sentence)["compound"] for sentence in split_sentences(text)]
    return sum(scores) / len(scores) if scores else 0.0


def text_features(text: str, window: str, long_doc_chars: int, analyzer: SentimentIntensityAnalyzer) -> dict[str, float]:
    """`text` is the whole minimally cleaned comment; `window` is the unmasked text around the drug."""
    tokens = _TOKEN_RE.findall(window.lower().replace("’", "'"))
    window_scores = analyzer.polarity_scores(window)
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "is_long_doc": int(len(text) > long_doc_chars),
        "window_word_count": len(window.split()),
        "exclamation_count": window.count("!"),
        "question_count": window.count("?"),
        "negation_count": sum(token in NEGATIONS or token.endswith("n't") for token in tokens),
        "first_person_count": sum(token in _FIRST_PERSON for token in tokens),
        "vader_window_compound": window_scores["compound"],
        "vader_window_pos": window_scores["pos"],
        "vader_window_neg": window_scores["neg"],
        "vader_full_mean": mean_sentence_compound(text, analyzer),
    }
