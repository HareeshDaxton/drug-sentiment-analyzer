"""Descriptive statistics for the EDA notebook.

These helpers describe the data; they are not model features. The leakage-safe feature pipeline
(fuzzy drug matching, masking, sentence windows) is built in preprocessing/ on the same ideas.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from drug_sentiment.preprocessing.cleaning import STOP_WORDS, clean_classical
from drug_sentiment.preprocessing.drug_context import split_sentences


def first_mention_word(text: str, drug: str) -> int:
    """Word index of the first literal mention of the drug; -1 when it is not found."""
    position = text.lower().find(drug.lower())
    return -1 if position < 0 else len(text[:position].split())


def drug_sentences(text: str, drug: str) -> str:
    """Sentences that mention the drug literally; the whole text when none do."""
    drug = drug.lower()
    sentences = [sentence for sentence in split_sentences(text) if drug in sentence.lower()]
    return " ".join(sentences) if sentences else text


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Length and drug-mention statistics per row (expects `text` and `drug` columns)."""
    out = df.copy()
    pairs = list(zip(out["text"], out["drug"]))
    out["word_count"] = out["text"].str.split().str.len()
    out["char_count"] = out["text"].str.len()
    out["mention_count"] = [text.lower().count(drug.lower()) for text, drug in pairs]
    out["first_mention_word"] = [first_mention_word(text, drug) for text, drug in pairs]
    out["drugs_sharing_text"] = out.groupby("text")["drug"].transform("nunique")
    out["drug_sentences"] = [drug_sentences(text, drug) for text, drug in pairs]
    return out


def label_share(df: pd.DataFrame, by: str | pd.Series, label_col: str = "label") -> pd.DataFrame:
    """Share of each label within each group (rows sum to 1) plus the group size `n`."""
    groups = df[by] if isinstance(by, str) else by
    share = pd.crosstab(groups, df[label_col], normalize="index")
    share["n"] = groups.value_counts()
    return share


def vader_compound(texts: Iterable[str]) -> np.ndarray:
    """VADER compound score in [-1, 1]. Rule-based lexicon: nothing is learned from the data."""
    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(text)["compound"] for text in texts])


def distinctive_terms(texts: Iterable[str], labels: pd.Series, exclude: set[str],
                      top_n: int = 15, min_df: int = 15) -> pd.DataFrame:
    """Words most over-represented in each class.

    Score: log of the (add-one smoothed) share of in-class documents containing the word minus the
    same share for the other classes. Stop words are removed except negations; `exclude` drops extra
    tokens such as drug names.
    """
    exclude = {token for token in exclude if re.fullmatch(r"[a-z]{2,}", token)}
    vectorizer = CountVectorizer(
        binary=True,
        stop_words=sorted(STOP_WORDS | exclude),
        min_df=min_df,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    matrix = vectorizer.fit_transform(clean_classical(text) for text in texts)
    vocab = vectorizer.get_feature_names_out()

    rows = []
    for label in sorted(labels.unique()):
        in_class = (labels == label).to_numpy()
        docs_in = np.asarray(matrix[in_class].sum(axis=0)).ravel()
        docs_out = np.asarray(matrix[~in_class].sum(axis=0)).ravel()
        log_ratio = np.log((docs_in + 1) / (in_class.sum() + 2)) - np.log((docs_out + 1) / ((~in_class).sum() + 2))
        for i in np.argsort(-log_ratio, kind="stable")[:top_n]:
            rows.append({"label": label, "term": vocab[i], "log_ratio": log_ratio[i], "docs_in_class": int(docs_in[i])})
    return pd.DataFrame(rows)
