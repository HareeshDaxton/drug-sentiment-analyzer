"""Model inputs: the preprocessed frame joined with MiniLM embeddings, and the feature transformers.

Everything built here is stateful (TF-IDF vocabularies, SVD, scalers, one-hot categories), so it is always
fitted on training rows only: on each training fold during the experiments, or on the full training set for the
main model. The frames it reads come from the row-wise preprocess and embed stages.

Three representations, one per model family:
- "sparse": word + character TF-IDF of the drug window, scaled numeric features, one-hot drug and match type,
  and MiniLM. For linear models, which handle ~60k sparse columns well.
- "dense": the same blocks with the TF-IDF compressed to SVD components, then every column standardised.
  Trees, KNN and RBF-SVC are slow or weak on very wide sparse input, and KNN / SVC need one scale.
- "counts": TF-IDF and one-hot only. ComplementNB needs non-negative features, so no scaled numbers or embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from drug_sentiment.config import Config, get_config
from drug_sentiment.features.embeddings import load_embeddings
from drug_sentiment.preprocessing.cleaning import STOP_WORDS
from drug_sentiment.utils.io import load_pickle

TEXT_COLUMN = "ctx_k1"
COUNT_FEATURES = [
    "mention_count", "n_mention_sentences", "n_other_drugs", "char_count", "word_count",
    "window_word_count", "exclamation_count", "question_count", "negation_count", "first_person_count",
]
SCALED_FEATURES = [
    "first_mention_pos", "is_long_doc", "vader_window_compound", "vader_window_pos", "vader_window_neg",
    "vader_full_mean",
]
CATEGORICAL_FEATURES = ["drug_norm", "match_type"]
REPRESENTATIONS = ("sparse", "dense", "counts")
# Blocks a representation is made of. `blocks` lets the evaluate stage drop one and measure what it was worth.
FEATURE_BLOCKS = ("text", "counts", "scaled", "categories", "minilm")


def load_model_frame(split: str, cfg: Config | None = None) -> pd.DataFrame:
    """Preprocessed rows of a split ("train" or "test") with their MiniLM embedding columns appended."""
    cfg = cfg or get_config()
    path = cfg.artifacts.train_preprocessed if split == "train" else cfg.artifacts.test_preprocessed
    frame = load_pickle(path).reset_index(drop=True)
    embeddings = load_embeddings(split, frame["unique_hash"], cfg.artifacts.embeddings_dir).reset_index(drop=True)
    return pd.concat([frame, embeddings], axis=1)


def _text_tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True,
                                     stop_words=sorted(STOP_WORDS))),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=50_000,
                                     sublinear_tf=True)),
        ]
    )


def build_features(representation: str, text_column: str = TEXT_COLUMN, svd_components: int = 300,
                   seed: int = 42, blocks: tuple[str, ...] = FEATURE_BLOCKS) -> ColumnTransformer | Pipeline:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; expected one of {REPRESENTATIONS}")
    if unknown := set(blocks) - set(FEATURE_BLOCKS):
        raise ValueError(f"unknown feature blocks {sorted(unknown)}; expected a subset of {FEATURE_BLOCKS}")

    categories = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)  # rare drugs share a column
    if representation == "counts":
        return ColumnTransformer(
            [("text", _text_tfidf(), text_column), ("categories", categories, CATEGORICAL_FEATURES)],
            sparse_threshold=1.0,
        )

    text = _text_tfidf()
    if representation == "dense":
        text = Pipeline([("tfidf", text), ("svd", TruncatedSVD(svd_components, random_state=seed))])
    counts = Pipeline(  # counts are heavy-tailed (up to 20k words), so log before scaling
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]
    )
    columns = ColumnTransformer(
        [
            transformer
            for transformer in [
                ("text", text, text_column),
                ("counts", counts, COUNT_FEATURES),
                ("scaled", StandardScaler(), SCALED_FEATURES),
                ("categories", categories, CATEGORICAL_FEATURES),
                ("minilm", "passthrough", make_column_selector(pattern="^minilm_")),
            ]
            if transformer[0] in blocks
        ],
        sparse_threshold=1.0 if representation == "sparse" else 0.0,
    )
    if representation == "sparse":
        return columns
    return Pipeline([("columns", columns), ("scale", StandardScaler())])
