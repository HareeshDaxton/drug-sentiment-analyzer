"""Stage `preprocess`: raw CSVs -> leakage-safe, model-ready DataFrames saved as pickles.

Every column is computed from its own row plus fixed reference data: the drug names in train.csv's `drug`
column, config/drug_synonyms.yaml and the VADER lexicon. Nothing is fitted on labels or test rows, so
the saved frames can feed every model and every CV fold. Stateful steps (TF-IDF, scaling, drug encoding)
are fitted later, inside each model's pipeline.

Columns
- ids and inputs: unique_hash, drug, text, drug_norm
- text for TF-IDF (classical cleaning): text_clean (drug-blind), full_masked, ctx_k0 / ctx_k1 / ctx_k2
  (mention sentences +/- k neighbours, drugs masked)
- ctx_natural: unmasked window for pretrained embedding models
- drug context: match_type, mention_count, first_mention_pos, n_mention_sentences, n_other_drugs
- handcrafted: see features/handcrafted.py
- train only: sentiment, fold
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from drug_sentiment.config import Config, get_config
from drug_sentiment.data.ingestion import load_raw_data
from drug_sentiment.features.handcrafted import text_features
from drug_sentiment.logger import get_logger
from drug_sentiment.preprocessing.cleaning import clean_classical, clean_minimal
from drug_sentiment.preprocessing.drug_context import DrugContextExtractor, DrugLexicon, normalise_drug
from drug_sentiment.utils.io import save_pickle

logger = get_logger(__name__)


def build_extractor(train_drugs: pd.Series, cfg: Config) -> DrugContextExtractor:
    p = cfg.preprocessing
    lexicon = DrugLexicon.from_train(train_drugs, fuzzy_threshold=p.fuzzy_threshold)
    return DrugContextExtractor(
        lexicon, p.target_token, p.other_token, p.window_sizes, p.natural_window_size, p.max_window_words
    )


def preprocess_row(text: str, drug: str, extractor: DrugContextExtractor,
                   analyzer: SentimentIntensityAnalyzer, long_doc_chars: int) -> dict[str, object]:
    text = clean_minimal(text)
    context = extractor.extract(text, drug)
    return {
        "drug_norm": normalise_drug(drug),
        "text_clean": clean_classical(text),
        "full_masked": clean_classical(context.full_masked),
        **{f"ctx_k{k}": clean_classical(window) for k, window in context.windows_masked.items()},
        "ctx_natural": context.window_natural,
        "match_type": context.match_type,
        "mention_count": context.mention_count,
        "first_mention_pos": context.first_mention_pos,
        "n_mention_sentences": context.n_mention_sentences,
        "n_other_drugs": context.n_other_drugs,
        **text_features(text, context.window_natural, long_doc_chars, analyzer),
    }


def preprocess_frame(df: pd.DataFrame, extractor: DrugContextExtractor, cfg: Config) -> pd.DataFrame:
    d = cfg.data
    analyzer = SentimentIntensityAnalyzer()
    rows = [
        preprocess_row(text, drug, extractor, analyzer, cfg.preprocessing.long_doc_chars)
        for text, drug in zip(df[d.text_col], df[d.drug_col])
    ]
    return pd.concat([df[[d.id_col, d.drug_col, d.text_col]], pd.DataFrame(rows, index=df.index)], axis=1)


def assign_folds(labels: pd.Series, texts: pd.Series, n_splits: int, seed: int, shuffle: bool = True) -> np.ndarray:
    """Fold id per row, stratified by label and grouped by comment text so a comment never spans folds."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = np.full(len(labels), -1)
    groups = pd.factorize(texts)[0]
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        folds[val_idx] = fold
    return folds


def run_preprocessing(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or get_config()
    d, a = cfg.data, cfg.artifacts
    raw = load_raw_data(cfg)
    extractor = build_extractor(raw.train[d.drug_col], cfg)
    logger.info("Drug lexicon: %d names from train drugs + synonym file", len(extractor.lexicon.all_names))

    frames = {}
    for name, df in (("train", raw.train), ("test", raw.test)):
        start = time.perf_counter()
        frames[name] = preprocess_frame(df, extractor, cfg)
        logger.info(
            "%s: %d rows preprocessed in %.0fs; drug match types %s",
            name, len(df), time.perf_counter() - start, frames[name]["match_type"].value_counts().to_dict(),
        )

    train, test = frames["train"], frames["test"]
    train[d.target_col] = raw.train[d.target_col]
    train["fold"] = assign_folds(train[d.target_col], train[d.text_col], cfg.cv.n_splits, cfg.seed, cfg.cv.shuffle)
    logger.info("Fold sizes: %s", train["fold"].value_counts().sort_index().to_dict())

    save_pickle(train, a.train_preprocessed)
    save_pickle(test, a.test_preprocessed)
    save_pickle(extractor.lexicon, a.drug_lexicon)
    logger.info("Saved %s %s and %s %s", a.train_preprocessed.name, train.shape, a.test_preprocessed.name, test.shape)
    return train, test
