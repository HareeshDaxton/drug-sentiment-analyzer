"""Stages `train` and `predict`: fit the project's main model on all training rows and write the submission.

The main model is **LinearSVC** on the sparse representation of the drug-centred window. NoteBook/
model_expriments.ipynb chose it: it screened 10 classical models on half of the training data, tuned the top 3 with
Optuna on the full data over the 5 stored folds, and saved the winner to artifacts/model/best_model_config.json --
estimator class, hyperparameters, feature representation, whether balanced sample weights are needed, and its
cross-validated scores.

`train` reads that file (falling back to MAIN_MODEL_CONFIG below, so the pipeline runs end to end without the
notebook), fits the feature transformer and the estimator together on all 5,279 training rows, and saves them as one
sklearn Pipeline. `predict` only transforms test.csv with the fitted pipeline and writes predictions.csv.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from drug_sentiment.config import Config, get_config
from drug_sentiment.logger import get_logger
from drug_sentiment.models.features import FEATURE_BLOCKS, build_features, load_model_frame
from drug_sentiment.utils.io import load_json, load_pickle, save_json, save_pickle

logger = get_logger(__name__)

BEST_CONFIG_FILE = "best_model_config.json"
MODEL_FILE = "best_model.pkl"
MODEL_CARD_FILE = "model_card.json"

# The model the project ships, as chosen in the experiments notebook. Used when best_model_config.json is absent,
# so a fresh clone can run `main.py all` without opening a notebook; the notebook rewrites the file with its own
# tuned values and cross-validation scores.
MAIN_MODEL_CONFIG: dict = {
    "model": "LinearSVC",
    "estimator": "LinearSVC",
    "params": {"C": 0.05, "class_weight": "balanced", "max_iter": 20000},
    "representation": "sparse",
    "text_column": "ctx_k0",  # the evaluate stage's window search chose the tightest window
    "sample_weight": False,
}

# The estimators the experiments notebook compares; the saved config names one of these classes.
ESTIMATORS = {
    estimator.__name__: estimator
    for estimator in (
        LogisticRegression, LinearSVC, ComplementNB, SVC, KNeighborsClassifier, DecisionTreeClassifier,
        RandomForestClassifier, GradientBoostingClassifier, LGBMClassifier, XGBClassifier,
    )
}


def class_scores(estimator, X) -> np.ndarray:
    """Per-class scores that sum to 1. LinearSVC has no predict_proba, so its decision values go through a softmax:
    it keeps the ranking of the classes and makes the three scores readable as confidences."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    decision = estimator.decision_function(X)
    decision = decision - decision.max(axis=1, keepdims=True)
    return np.exp(decision) / np.exp(decision).sum(axis=1, keepdims=True)


def load_model_config(cfg: Config | None = None) -> dict:
    """The main model's definition: the notebook's saved config if it exists, else the built-in default."""
    cfg = cfg or get_config()
    path = cfg.artifacts.model_dir / BEST_CONFIG_FILE
    if not path.exists():
        logger.warning("%s not found; using the built-in %s config", BEST_CONFIG_FILE, MAIN_MODEL_CONFIG["model"])
        return dict(MAIN_MODEL_CONFIG)
    config = load_json(path)
    logger.info("Main model %s from %s: %s", config["model"], BEST_CONFIG_FILE, config["params"])
    return config


def build_pipeline(config: dict, seed: int, blocks: tuple[str, ...] = FEATURE_BLOCKS) -> Pipeline:
    """Feature transformer + estimator described by a best_model_config.json dictionary. `blocks` narrows the
    feature set, which the evaluate stage uses to measure what each block is worth."""
    estimator = ESTIMATORS[config["estimator"]](**config["params"])
    features = build_features(config["representation"], text_column=config["text_column"], seed=seed, blocks=blocks)
    return Pipeline([("features", features), ("model", estimator)])


def validate_submission(submission: pd.DataFrame, test_ids: pd.Series) -> None:
    """The brief's submission checklist: two columns, one row per test id in order, labels 0/1/2 as integers."""
    if list(submission.columns) != ["id", "sentiment"]:
        raise ValueError(f"columns must be ['id', 'sentiment'], got {list(submission.columns)}")
    if len(submission) != len(test_ids):
        raise ValueError(f"expected {len(test_ids)} rows, got {len(submission)}")
    if submission["id"].tolist() != list(test_ids):
        raise ValueError("ids must match test.csv's unique_hash values in the same order")
    labels = submission["sentiment"]
    if not pd.api.types.is_integer_dtype(labels) or not labels.isin([0, 1, 2]).all():
        raise ValueError("sentiment must contain only the integers 0, 1 and 2")


def run_training(cfg: Config | None = None) -> Pipeline:
    cfg = cfg or get_config()
    model_dir = cfg.artifacts.model_dir
    config = load_model_config(cfg)
    train = load_model_frame("train", cfg)
    y = train[cfg.data.target_col].to_numpy()

    pipeline = build_pipeline(config, cfg.seed)
    fit_params = {"model__sample_weight": compute_sample_weight("balanced", y)} if config.get("sample_weight") else {}
    start = time.perf_counter()
    pipeline.fit(train, y, **fit_params)
    seconds = round(time.perf_counter() - start)
    n_features = pipeline.named_steps["features"].transform(train.head(1)).shape[1]

    save_pickle(pipeline, model_dir / MODEL_FILE)
    save_json(
        {
            **{key: config[key] for key in ("model", "estimator", "params", "representation", "text_column")},
            "cv_metrics": config.get("cv_metrics"),
            "fold_f1_std": config.get("fold_f1_std"),
            "selection": config.get("selection"),
            "trained_rows": len(train),
            "n_features": int(n_features),
            "class_counts": {int(label): int(count) for label, count in pd.Series(y).value_counts().sort_index().items()},
            "fit_seconds": seconds,
            "seed": cfg.seed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
        model_dir / MODEL_CARD_FILE,
    )
    logger.info("Trained %s on %d rows (%d features) in %ds; saved %s",
                config["model"], len(train), n_features, seconds, MODEL_FILE)
    return pipeline


def run_prediction(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    pipeline = load_pickle(cfg.artifacts.model_dir / MODEL_FILE)
    test = load_model_frame("test", cfg)
    submission = pd.DataFrame({"id": test["unique_hash"], "sentiment": pipeline.predict(test).astype(int)})
    validate_submission(submission, pd.read_csv(cfg.data.sample_submission_path)["id"])

    path = cfg.artifacts.submission_file
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    logger.info("Wrote %s: %d rows, predicted label counts %s", path.name, len(submission),
                submission["sentiment"].value_counts().sort_index().to_dict())
    return submission
