import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from drug_sentiment.models.features import COUNT_FEATURES, SCALED_FEATURES, build_features
from drug_sentiment.models.train import ESTIMATORS, build_pipeline, validate_submission


def make_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    words = ["good", "bad", "pain", "relief", "targetdrug", "worse", "better", "news", "trial", "not"]
    frame = pd.DataFrame({"ctx_k1": [" ".join(rng.choice(words, size=8)) for _ in range(n)]})
    for column in COUNT_FEATURES:
        frame[column] = rng.integers(0, 20, n)
    for column in SCALED_FEATURES:
        frame[column] = rng.uniform(-1, 1, n)
    frame["drug_norm"] = rng.choice(["humira", "gilenya"], n)
    frame["match_type"] = "exact"
    for i in range(4):
        frame[f"minilm_{i:03d}"] = rng.normal(size=n)
    return frame


def test_each_representation_suits_its_model_family():
    frame = make_frame()
    sparse_matrix = build_features("sparse").fit_transform(frame)
    dense_matrix = build_features("dense", svd_components=5).fit_transform(frame)
    counts_matrix = build_features("counts").fit_transform(frame)

    assert sparse.issparse(sparse_matrix)
    assert isinstance(dense_matrix, np.ndarray) and dense_matrix.shape[0] == len(frame)
    assert np.allclose(dense_matrix.mean(axis=0), 0, atol=1e-6)  # standardised for KNN and SVC
    assert counts_matrix.min() >= 0  # ComplementNB requires non-negative features


def test_every_notebook_model_can_be_rebuilt_by_the_train_stage():
    assert set(ESTIMATORS) == {
        "LogisticRegression", "LinearSVC", "ComplementNB", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier",
        "RandomForestClassifier", "GradientBoostingClassifier", "LGBMClassifier", "XGBClassifier",
    }


def test_training_pipeline_is_rebuilt_from_the_saved_config():
    config = {
        "estimator": "LogisticRegression",
        "params": {"C": 0.5, "class_weight": "balanced", "max_iter": 2000},
        "representation": "sparse",
        "text_column": "ctx_k1",
    }
    frame = make_frame()
    pipeline = build_pipeline(config, seed=0).fit(frame, np.array([0, 1, 2] * 13 + [2]))

    assert set(pipeline.predict(frame)) <= {0, 1, 2}


def test_submission_validation_rejects_bad_files():
    ids = pd.Series(["a", "b", "c"])
    good = pd.DataFrame({"id": ids, "sentiment": [0, 1, 2]})
    validate_submission(good, ids)

    for bad in (good.iloc[:2], good.assign(sentiment=[0, 1, 3]), good.assign(extra=1), good.iloc[::-1]):
        with pytest.raises(ValueError):
            validate_submission(bad, ids)
