import numpy as np
import pandas as pd
import pytest

from drug_sentiment.config import get_config
from drug_sentiment.models import evaluate
from drug_sentiment.models.features import COUNT_FEATURES, FEATURE_BLOCKS, SCALED_FEATURES, build_features
from drug_sentiment.models.train import MAIN_MODEL_CONFIG, build_pipeline

WORDS = ["helped", "pain", "relief", "worse", "targetdrug", "otherdrug", "news", "trial", "not", "rash"]


def make_frame(n: int = 75, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "unique_hash": [f"h{i:03d}" for i in range(n)],
        "drug": rng.choice(["humira", "gilenya", "opdivo"], n),
        "text": [f"comment {i % 40}" for i in range(n)],  # some comments repeat across drugs
        "ctx_k0": [" ".join(rng.choice(WORDS, size=7)) for _ in range(n)],  # the window the main model reads
        "ctx_k1": [" ".join(rng.choice(WORDS, size=9)) for _ in range(n)],
        "sentiment": np.tile([0, 1, 2], n // 3 + 1)[:n],
        "fold": np.arange(n) % 5,
    })
    for column in COUNT_FEATURES:
        frame[column] = rng.integers(0, 20, n)
    frame["char_count"] = rng.integers(100, 12_000, n)
    for column in SCALED_FEATURES:
        frame[column] = rng.uniform(-1, 1, n)
    frame["drug_norm"] = frame["drug"]
    frame["match_type"] = rng.choice(["exact", "fuzzy", "none"], n)
    for i in range(4):
        frame[f"minilm_{i:03d}"] = rng.normal(size=n)
    return frame


def test_dropping_a_feature_block_drops_its_columns():
    frame = make_frame()
    full = build_features("sparse").fit_transform(frame).shape[1]
    without_embeddings = build_features("sparse", blocks=tuple(b for b in FEATURE_BLOCKS if b != "minilm"))
    assert without_embeddings.fit_transform(frame).shape[1] == full - 4

    with pytest.raises(ValueError):
        build_features("sparse", blocks=("text", "typo"))


def test_cross_validation_predicts_every_row_out_of_fold():
    frame = make_frame()
    y = frame["sentiment"].to_numpy()
    result = evaluate.cross_validate(MAIN_MODEL_CONFIG, frame, y, get_config())
    text_only = evaluate.cross_validate(MAIN_MODEL_CONFIG, frame, y, get_config(), blocks=("text",))

    assert len(result["predicted"]) == len(frame)
    assert len(text_only["predicted"]) == len(frame)  # an ablation runs the same way with fewer blocks
    assert set(result["predicted"]) <= {0, 1, 2}
    assert np.allclose(result["scores"].sum(axis=1), 1)  # softmax scores for a model without predict_proba
    assert 0 <= result["f1_weighted"] <= 1


def test_every_row_is_counted_once_per_slice_dimension():
    frame = make_frame()
    y = frame["sentiment"].to_numpy()
    table = evaluate.slice_scores(frame, y, y.copy())

    assert table.groupby("dimension")["rows"].sum().eq(len(frame)).all()
    assert table["f1_weighted"].eq(1).all()  # perfect predictions score 1 in every slice


def test_each_ablation_group_has_exactly_one_shipped_setting():
    groups = pd.DataFrame(evaluate.ablation_variants()).groupby("group")["name"]
    assert groups.apply(lambda names: (names == evaluate.SHIPPED).sum()).eq(1).all()


def test_top_words_reports_word_features_only():
    frame = make_frame()
    pipeline = build_pipeline(MAIN_MODEL_CONFIG, seed=0).fit(frame, frame["sentiment"].to_numpy())
    words = evaluate.top_words(pipeline, get_config(), n=5)

    assert set(words["sentiment"]) == {"positive", "negative", "neutral"}
    assert words.groupby("sentiment").size().eq(5).all()
    assert not words["word"].str.startswith("minilm").any()
    assert set(words["word"]) <= set(WORDS) | {f"{a} {b}" for a in WORDS for b in WORDS}  # unigrams and bigrams
