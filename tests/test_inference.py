import numpy as np
import pandas as pd

from drug_sentiment.config import get_config
from drug_sentiment.features.embeddings import FrozenEncoder
from drug_sentiment.inference.predict import DrugSentimentPredictor
from drug_sentiment.models.train import MAIN_MODEL_CONFIG, build_pipeline
from drug_sentiment.preprocessing.drug_context import DrugContextExtractor, DrugLexicon

DRUGS = ["Humira", "Gilenya", "Opdivo"]
COMMENTS = [
    "Humira cleared my psoriasis in three weeks and I have had no side effects at all.",
    "Gilenya gave me headaches every single day and my doctor stopped it.",
    "The trial compared Opdivo with Humira in previously treated patients.",
]


def make_predictor() -> DrugSentimentPredictor:
    """A predictor whose pieces are local: the real lexicon and windowing, a stand-in encoder, a pipeline
    fitted on the pairs below, so the test needs no downloaded model and no saved artifacts."""
    cfg = get_config()
    p = cfg.preprocessing
    lexicon = DrugLexicon.from_train(pd.Series(DRUGS * 4), fuzzy_threshold=p.fuzzy_threshold)
    extractor = DrugContextExtractor(lexicon, p.target_token, p.other_token, p.window_sizes,
                                     p.natural_window_size, p.max_window_words)
    columns = [f"minilm_{i:03d}" for i in range(4)]
    encoder = FrozenEncoder("stand-in", None, columns,
                            lambda texts: np.tile(np.linspace(0, 1, 4), (len(texts), 1)))
    return DrugSentimentPredictor(pipeline=None, extractor=extractor, encoder=encoder, cfg=cfg)


def training_pairs() -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs = [(drug, comment) for comment in COMMENTS for drug in DRUGS] * 3
    labels = np.tile([0, 1, 2], len(pairs) // 3 + 1)[: len(pairs)]
    return pairs, labels


def test_a_raw_pair_gets_the_same_columns_the_model_was_trained_on():
    predictor = make_predictor()
    frame = predictor.build_frame([("Humira", COMMENTS[0]), ("Opdivo", COMMENTS[2])])

    assert not frame.isna().any().any()
    assert frame.filter(like="minilm_").shape == (2, 4)
    assert "targetdrug" in frame.loc[0, "ctx_k1"] and "humira" not in frame.loc[0, "ctx_k1"]
    assert "otherdrug" in frame.loc[1, "ctx_k1"]  # the other drug in the same sentence stays anonymous


def test_the_same_comment_gives_each_drug_its_own_prediction():
    predictor = make_predictor()
    pairs, labels = training_pairs()
    predictor.pipeline = build_pipeline(MAIN_MODEL_CONFIG, seed=0).fit(predictor.build_frame(pairs), labels)

    shared = COMMENTS[2]  # one comment, two drugs
    first, second = predictor.predict([("Opdivo", shared), ("Humira", shared)])

    assert first.window != second.window
    assert first.label in {"positive", "negative", "neutral"}
    assert abs(sum(first.scores.values()) - 1) < 1e-9
    assert first.match_type == "exact" and first.mention_count >= 1
    assert predictor.predict_one("Opdivo", shared).sentiment == first.sentiment  # deterministic
