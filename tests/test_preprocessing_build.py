import pandas as pd

from drug_sentiment.config import get_config
from drug_sentiment.data.ingestion import load_raw_data
from drug_sentiment.preprocessing.build import assign_folds, preprocess_frame
from drug_sentiment.preprocessing.drug_context import DrugContextExtractor, DrugLexicon


def test_preprocessing_is_row_wise():
    """A row's output must not depend on the other rows processed with it (nothing is fitted)."""
    cfg = get_config()
    comment = "Humira made me sick. Stelara works!"
    df = pd.DataFrame(
        {
            "unique_hash": ["a", "b", "c"],
            "text": [comment, comment, "No drug is named here."],
            "drug": ["humira", "stelara", "humira"],
        }
    )
    lexicon = DrugLexicon(["humira", "stelara"], [], fuzzy_threshold=80)
    extractor = DrugContextExtractor(lexicon, "targetdrug", "otherdrug", (0, 1, 2), 1, 250)

    batch = preprocess_frame(df, extractor, cfg)
    one_by_one = pd.concat([preprocess_frame(df.iloc[[i]], extractor, cfg) for i in range(len(df))])

    pd.testing.assert_frame_equal(batch, one_by_one)
    assert batch.loc[0, "ctx_k0"] == "targetdrug made me sick"
    assert batch.loc[1, "ctx_k0"] == "targetdrug works !"
    assert batch.loc[1, "full_masked"] == "otherdrug made me sick targetdrug works !"


def test_folds_are_stratified_and_never_split_a_comment():
    cfg = get_config()
    train = load_raw_data(cfg).train
    folds = assign_folds(train["sentiment"], train["text"], n_splits=5, seed=cfg.seed)

    assert set(folds) == set(range(5))
    assert (pd.Series(folds).groupby(train["text"].to_numpy()).nunique() == 1).all()
    overall = train["sentiment"].value_counts(normalize=True)
    for fold in range(5):
        in_fold = train.loc[folds == fold, "sentiment"].value_counts(normalize=True)
        assert (in_fold - overall).abs().max() < 0.03
