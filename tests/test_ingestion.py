import pytest

from drug_sentiment.config import get_config
from drug_sentiment.data.ingestion import RawData, count_physical_lines, load_raw_data, validate_raw_data


@pytest.fixture(scope="module")
def raw() -> RawData:
    return load_raw_data()


def test_real_data_passes_validation(raw):
    cfg = get_config()
    assert len(raw.train) == cfg.data.expected_train_rows
    assert len(raw.test) == cfg.data.expected_test_rows
    assert raw.sample_submission["id"].tolist() == raw.test[cfg.data.id_col].tolist()


def test_brief_row_counts_are_physical_lines():
    """The brief's 5,619 / 3,107 are line counts; comments with line breaks make them exceed the record counts."""
    cfg = get_config()
    assert count_physical_lines(cfg.data.train_path) == 5619
    assert count_physical_lines(cfg.data.test_path) == 3107


def _drop_train_row(r: RawData) -> RawData:
    return RawData(r.train.iloc[1:], r.test, r.sample_submission)


def _duplicate_id(r: RawData) -> RawData:
    train = r.train.copy()
    train.loc[1, "unique_hash"] = train.loc[0, "unique_hash"]
    return RawData(train, r.test, r.sample_submission)


def _null_text(r: RawData) -> RawData:
    train = r.train.copy()
    train.loc[0, "text"] = None
    return RawData(train, r.test, r.sample_submission)


def _unknown_label(r: RawData) -> RawData:
    train = r.train.copy()
    train.loc[0, "sentiment"] = 3
    return RawData(train, r.test, r.sample_submission)


def _reordered_submission(r: RawData) -> RawData:
    return RawData(r.train, r.test, r.sample_submission.iloc[::-1].reset_index(drop=True))


@pytest.mark.parametrize(
    "corrupt", [_drop_train_row, _duplicate_id, _null_text, _unknown_label, _reordered_submission]
)
def test_validation_rejects_corrupt_data(raw, corrupt):
    with pytest.raises(ValueError):
        validate_raw_data(corrupt(raw), get_config())
