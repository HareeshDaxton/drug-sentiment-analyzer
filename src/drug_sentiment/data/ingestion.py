"""Load the raw competition CSVs and validate them before anything downstream touches them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drug_sentiment.config import Config, get_config
from drug_sentiment.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def count_physical_lines(path: Path) -> int:
    """Data lines as `wc -l` sees them (header excluded). Larger than the record count
    because some comments contain line breaks inside a quoted CSV field."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def validate_raw_data(raw: RawData, cfg: Config) -> None:
    """Fail fast if the files do not match what the rest of the pipeline assumes."""
    d = cfg.data
    required = {
        "train": [d.id_col, d.text_col, d.drug_col, d.target_col],
        "test": [d.id_col, d.text_col, d.drug_col],
    }
    expected_rows = {"train": d.expected_train_rows, "test": d.expected_test_rows}

    for name, df in (("train", raw.train), ("test", raw.test)):
        missing = set(required[name]) - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if len(df) != expected_rows[name]:
            raise ValueError(f"{name}: expected {expected_rows[name]} records, found {len(df)}")
        null_counts = df[required[name]].isna().sum()
        if null_counts.any():
            raise ValueError(f"{name}: null values found {null_counts[null_counts > 0].to_dict()}")
        if df[d.id_col].duplicated().any():
            raise ValueError(f"{name}: duplicate {d.id_col} values")

    unknown_labels = set(raw.train[d.target_col].unique()) - set(d.label_names)
    if unknown_labels:
        raise ValueError(f"train: unexpected labels {sorted(unknown_labels)}")

    submission = raw.sample_submission
    if list(submission.columns) != ["id", d.target_col]:
        raise ValueError(f"sample submission columns {list(submission.columns)} != ['id', '{d.target_col}']")
    if submission["id"].tolist() != raw.test[d.id_col].tolist():
        raise ValueError("sample submission ids do not match test ids in order")


def load_raw_data(cfg: Config | None = None, validate: bool = True) -> RawData:
    cfg = cfg or get_config()
    d = cfg.data
    raw = RawData(
        train=_read_csv(d.train_path),
        test=_read_csv(d.test_path),
        sample_submission=_read_csv(d.sample_submission_path),
    )
    for name, path, df in (("train", d.train_path, raw.train), ("test", d.test_path, raw.test)):
        logger.info(
            "%s: %d records from %d physical lines (multi-line comments)",
            name, len(df), count_physical_lines(path),
        )
    if validate:
        validate_raw_data(raw, cfg)
        logger.info("Raw data validation passed")
    return raw
