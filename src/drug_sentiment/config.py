"""Load config/config.yaml into typed, immutable settings with paths resolved against the project root."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    expected_train_rows: int
    expected_test_rows: int
    id_col: str
    text_col: str
    drug_col: str
    target_col: str
    label_names: dict[int, str]


@dataclass(frozen=True)
class ArtifactConfig:
    train_preprocessed: Path
    test_preprocessed: Path
    drug_lexicon: Path
    embeddings_dir: Path
    model_dir: Path
    reports_dir: Path
    figures_dir: Path
    submission_file: Path
    presentation_file: Path


@dataclass(frozen=True)
class CVConfig:
    n_splits: int
    shuffle: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    target_token: str
    other_token: str
    fuzzy_threshold: int
    window_sizes: tuple[int, ...]
    natural_window_size: int
    max_window_words: int
    long_doc_chars: int


@dataclass(frozen=True)
class EmbeddingConfig:
    minilm_model: str
    batch_size: int
    max_length: int
    num_threads: int


@dataclass(frozen=True)
class Config:
    seed: int
    n_jobs: int
    data: DataConfig
    artifacts: ArtifactConfig
    cv: CVConfig
    preprocessing: PreprocessingConfig
    embeddings: EmbeddingConfig


def _resolve(path: str) -> Path:
    return PROJECT_ROOT / path


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = raw["data"]
    raw_dir = _resolve(data["raw_dir"])
    return Config(
        seed=raw["project"]["seed"],
        n_jobs=raw["project"]["n_jobs"],
        data=DataConfig(
            train_path=raw_dir / data["train_file"],
            test_path=raw_dir / data["test_file"],
            sample_submission_path=raw_dir / data["sample_submission_file"],
            expected_train_rows=data["expected_train_rows"],
            expected_test_rows=data["expected_test_rows"],
            id_col=data["id_col"],
            text_col=data["text_col"],
            drug_col=data["drug_col"],
            target_col=data["target_col"],
            label_names={int(k): v for k, v in data["labels"].items()},
        ),
        artifacts=ArtifactConfig(**{k: _resolve(v) for k, v in raw["artifacts"].items()}),
        cv=CVConfig(**raw["cv"]),
        preprocessing=PreprocessingConfig(
            **{**raw["preprocessing"], "window_sizes": tuple(raw["preprocessing"]["window_sizes"])}
        ),
        embeddings=EmbeddingConfig(**raw["embeddings"]),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def set_seed(seed: int) -> None:
    """Seed every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
