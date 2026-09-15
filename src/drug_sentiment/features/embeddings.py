"""Stage `embed`: frozen MiniLM sentence embeddings of each row's drug window, computed on CPU.

sentence-transformers/all-MiniLM-L6-v2 is a small, fast sentence encoder (6 layers, about 22M parameters).
It reads the unmasked drug window (`ctx_natural` from the preprocess stage) and returns a 384-d unit-length
vector of what the window means, including wording TF-IDF cannot match ("didn't help" vs "was useless").

The model is used as it is, never trained on this data, so each row's embedding depends only on that row.
Embeddings are computed once for train and test and saved as pickles next to the preprocessed frames.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drug_sentiment.config import Config, EmbeddingConfig, get_config
from drug_sentiment.logger import get_logger
from drug_sentiment.utils.io import load_pickle, save_json, save_pickle

logger = get_logger(__name__)

TEXT_COLUMN = "ctx_natural"


@dataclass(frozen=True)
class FrozenEncoder:
    model_name: str
    revision: str | None
    columns: list[str]
    encode: Callable[[list[str]], np.ndarray]


def length_sorted_batches(texts: Sequence[str], batch_size: int) -> list[np.ndarray]:
    """Row indices in batches of similar text length, so each batch needs little padding (faster on CPU)."""
    order = np.argsort([len(text) for text in texts], kind="stable")
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def encode_in_batches(texts: Sequence[str], encode: Callable[[list[str]], np.ndarray],
                      batch_size: int, desc: str) -> np.ndarray:
    """Encode texts in length-sorted batches; returns one float32 row per text, in the original order."""
    batches = length_sorted_batches(texts, batch_size)
    output: np.ndarray | None = None
    start = time.perf_counter()
    for n, rows in enumerate(batches, start=1):
        values = encode([texts[i] for i in rows])
        if output is None:
            output = np.empty((len(texts), values.shape[1]), dtype=np.float32)
        output[rows] = values
        if n % 50 == 0 or n == len(batches):
            logger.info("%s: %d/%d batches, %.0fs", desc, n, len(batches), time.perf_counter() - start)
    if output is None:
        raise ValueError("no texts to encode")
    return output


def build_minilm(cfg: EmbeddingConfig) -> FrozenEncoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.minilm_model, device="cpu")
    model.max_seq_length = cfg.max_length

    def encode(batch: list[str]) -> np.ndarray:
        return model.encode(batch, batch_size=len(batch), normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)

    transformer = model[0]
    hf_model = getattr(transformer, "auto_model", None) or getattr(transformer, "model", None)
    revision = getattr(getattr(hf_model, "config", None), "_commit_hash", None)
    dimensions = encode(["dimension probe"]).shape[1]
    return FrozenEncoder(cfg.minilm_model, revision, [f"minilm_{i:03d}" for i in range(dimensions)], encode)


def embedding_path(embeddings_dir: Path, split: str) -> Path:
    return embeddings_dir / f"minilm_{split}.pkl"


def load_embeddings(split: str, unique_hash: Sequence[str], embeddings_dir: Path | None = None) -> pd.DataFrame:
    """Saved MiniLM features for a split ("train" or "test") in the order of `unique_hash`."""
    embeddings_dir = embeddings_dir or get_config().artifacts.embeddings_dir
    features = load_pickle(embedding_path(embeddings_dir, split))
    missing = pd.Index(unique_hash).difference(features.index)
    if len(missing):
        raise ValueError(f"minilm/{split}: {len(missing)} rows have no saved embedding; rerun `main.py embed`")
    return features.loc[list(unique_hash)]


def run_embeddings(cfg: Config | None = None) -> None:
    import torch

    cfg = cfg or get_config()
    torch.set_num_threads(cfg.embeddings.num_threads)
    artifacts = cfg.artifacts
    encoder = build_minilm(cfg.embeddings)
    info: dict[str, object] = {
        "model": encoder.model_name,
        "revision": encoder.revision,
        "text_column": TEXT_COLUMN,
        "max_length": cfg.embeddings.max_length,
        "features": len(encoder.columns),
        "seconds": {},
    }
    for split, path in (("train", artifacts.train_preprocessed), ("test", artifacts.test_preprocessed)):
        frame = load_pickle(path)
        start = time.perf_counter()
        values = encode_in_batches(frame[TEXT_COLUMN].tolist(), encoder.encode, cfg.embeddings.batch_size,
                                   f"minilm/{split}")
        features = pd.DataFrame(values, columns=encoder.columns,
                                index=pd.Index(frame["unique_hash"], name="unique_hash"))
        save_pickle(features, embedding_path(artifacts.embeddings_dir, split))
        info["seconds"][split] = round(time.perf_counter() - start)
        logger.info("minilm/%s: saved %s features in %ds", split, features.shape, info["seconds"][split])
    save_json(info, artifacts.embeddings_dir / "embedding_info.json")
