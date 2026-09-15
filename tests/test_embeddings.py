import numpy as np
import pandas as pd
import pytest

from drug_sentiment.features.embeddings import (
    embedding_path,
    encode_in_batches,
    length_sorted_batches,
    load_embeddings,
)
from drug_sentiment.utils.io import save_pickle

TEXTS = ["ccc", "a", "bbbb", "dd", "e"]


def test_batches_cover_every_row_once_in_length_order():
    batches = length_sorted_batches(TEXTS, batch_size=2)
    flat = np.concatenate(batches)

    assert sorted(flat.tolist()) == list(range(len(TEXTS)))
    assert [len(TEXTS[i]) for i in flat] == sorted(len(text) for text in TEXTS)
    assert max(len(batch) for batch in batches) == 2


def test_encoded_rows_come_back_in_the_original_order():
    def fake_encode(batch):
        return np.array([[len(text), ord(text[0])] for text in batch])

    values = encode_in_batches(TEXTS, fake_encode, batch_size=2, desc="test")

    assert values.dtype == np.float32
    assert values[:, 0].tolist() == [3, 1, 4, 2, 1]
    assert values[:, 1].tolist() == [ord("c"), ord("a"), ord("b"), ord("d"), ord("e")]


def test_load_embeddings_follows_the_requested_row_order(tmp_path):
    saved = pd.DataFrame({"f0": [1.0, 2.0, 3.0]}, index=pd.Index(["x", "y", "z"], name="unique_hash"))
    save_pickle(saved, embedding_path(tmp_path, "train"))

    loaded = load_embeddings("train", ["z", "x", "y"], embeddings_dir=tmp_path)
    assert loaded["f0"].tolist() == [3.0, 1.0, 2.0]

    with pytest.raises(ValueError):
        load_embeddings("train", ["x", "missing"], embeddings_dir=tmp_path)
