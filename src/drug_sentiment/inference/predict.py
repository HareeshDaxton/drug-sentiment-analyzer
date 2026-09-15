"""Predict the sentiment toward a drug in text that is not in the competition CSVs.

`main.py predict` scores the whole test file from the preprocessed pickles. This module is the other half: it takes
a raw (drug, comment) pair -- typed on the command line, or a column of any CSV -- and runs it through exactly the
same steps as training, in the same order:

    minimal cleaning -> drug lexicon lookup -> drug-centred window with the drug names masked
    -> handcrafted and VADER features -> MiniLM embedding of the window -> the fitted pipeline

Nothing here is fitted: the drug lexicon, the sentence encoder and the model pipeline are all loaded from
`artifacts/`, so a prediction for one comment uses the same vocabulary, scaling and weights as the submission.

    uv run python -m drug_sentiment.inference.predict --drug Humira --text "Humira cleared my psoriasis in weeks."
    uv run python -m drug_sentiment.inference.predict --csv comments.csv --out scored.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from drug_sentiment.config import Config, get_config
from drug_sentiment.features.embeddings import FrozenEncoder, build_minilm
from drug_sentiment.logger import get_logger
from drug_sentiment.models.train import MODEL_FILE, class_scores, load_model_config
from drug_sentiment.preprocessing.build import preprocess_row
from drug_sentiment.preprocessing.drug_context import DrugContextExtractor
from drug_sentiment.utils.io import load_pickle

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One scored (drug, comment) pair, with the evidence the model actually saw."""

    drug: str
    sentiment: int
    label: str
    scores: dict[str, float]
    match_type: str
    mention_count: int
    window: str

    def __str__(self) -> str:
        scores = ", ".join(f"{name} {value:.2f}" for name, value in self.scores.items())
        return (f"{self.drug}: {self.label} ({self.sentiment})\n"
                f"  scores      {scores}\n"
                f"  drug match  {self.match_type}, mentioned {self.mention_count}x\n"
                f"  window      {self.window[:200]}")


class DrugSentimentPredictor:
    """The trained pipeline plus the row-wise preprocessing it expects. Build it with `from_artifacts()`."""

    def __init__(self, pipeline, extractor: DrugContextExtractor, encoder: FrozenEncoder,
                 cfg: Config | None = None, text_column: str = "ctx_k0") -> None:
        self.cfg = cfg or get_config()
        self.pipeline = pipeline
        self.extractor = extractor
        self.encoder = encoder
        self.text_column = text_column  # the window the model reads, so a Prediction shows the text it judged
        self.analyzer = SentimentIntensityAnalyzer()

    @classmethod
    def from_artifacts(cls, cfg: Config | None = None) -> DrugSentimentPredictor:
        cfg = cfg or get_config()
        p = cfg.preprocessing
        lexicon = load_pickle(cfg.artifacts.drug_lexicon)
        extractor = DrugContextExtractor(lexicon, p.target_token, p.other_token, p.window_sizes,
                                         p.natural_window_size, p.max_window_words)
        return cls(load_pickle(cfg.artifacts.model_dir / MODEL_FILE), extractor, build_minilm(cfg.embeddings), cfg,
                   text_column=load_model_config(cfg)["text_column"])

    def build_frame(self, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        """The model-ready row per (drug, text) pair: the preprocess stage's columns plus the MiniLM columns."""
        rows = [
            preprocess_row(text, drug, self.extractor, self.analyzer, self.cfg.preprocessing.long_doc_chars)
            for drug, text in pairs
        ]
        frame = pd.DataFrame(rows)
        frame.insert(0, "drug", [drug for drug, _ in pairs])
        embeddings = self.encoder.encode(frame["ctx_natural"].tolist())
        return pd.concat([frame, pd.DataFrame(embeddings, columns=self.encoder.columns)], axis=1)

    def predict(self, pairs: list[tuple[str, str]]) -> list[Prediction]:
        frame = self.build_frame(pairs)
        labels = self.pipeline.predict(frame).astype(int)
        scores = class_scores(self.pipeline, frame)
        names = self.cfg.data.label_names
        return [
            Prediction(
                drug=row.drug,
                sentiment=int(label),
                label=names[int(label)],
                scores={names[position]: float(value) for position, value in enumerate(row_scores)},
                match_type=row.match_type,
                mention_count=int(row.mention_count),
                window=getattr(row, self.text_column),
            )
            for row, label, row_scores in zip(frame.itertuples(), labels, scores)
        ]

    def predict_one(self, drug: str, text: str) -> Prediction:
        return self.predict([(drug, text)])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drug", help="drug the sentiment is about")
    parser.add_argument("--text", help="the comment to score")
    parser.add_argument("--csv", help="CSV of comments to score instead of a single pair")
    parser.add_argument("--drug-col", default="drug")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--out", help="where to write the scored CSV (default: print the first rows)")
    args = parser.parse_args()

    predictor = DrugSentimentPredictor.from_artifacts()
    if args.csv:
        frame = pd.read_csv(args.csv)
        predictions = predictor.predict(list(zip(frame[args.drug_col], frame[args.text_col])))
        scored = frame.assign(
            sentiment=[prediction.sentiment for prediction in predictions],
            sentiment_label=[prediction.label for prediction in predictions],
        )
        if args.out:
            scored.to_csv(args.out, index=False)
            logger.info("Wrote %s: %d scored rows", args.out, len(scored))
        else:
            print(scored[[args.drug_col, "sentiment", "sentiment_label"]].head(20).to_string(index=False))
        return

    if not (args.drug and args.text):
        parser.error("give --drug and --text, or --csv")
    print(predictor.predict_one(args.drug, args.text))


if __name__ == "__main__":
    main()
