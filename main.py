"""Pipeline entry point. Run one stage, e.g. `uv run python main.py ingest`, or every stage with `all`."""

import argparse
from collections.abc import Callable

from drug_sentiment.config import get_config, set_seed
from drug_sentiment.logger import get_logger

logger = get_logger("main")


# Stage functions import their modules lazily so a light stage never pays for heavy imports (e.g. torch).
def run_ingest() -> None:
    from drug_sentiment.data.ingestion import load_raw_data

    load_raw_data()


def run_preprocess() -> None:
    from drug_sentiment.preprocessing.build import run_preprocessing

    run_preprocessing()


def run_embed() -> None:
    from drug_sentiment.features.embeddings import run_embeddings

    run_embeddings()


def run_train() -> None:
    from drug_sentiment.models.train import run_training

    run_training()


def run_evaluate() -> None:
    from drug_sentiment.models.evaluate import run_evaluation

    run_evaluation()


def run_window_search() -> None:
    from drug_sentiment.models.evaluate import run_window_search as search

    search()


def run_predict() -> None:
    from drug_sentiment.models.train import run_prediction

    run_prediction()


def run_slides() -> None:
    from drug_sentiment.reporting.build_ppt import build_deck

    build_deck()


def run_package() -> None:
    from drug_sentiment.reporting.package import build_archive

    build_archive()


STAGES: dict[str, Callable[[], None]] = {
    "ingest": run_ingest,
    "preprocess": run_preprocess,
    "embed": run_embed,
    "train": run_train,
    "evaluate": run_evaluate,
    "window-search": run_window_search,
    "predict": run_predict,
    "slides": run_slides,
    "package": run_package,
}

# `all` rebuilds the submission from the raw CSVs; the deck and the zip are built on demand, after it.
PIPELINE_STAGES = ["ingest", "preprocess", "embed", "train", "evaluate", "predict"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    args = parser.parse_args()

    set_seed(get_config().seed)
    for name in PIPELINE_STAGES if args.stage == "all" else [args.stage]:
        logger.info("===== Stage: %s =====", name)
        STAGES[name]()


if __name__ == "__main__":
    main()
