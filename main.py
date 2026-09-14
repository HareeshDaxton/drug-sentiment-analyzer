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


STAGES: dict[str, Callable[[], None]] = {
    "ingest": run_ingest,
    "preprocess": run_preprocess,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    args = parser.parse_args()

    set_seed(get_config().seed)
    for name in STAGES if args.stage == "all" else [args.stage]:
        logger.info("===== Stage: %s =====", name)
        STAGES[name]()


if __name__ == "__main__":
    main()
