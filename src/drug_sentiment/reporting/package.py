"""Stage `package`: zip the deliverables the capstone asks for.

Included: the end-to-end notebook, the two working notebooks, the whole source package and its config, the
presentation, predictions.csv (also at the root of the zip, where a grader looks first), the report tables and the
figures. Excluded: the raw competition data, the pickled models and embeddings, logs and caches -- everything that
is either not redistributable or rebuildable with `uv run python main.py all`.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from drug_sentiment.config import PROJECT_ROOT, Config, get_config
from drug_sentiment.logger import get_logger

logger = get_logger(__name__)

ARCHIVE_NAME = "Drug_Sentiment_Capstone_submission.zip"
SKIP_DIRECTORIES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}

FILES = [
    "Drug_Sentiment_Capstone.ipynb",
    "NoteBook/01_EDA.ipynb",
    "NoteBook/model_expriments.ipynb",
    "main.py",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "docs/viva_prep.md",
]
DIRECTORIES = [
    ("src", "*.py"),
    ("config", "*.yaml"),
    ("tests", "*.py"),
    ("artifacts/reports", "*.csv"),
    ("artifacts/reports/figures", "*.png"),
]


def collect(cfg: Config) -> list[tuple[Path, str]]:
    """(file on disk, name inside the zip) for everything that goes in."""
    items: list[tuple[Path, str]] = []
    for name in FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            items.append((path, name))
    for directory, pattern in DIRECTORIES:
        for path in sorted((PROJECT_ROOT / directory).rglob(pattern)):
            if not SKIP_DIRECTORIES.intersection(path.parts):
                items.append((path, str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")))
    for path, name in ((cfg.artifacts.submission_file, "predictions.csv"),
                       (cfg.artifacts.presentation_file, cfg.artifacts.presentation_file.name)):
        if path.exists():
            items.append((path, name))
    return items


def build_archive(cfg: Config | None = None) -> Path:
    cfg = cfg or get_config()
    items = collect(cfg)
    archive = PROJECT_ROOT / "submission" / ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path, name in items:
            bundle.write(path, name)

    missing = [name for name in ("Drug_Sentiment_Capstone.ipynb", "predictions.csv",
                                cfg.artifacts.presentation_file.name) if name not in {n for _, n in items}]
    if missing:
        logger.warning("Not in the archive yet: %s", ", ".join(missing))
    logger.info("Wrote %s: %d files, %.1f MB", archive, len(items), archive.stat().st_size / 1e6)
    return archive
