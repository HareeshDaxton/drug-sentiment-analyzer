"""Persistence helpers so every artifact is written the same way (parent folders created automatically)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_pickle(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_pickle(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}. Run the pipeline stage that produces it first.")
    return joblib.load(path)


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
