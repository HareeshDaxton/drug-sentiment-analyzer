"""Stage `evaluate`: how good the main model is, where it fails, and which design decisions earned their keep.

Everything here is measured out of fold on train.csv, over the same 5 folds (stratified by label, grouped by comment
text) that the experiments notebook used, so the numbers are comparable with the model-selection tables and
`test.csv` is never touched.

Three things are produced:
1. **Scores** - out-of-fold classification report and confusion matrix for the main model.
2. **Error analysis** - weighted F1 per slice of the data (comment length, where the drug is first mentioned, how
   often it is mentioned, how many other drugs share the comment, how the drug name was matched), a per-drug table
   and a sample of misclassified rows to read.
3. **Ablations** - what this project's two design decisions are worth, each re-cross-validated end to end: the
   drug-centred window (against the whole comment, drug-blind or masked) and the feature blocks stacked on the
   TF-IDF, plus the class weighting that handles the 72% neutral majority.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, recall_score,
                             roc_auc_score)

from drug_sentiment.config import Config, get_config
from drug_sentiment.logger import get_logger
from drug_sentiment.models.features import FEATURE_BLOCKS, load_model_frame
from drug_sentiment.models.train import MODEL_FILE, build_pipeline, class_scores, load_model_config
from drug_sentiment.utils.io import load_pickle
from drug_sentiment.visualization import model_plots
from drug_sentiment.visualization.style import apply_style, save_figure

logger = get_logger(__name__)

OOF_FILE = "oof_predictions.csv"
REPORT_FILE = "oof_classification_report.csv"
SLICES_FILE = "oof_slice_scores.csv"
DRUGS_FILE = "oof_drug_scores.csv"
ERRORS_FILE = "oof_error_examples.csv"
WORDS_FILE = "model_top_words.csv"
ABLATION_FILE = "ablation_scores.csv"

SHIPPED = "window k=0 + all features"  # the label the shipped setting carries in every ablation group
WINDOW_SEARCH_FILE = "window_c_search.csv"
WINDOW_COLUMNS = ("ctx_k0", "ctx_k1", "ctx_k2")
C_GRID = (0.03, 0.05, 0.0847, 0.15, 0.3)


def score_predictions(y: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y, predicted),
        "f1_weighted": f1_score(y, predicted, average="weighted"),
        "f1_macro": f1_score(y, predicted, average="macro"),
        "recall_macro": recall_score(y, predicted, average="macro"),
        "roc_auc_ovr": roc_auc_score(y, scores, multi_class="ovr", average="weighted"),
    }


def cross_validate(config: dict, frame: pd.DataFrame, y: np.ndarray, cfg: Config,
                   text_column: str | None = None, blocks: tuple[str, ...] = FEATURE_BLOCKS,
                   params: dict | None = None) -> dict:
    """Out-of-fold predictions over the stored folds. The feature transformer is fitted inside every fold, so a
    validation row never influences the vocabulary, the scaling or the drug encoding it is then scored with."""
    variant = dict(config, text_column=text_column or config["text_column"])
    if params:
        variant = dict(variant, params={**config["params"], **params})
    template = build_pipeline(variant, cfg.seed, blocks=tuple(blocks))

    folds = frame["fold"].to_numpy()
    predicted = np.zeros(len(y), dtype=int)
    scores = np.zeros((len(y), 3))
    fold_f1, start = [], time.perf_counter()
    for fold in np.unique(folds):
        train_idx, val_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        pipeline = clone(template).fit(frame.iloc[train_idx], y[train_idx])
        predicted[val_idx] = pipeline.predict(frame.iloc[val_idx])
        scores[val_idx] = class_scores(pipeline, frame.iloc[val_idx])
        fold_f1.append(f1_score(y[val_idx], predicted[val_idx], average="weighted"))
    return {
        "predicted": predicted,
        "scores": scores,
        "fold_f1_std": float(np.std(fold_f1)),
        "seconds": time.perf_counter() - start,
        **score_predictions(y, predicted, scores),
    }


def _slice_labels(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The row groups the error analysis reports on, each as one labelled series."""
    shared = frame["text"].duplicated(keep=False)
    return {
        "comment length": pd.cut(frame["char_count"], [0, 500, 2_000, 8_000, np.inf],
                                 labels=["< 500 chars", "500-2k", "2k-8k", "> 8k"]),
        # first_mention_pos is -1 when the drug was never found, otherwise the relative position of the first
        # mention; the first bin must exclude 0.0, which means "the comment opens with the drug name".
        "first drug mention": pd.cut(frame["first_mention_pos"], [-1.01, -0.0001, 0.1, 0.5, 1.01],
                                     labels=["not found", "first 10%", "10-50%", "after half"]),
        "times the drug is mentioned": pd.cut(frame["mention_count"], [-1, 1, 3, np.inf],
                                              labels=["once", "2-3 times", "4+ times"]),
        "other drugs in the comment": pd.cut(frame["n_other_drugs"], [-1, 0, 2, np.inf],
                                             labels=["none", "1-2", "3+"]),
        "how the drug name matched": frame["match_type"],
        "comment shared with another drug": shared.map({True: "shared", False: "one drug only"}),
    }


def slice_scores(frame: pd.DataFrame, y: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    rows = []
    for dimension, labels in _slice_labels(frame).items():
        for value, position in labels.groupby(labels, observed=True).indices.items():
            rows.append({
                "dimension": dimension,
                "slice": str(value),
                "rows": len(position),
                "share": len(position) / len(frame),
                "f1_weighted": f1_score(y[position], predicted[position], average="weighted"),
                "accuracy": accuracy_score(y[position], predicted[position]),
            })
    return pd.DataFrame(rows)


def drug_scores(frame: pd.DataFrame, y: np.ndarray, predicted: np.ndarray, min_rows: int = 20) -> pd.DataFrame:
    """Per-drug weighted F1 for the drugs with enough rows to mean something, worst first."""
    rows = []
    for drug, position in frame.groupby("drug_norm").indices.items():
        if len(position) < min_rows:
            continue
        rows.append({
            "drug": drug,
            "rows": len(position),
            "majority_class_share": pd.Series(y[position]).value_counts(normalize=True).max(),
            "f1_weighted": f1_score(y[position], predicted[position], average="weighted"),
        })
    return pd.DataFrame(rows).sort_values("f1_weighted")


def error_examples(frame: pd.DataFrame, y: np.ndarray, predicted: np.ndarray, cfg: Config,
                   text_column: str = "ctx_k1", n: int = 40, seed: int = 42) -> pd.DataFrame:
    """A readable sample of the mistakes: what the model saw and what it should have said."""
    wrong = np.flatnonzero(y != predicted)
    sample = np.sort(np.random.default_rng(seed).choice(wrong, size=min(n, len(wrong)), replace=False))
    names = cfg.data.label_names
    return pd.DataFrame({
        "unique_hash": frame["unique_hash"].to_numpy()[sample],
        "drug": frame["drug"].to_numpy()[sample],
        "true": [names[label] for label in y[sample]],
        "predicted": [names[label] for label in predicted[sample]],
        "mention_count": frame["mention_count"].to_numpy()[sample],
        "vader_window_compound": frame["vader_window_compound"].to_numpy()[sample].round(3),
        "window": [text[:400] for text in frame[text_column].to_numpy()[sample]],
    })


def top_words(pipeline, cfg: Config, n: int = 12) -> pd.DataFrame:
    """The words the fitted linear model leans on per class: one weight per TF-IDF word feature, largest first."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["sentiment", "word", "weight"])
    names = np.asarray(pipeline.named_steps["features"].get_feature_names_out(), dtype=str)
    is_word = np.char.startswith(names, "text__word__")
    words = np.char.replace(names[is_word], "text__word__", "")
    rows = []
    for position, label in enumerate(model.classes_):
        weights = np.asarray(model.coef_)[position][is_word]
        for index in np.argsort(weights)[::-1][:n]:
            rows.append({"sentiment": cfg.data.label_names[int(label)], "word": str(words[index]),
                         "weight": float(weights[index])})
    return pd.DataFrame(rows)


def ablation_variants() -> list[dict]:
    """Each entry re-runs the whole 5-fold cross-validation with exactly one design decision changed."""
    handcrafted = ("text", "counts", "scaled")
    return [
        {"group": "text given to the model", "name": "whole comment, drug-blind", "text_column": "text_clean"},
        {"group": "text given to the model", "name": "whole comment, drugs masked", "text_column": "full_masked"},
        {"group": "text given to the model", "name": SHIPPED, "text_column": "ctx_k0"},
        {"group": "text given to the model", "name": "window k=1 (+/- 1 sentence)", "text_column": "ctx_k1"},
        {"group": "text given to the model", "name": "window k=2 (+/- 2 sentences)", "text_column": "ctx_k2"},
        {"group": "features on top of TF-IDF", "name": "TF-IDF only", "blocks": ("text",)},
        {"group": "features on top of TF-IDF", "name": "+ handcrafted + VADER", "blocks": handcrafted},
        {"group": "features on top of TF-IDF", "name": "+ drug identity", "blocks": (*handcrafted, "categories")},
        {"group": "features on top of TF-IDF", "name": SHIPPED, "blocks": FEATURE_BLOCKS},
        {"group": "features on top of TF-IDF", "name": "all but the drug identity",
         "blocks": (*handcrafted, "minilm")},
        {"group": "handling the 72% neutral class", "name": "no class weights", "params": {"class_weight": None}},
        {"group": "handling the 72% neutral class", "name": SHIPPED, "params": {}},
    ]


def run_ablations(config: dict, frame: pd.DataFrame, y: np.ndarray, cfg: Config) -> pd.DataFrame:
    cache: dict[tuple, dict] = {}
    rows = []
    for variant in ablation_variants():
        key = (variant.get("text_column", config["text_column"]),
               tuple(variant.get("blocks", FEATURE_BLOCKS)),
               tuple(sorted((variant.get("params") or {}).items(), key=str)))
        if key not in cache:
            cache[key] = cross_validate(config, frame, y, cfg, text_column=variant.get("text_column"),
                                        blocks=variant.get("blocks", FEATURE_BLOCKS), params=variant.get("params"))
            logger.info("Ablation %-32s weighted F1 %.4f (%.0fs)", variant["name"], cache[key]["f1_weighted"],
                        cache[key]["seconds"])
        result = cache[key]
        rows.append({
            "group": variant["group"], "variant": variant["name"], "shipped": variant["name"] == SHIPPED,
            **{name: result[name] for name in ("f1_weighted", "f1_macro", "accuracy", "fold_f1_std", "seconds")},
        })
    return pd.DataFrame(rows)


def run_window_search(cfg: Config | None = None) -> pd.DataFrame:
    """Stage `window-search`: which drug window to ship.

    The first ablation showed the tightest window (mention sentences only) ahead of the k=1 window the experiments
    notebook used. That comparison was not quite fair -- C had been tuned on k=1 -- so this re-tunes C for each
    window on the same folds and reports the grid. It is what chose `ctx_k0` with C = 0.05 for the main model.
    """
    cfg = cfg or get_config()
    config = load_model_config(cfg)
    frame = load_model_frame("train", cfg)
    y = frame[cfg.data.target_col].to_numpy()

    rows = []
    for text_column in WINDOW_COLUMNS:
        for c in C_GRID:
            result = cross_validate(config, frame, y, cfg, text_column=text_column, params={"C": c})
            rows.append({"text_column": text_column, "C": c,
                         **{name: result[name] for name in ("f1_weighted", "f1_macro", "accuracy", "fold_f1_std",
                                                            "seconds")}})
            logger.info("%s C=%-7s weighted F1 %.4f  macro F1 %.4f  (%.0fs)", text_column, c,
                        result["f1_weighted"], result["f1_macro"], result["seconds"])

    table = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    table.to_csv(cfg.artifacts.reports_dir / WINDOW_SEARCH_FILE, index=False)
    best = table.iloc[0]
    logger.info("Best: %s with C=%s, weighted F1 %.4f", best["text_column"], best["C"], best["f1_weighted"])
    return table


def run_evaluation(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    apply_style()
    reports = cfg.artifacts.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    config = load_model_config(cfg)
    frame = load_model_frame("train", cfg)
    y = frame[cfg.data.target_col].to_numpy()
    label_names = [cfg.data.label_names[label] for label in (0, 1, 2)]

    result = cross_validate(config, frame, y, cfg)
    predicted = result["predicted"]
    logger.info("%s out of fold: weighted F1 %.4f (std %.4f across folds), macro F1 %.4f, accuracy %.4f",
                config["model"], result["f1_weighted"], result["fold_f1_std"], result["f1_macro"], result["accuracy"])

    pd.DataFrame({
        "unique_hash": frame["unique_hash"], "drug": frame["drug"], "fold": frame["fold"],
        "true": y, "predicted": predicted,
        **{f"score_{name}": result["scores"][:, label] for label, name in enumerate(label_names)},
    }).to_csv(reports / OOF_FILE, index=False)

    report = pd.DataFrame(classification_report(y, predicted, labels=[0, 1, 2], target_names=label_names,
                                                output_dict=True)).T
    report.to_csv(reports / REPORT_FILE)

    display_order = [1, 2, 0]  # negative, neutral, positive
    counts = confusion_matrix(y, predicted, labels=display_order)
    save_figure(model_plots.plot_confusion_matrix(
        counts, [cfg.data.label_names[label] for label in display_order],
        title=f"Where {config['model']} goes wrong",
        subtitle=f"Out-of-fold predictions on the {len(frame):,} training comments. Each row adds up to 100% of "
                 "that true class.",
    ), "eval_01_confusion_matrix")

    slices = slice_scores(frame, y, predicted)
    slices.to_csv(reports / SLICES_FILE, index=False)
    save_figure(model_plots.plot_slice_scores(
        slices, result["f1_weighted"],
        title="Which comments the model finds hard",
        subtitle="Out-of-fold weighted F1 per group of rows, with the number of rows in each. The line is the score "
                 "over all rows, so a bar left of it is a weak spot.",
    ), "eval_02_slice_scores")

    drugs = drug_scores(frame, y, predicted)
    drugs.to_csv(reports / DRUGS_FILE, index=False)
    error_examples(frame, y, predicted, cfg, config["text_column"]).to_csv(reports / ERRORS_FILE,
                                                                            index=False)

    pipeline = (load_pickle(cfg.artifacts.model_dir / MODEL_FILE)
                if (cfg.artifacts.model_dir / MODEL_FILE).exists()
                else build_pipeline(config, cfg.seed).fit(frame, y))
    words = top_words(pipeline, cfg)
    if not words.empty:
        words.to_csv(reports / WORDS_FILE, index=False)
        save_figure(model_plots.plot_top_words(
            words, title=f"What {config['model']} reads as sentiment",
            subtitle="Largest weight per class among the word features of the drug window. `targetdrug` is the "
                     "masked drug the row asks about, `otherdrug` any other drug in the same comment.",
        ), "eval_03_top_words")

    ablations = run_ablations(config, frame, y, cfg)
    ablations.to_csv(reports / ABLATION_FILE, index=False)
    save_figure(model_plots.plot_ablation(
        ablations, title="What each design decision is worth",
        subtitle="Weighted F1 of the same model, cross-validated on the same folds, with one decision changed at a "
                 "time. The filled bar in each group is the setting the project ships.",
    ), "eval_04_ablations")

    logger.info("Evaluation written to %s", reports)
    return ablations
