"""Model-evaluation figures: model comparison dot plot and confusion matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from drug_sentiment.visualization.style import (
    BASELINE,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    LABEL_ORDER,
    SENTIMENT_COLORS,
    SERIES_COLOR,
    SURFACE,
    finish_figure,
    ink_on,
)

# Sequential blue ramp (steps 100-700 of the documented palette), for magnitudes such as confusion-matrix shares.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot_model_comparison(table: pd.DataFrame, baseline_f1: float, title: str, subtitle: str) -> Figure:
    """Validation (out-of-fold) weighted F1 per model as filled dots and training-fold F1 as hollow dots.
    `table` is indexed by model name with columns cv_f1_weighted and train_f1_weighted."""
    table = table.sort_values("cv_f1_weighted")
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.42 * len(table)))
    ax.hlines(y, table["cv_f1_weighted"], table["train_f1_weighted"], color=BASELINE, linewidth=2, zorder=2)
    ax.scatter(table["train_f1_weighted"], y, s=64, facecolors=SURFACE, edgecolors=SERIES_COLOR, linewidths=2,
               zorder=3, label="training folds")
    ax.scatter(table["cv_f1_weighted"], y, s=64, color=SERIES_COLOR, edgecolors=SURFACE, linewidths=2,
               zorder=3, label="validation folds (out-of-fold)")
    for row, value in zip(y, table["cv_f1_weighted"]):
        ax.annotate(f"{value:.3f}", (value, row), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK_PRIMARY)
    ax.axvline(baseline_f1, color=INK_SECONDARY, linewidth=1, zorder=1)
    ax.annotate(f"always neutral {baseline_f1:.3f}", (baseline_f1, -0.6), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(y, table.index)
    ax.set_ylim(-1, len(table) - 0.4)
    ax.set_xlim(left=table["cv_f1_weighted"].min() - 0.07)  # room for the value label of the lowest bar
    ax.set_xlabel("Weighted F1")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    return finish_figure(fig, title, subtitle)


def plot_confusion_matrix(counts: np.ndarray, labels: list[str], title: str, subtitle: str) -> Figure:
    """Rows are true classes; each cell shows its share of the true class (recall on the diagonal) and the count."""
    shares = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    image = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            fill = to_hex(cmap(shares[i, j]))
            ax.text(j, i, f"{shares[i, j]:.0%}\n{counts[i, j]:,}", ha="center", va="center", fontsize=9.5,
                    color=ink_on(fill))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # surface-coloured gap between cells
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, format=PercentFormatter(xmax=1, decimals=0))
    colorbar.outline.set_visible(False)
    return finish_figure(fig, title, subtitle)


def _grouped_bars(ax, rows: list[dict], value_range: tuple[float, float]) -> None:
    """Horizontal bars with an unfilled header row per group, so group names read as a left-hand hierarchy."""
    y = np.arange(len(rows))[::-1]
    for row, position in zip(rows, y):
        if row["header"]:
            continue
        ax.barh(position, row["value"], height=0.62, color=row["color"], zorder=2)
        ax.annotate(f"{row['value']:.3f}", (row["value"], position), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_PRIMARY)
        if row["note"]:
            ax.annotate(row["note"], (value_range[0], position), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=ink_on(row["color"]))
    ax.set_yticks(y, [row["label"] for row in rows])
    for label, row in zip(ax.get_yticklabels(), rows):
        if row["header"]:
            label.set_color(INK_PRIMARY)
            label.set_fontweight("semibold")
    ax.set_xlim(*value_range)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("Weighted F1")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def plot_slice_scores(table: pd.DataFrame, overall_f1: float, title: str, subtitle: str) -> Figure:
    """Out-of-fold weighted F1 per slice of the data, grouped by the question each slice answers."""
    rows = []
    for dimension, group in table.groupby("dimension", sort=False):
        rows.append({"label": dimension, "header": True, "value": None, "color": None, "note": None})
        for _, slice_row in group.iterrows():
            rows.append({"label": f"   {slice_row['slice']}", "header": False, "value": slice_row["f1_weighted"],
                         "color": SERIES_COLOR, "note": f"{int(slice_row['rows']):,} rows"})

    low = max(0.0, min(row["value"] for row in rows if not row["header"]) - 0.12)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.26 * len(rows)))
    _grouped_bars(ax, rows, (low, 1.0))
    ax.axvline(overall_f1, color=INK_SECONDARY, linewidth=1, zorder=3)
    ax.annotate(f"all rows {overall_f1:.3f}", (overall_f1, len(rows) - 0.6), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK_SECONDARY)
    return finish_figure(fig, title, subtitle)


def plot_ablation(table: pd.DataFrame, title: str, subtitle: str) -> Figure:
    """One bar per variant, grouped by the decision it changes; the shipped setting is the filled one."""
    rows = []
    for group_name, group in table.groupby("group", sort=False):
        shipped_f1 = group.loc[group["shipped"], "f1_weighted"].iloc[0]
        rows.append({"label": group_name, "header": True, "value": None, "color": None, "note": None})
        for _, variant in group.iterrows():
            delta = variant["f1_weighted"] - shipped_f1
            rows.append({
                "label": f"   {variant['variant']}", "header": False, "value": variant["f1_weighted"],
                "color": SERIES_COLOR if variant["shipped"] else BASELINE,
                "note": "shipped" if variant["shipped"] else f"{delta:+.3f}",
            })

    values = [row["value"] for row in rows if not row["header"]]
    low = max(0.0, min(values) - 0.08)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.3 * len(rows)))
    _grouped_bars(ax, rows, (low, max(values) + 0.03))
    return finish_figure(fig, title, subtitle)


def plot_top_words(words: pd.DataFrame, title: str, subtitle: str) -> Figure:
    """The strongest word weights of a linear model, one panel per class."""
    classes = [name for name in LABEL_ORDER if name in set(words["sentiment"])]
    fig, axes = plt.subplots(1, len(classes), figsize=(8.4, 0.5 + 0.28 * words.groupby("sentiment").size().max()))
    for ax, sentiment in zip(np.atleast_1d(axes), classes):
        group = words[words["sentiment"] == sentiment].sort_values("weight")
        y = np.arange(len(group))
        ax.barh(y, group["weight"], height=0.7, color=SENTIMENT_COLORS[sentiment], zorder=2)
        ax.set_yticks(y, group["word"])
        ax.set_title(sentiment, color=SENTIMENT_COLORS[sentiment], fontweight="semibold")
        ax.set_xlabel("weight")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
    return finish_figure(fig, title, subtitle)
