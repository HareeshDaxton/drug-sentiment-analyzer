"""EDA figures. Every function returns a Figure; the caller supplies the title and a one-line takeaway."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

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

BAR_HEIGHT = 0.6
GAP_LINEWIDTH = 0.8  # surface-coloured edge: a thin gap between touching fills, not an ink border
THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")


def _value_grid(ax: Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def _stacked_label_share(ax: Axes, share: pd.DataFrame, min_label_pct: float = 6.0) -> None:
    """100% stacked bars ordered negative -> neutral -> positive; polar shares labelled where they fit."""
    y = np.arange(len(share))
    left = np.zeros(len(share))
    for label in LABEL_ORDER:
        widths = share[label].to_numpy() * 100
        color = SENTIMENT_COLORS[label]
        ax.barh(y, widths, left=left, height=BAR_HEIGHT, color=color,
                edgecolor=SURFACE, linewidth=GAP_LINEWIDTH, label=label)
        if label != "neutral":
            for row, (start, width) in enumerate(zip(left, widths)):
                if width >= min_label_pct:
                    ax.text(start + width / 2, row, f"{width:.0f}%", ha="center", va="center",
                            fontsize=8.5, color=ink_on(color))
        left += widths
    ax.set_yticks(y, [f"{name}  ·  n={n:,}" for name, n in zip(share.index, share["n"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.spines["left"].set_visible(False)


def _class_boxes(ax: Axes, groups: dict[str, pd.Series]) -> None:
    """One horizontal box per class: box = interquartile range, whiskers = 5th-95th percentile, no outliers."""
    parts = ax.boxplot(
        [groups[label].to_numpy() for label in LABEL_ORDER],
        orientation="horizontal", widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
    )
    for i, label in enumerate(LABEL_ORDER):
        color = SENTIMENT_COLORS[label]
        parts["boxes"][i].set(facecolor=to_rgba(color, 0.12), edgecolor=color, linewidth=1.5)
        parts["medians"][i].set(color=color, linewidth=2.5)
        for line in parts["whiskers"][2 * i : 2 * i + 2] + parts["caps"][2 * i : 2 * i + 2]:
            line.set(color=color, linewidth=1.5)
    ax.set_yticks(range(1, len(LABEL_ORDER) + 1), LABEL_ORDER)
    ax.invert_yaxis()


def plot_class_distribution(labels: pd.Series, title: str, subtitle: str) -> Figure:
    counts = labels.value_counts().reindex(LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    y = np.arange(len(counts))
    ax.barh(y, counts, height=BAR_HEIGHT, color=[SENTIMENT_COLORS[label] for label in counts.index])
    for row, count in enumerate(counts):
        ax.text(count + counts.max() * 0.01, row, f"{count:,}  ·  {count / counts.sum():.1%}",
                va="center", color=INK_PRIMARY)
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.2)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Rows in train.csv")
    _value_grid(ax)
    return finish_figure(fig, title, subtitle)


def plot_drug_frequency(train_drugs: pd.Series, test_drugs: pd.Series, title: str, subtitle: str,
                        top_n: int = 20) -> Figure:
    """Dot plot of each file's share of rows for the most frequent train drugs."""
    train_share = train_drugs.value_counts(normalize=True) * 100
    test_share = test_drugs.value_counts(normalize=True).reindex(train_share.index, fill_value=0) * 100
    top = train_share.index[:top_n]
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(8, 6.0))
    ax.hlines(y, np.minimum(train_share[top], test_share[top]), np.maximum(train_share[top], test_share[top]),
              color=BASELINE, linewidth=2, zorder=2)
    ax.scatter(train_share[top], y, s=64, color=SERIES_COLOR, edgecolors=SURFACE, linewidths=2, zorder=3,
               label="train")
    ax.scatter(test_share[top], y, s=64, facecolors=SURFACE, edgecolors=SERIES_COLOR, linewidths=2, zorder=3,
               label="test")
    ax.set_yticks(y, top)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Share of the file's rows")
    ax.spines["left"].set_visible(False)
    _value_grid(ax)
    ax.legend(loc="lower right")
    return finish_figure(fig, title, subtitle)


def plot_label_share(share: pd.DataFrame, title: str, subtitle: str, xlabel: str = "Share of rows") -> Figure:
    """Label mix per group; `share` holds negative/neutral/positive shares and group size `n`."""
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.42 * len(share)))
    _stacked_label_share(ax, share)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=0.9, handleheight=0.9,
              borderaxespad=0.2)
    return finish_figure(fig, title, subtitle)


def plot_length_by_class(word_counts: dict[str, pd.Series], quartile_share: pd.DataFrame,
                         title: str, subtitle: str) -> Figure:
    fig, (ax_box, ax_share) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})

    _class_boxes(ax_box, word_counts)
    ax_box.set_xscale("log")
    ax_box.xaxis.set_major_formatter(THOUSANDS)
    ax_box.xaxis.set_minor_formatter(NullFormatter())
    ax_box.set_xlabel("Words per comment (log scale)")
    ax_box.set_title("Length by class")
    ax_box.spines["left"].set_visible(False)
    _value_grid(ax_box)

    _stacked_label_share(ax_share, quartile_share)
    ax_share.set_title("Label mix by length quartile")
    ax_share.set_xlabel("Share of rows")
    ax_share.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, handlelength=0.9, handleheight=0.9)
    return finish_figure(fig, title, subtitle)


def plot_first_mention_position(first_mention_word: pd.Series, cutoff_words: int, title: str, subtitle: str,
                                max_words: int = 1000) -> Figure:
    """Share of rows (where the drug is found) whose first mention comes after each word position."""
    found = np.sort(first_mention_word[first_mention_word >= 0].to_numpy())
    positions = np.arange(max_words + 1)
    share_after = 100 * (1 - np.searchsorted(found, positions, side="right") / len(found))
    median_word = float(np.median(found))

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(positions, share_after, color=SERIES_COLOR, alpha=0.1, linewidth=0)
    ax.plot(positions, share_after, color=SERIES_COLOR, linewidth=2, solid_capstyle="round")
    points = (
        (median_word, 50.0, f"half of rows name the drug by word {median_word:.0f}"),
        (cutoff_words, share_after[cutoff_words],
         f"{share_after[cutoff_words]:.1f}% name it only after word {cutoff_words} (≈ 512 tokens)"),
    )
    for x, y, text in points:
        ax.scatter([x], [y], s=64, color=SERIES_COLOR, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(text, (x, y), xytext=(10, 10), textcoords="offset points", color=INK_PRIMARY)
    ax.set_xlim(0, max_words)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Word position in the comment")
    ax.set_ylabel("Rows with first mention later")
    _value_grid(ax, "y")
    return finish_figure(fig, title, subtitle)


def plot_distinctive_terms(terms: pd.DataFrame, title: str, subtitle: str) -> Figure:
    """`terms` has columns label, term, log_ratio (top terms per label)."""
    n_terms = terms.groupby("label").size().max()
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(11, 1.0 + 0.27 * n_terms))
    for ax, label in zip(axes, LABEL_ORDER):
        rows = terms[terms["label"] == label].sort_values("log_ratio")
        ax.barh(rows["term"], rows["log_ratio"], height=BAR_HEIGHT, color=SENTIMENT_COLORS[label])
        ax.set_title(label)
        ax.set_xlabel("Log ratio vs other classes")
        ax.spines["left"].set_visible(False)
        _value_grid(ax)
    return finish_figure(fig, title, subtitle)


def plot_vader_by_class(full_scores: dict[str, pd.Series], window_scores: dict[str, pd.Series],
                        panel_titles: tuple[str, str], title: str, subtitle: str) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, groups, panel_title in zip(axes, (full_scores, window_scores), panel_titles):
        _class_boxes(ax, groups)
        ax.axvline(0, color=BASELINE, linewidth=1, zorder=0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("VADER compound score")
        ax.set_title(panel_title)
        ax.spines["left"].set_visible(False)
        _value_grid(ax)
    return finish_figure(fig, title, subtitle)
