"""The pipeline diagram used in the README and the deck: what happens to a row, and where fitting is allowed."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from drug_sentiment.visualization.style import (
    BASELINE,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_COLOR,
    SURFACE,
    finish_figure,
)

STAGES = [
    ("raw CSVs", ["train 5,279 rows", "test 2,924 rows", "(drug, comment)"]),
    ("preprocess", ["clean text", "find the drug", "keep its sentences", "mask targetdrug /", "otherdrug",
                    "handcrafted + VADER"]),
    ("embed", ["MiniLM, frozen", "384-d vector", "of the window"]),
    ("features", ["TF-IDF word + char", "log-scaled counts", "one-hot drug", "+ MiniLM"]),
    ("LinearSVC", ["class_weight", "balanced", "one weight", "per feature"]),
    ("predictions", ["0 positive", "1 negative", "2 neutral"]),
]


def plot_pipeline() -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(0, len(STAGES) * 2)
    ax.set_ylim(0, 10)
    ax.axis("off")

    for index, (name, lines) in enumerate(STAGES):
        left, width = index * 2 + 0.12, 1.76
        accent = index in (1, 2)  # the drug-conditioning stages this project is built around
        ax.add_patch(FancyBboxPatch((left, 3.1), width, 5.4, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor=SURFACE, edgecolor=SERIES_COLOR if accent else BASELINE,
                                    linewidth=2 if accent else 1.2, zorder=2))
        ax.text(left + width / 2, 7.9, name, ha="center", va="top", fontsize=12, fontweight="semibold",
                color=SERIES_COLOR if accent else INK_PRIMARY, zorder=3)
        ax.text(left + width / 2, 7.0, "\n".join(lines), ha="center", va="top", fontsize=9.5,
                color=INK_SECONDARY, linespacing=1.5, zorder=3)
        if index:
            ax.add_patch(FancyArrowPatch((left - 0.24, 5.8), (left - 0.02, 5.8), arrowstyle="-|>",
                                         mutation_scale=14, color=BASELINE, linewidth=1.4, zorder=1))

    for start, end, label in [
        (0, 3, "row-wise only, cached as pickles\nnothing here is learned from the data"),
        (3, 5, "fitted inside each training fold\nno validation or test row leaks in"),
        (5, 6, "written\nonce"),
    ]:
        x0, x1 = start * 2 + 0.12, end * 2 - 0.12
        ax.plot([x0, x0, x1, x1], [2.5, 2.1, 2.1, 2.5], color=GRIDLINE, linewidth=1.4, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 1.3, label, ha="center", va="center", fontsize=9.5, color=INK_SECONDARY,
                linespacing=1.5)

    return finish_figure(
        fig,
        "One row, end to end",
        "The drug-centred window and the masking (violet) are what make a prediction about one drug rather than "
        "about the comment as a whole.",
    )
