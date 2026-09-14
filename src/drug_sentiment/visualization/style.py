"""Shared look for every static figure: palette, typography, recessive chrome, titles and saving.

Colour rules:
- Sentiment is an ordered scale, so the classes use a diverging scheme: red (negative), a gray
  midpoint (neutral) and blue (positive). Checked with the data-viz palette validator on the light
  surface, all pairs: the poles are colour-blind separated (deltaE 21.6), every pair clears the
  normal-vision floor (deltaE >= 17.8) and all three clear 3:1 contrast. The midpoint is gray on
  purpose so it reads as "no polarity".
- Charts that are not about sentiment use one violet that no class uses, so blue always means positive.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from drug_sentiment.config import get_config

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

LABEL_ORDER = ("negative", "neutral", "positive")
SENTIMENT_COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
SERIES_COLOR = "#4a3aa7"


def apply_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # silence per-glyph fallback lookups
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 11,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 1,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ink_on(fill_hex: str) -> str:
    """Label colour for text placed inside a filled mark: white or primary ink, whichever contrasts more."""
    fill = _relative_luminance(fill_hex) + 0.05
    white_contrast = 1.05 / fill
    ink_contrast = fill / (_relative_luminance(INK_PRIMARY) + 0.05)
    return "#ffffff" if white_contrast > ink_contrast else INK_PRIMARY


def finish_figure(fig: Figure, title: str, subtitle: str | None = None) -> Figure:
    """Add a header band with a left-aligned title and optional takeaway; the figure grows to fit it,
    so the figsize a plot function asks for is the height of its plot area."""
    width, plot_height = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, width=int(width * 13)) if subtitle else []
    header_inches = 0.5 + 0.19 * len(lines)
    height = plot_height + header_inches
    fig.set_size_inches(width, height)
    fig.tight_layout(rect=(0, 0, 1, 1 - header_inches / height))
    fig.text(0.01, 1 - 0.1 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK_PRIMARY)
    if lines:
        fig.text(
            0.01, 1 - 0.38 / height, "\n".join(lines),
            ha="left", va="top", fontsize=10, color=INK_SECONDARY, linespacing=1.35,
        )
    return fig


def save_figure(fig: Figure, name: str) -> Path:
    path = get_config().artifacts.figures_dir / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    return path
