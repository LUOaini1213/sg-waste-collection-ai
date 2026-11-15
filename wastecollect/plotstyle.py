"""Shared chart style for the static PNG figures (light surface).

Categorical hues are assigned in a fixed order and follow the entity (a method or a policy
always has the same colour across figures).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

METHOD_COLOR = {"persistence": SERIES[0], "seasonal_naive": SERIES[1], "mean_rate": SERIES[2], "lightgbm": SERIES[3]}
METHOD_LABEL = {"persistence": "Last value", "seasonal_naive": "Seasonal naive (weekly)",
                "mean_rate": "14-day mean fill rate", "lightgbm": "LightGBM"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 150, "svg.hashsalt": "sg-waste",
    })


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
