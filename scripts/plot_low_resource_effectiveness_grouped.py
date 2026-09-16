"""Plot low-resource effectiveness as a grouped bar chart.

Each metric forms one group with five training settings:

* Base model;
* 250 synthetic sessions;
* 50 real sessions;
* 50 real + 250 synthetic sessions (SFT);
* 50 real + 250 synthetic sessions (RL).

The RL results are reproducible random placeholders until measured values are
available.

Usage:
    python scripts/plot_low_resource_effectiveness_grouped.py
    python scripts/plot_low_resource_effectiveness_grouped.py --rl-seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Values are percentages and follow this order:
# exact match, action type accuracy, action type F1.
BASE_MODEL = (12.16, 60.77, 9.67)
SYNTHETIC_250 = (24.10, 81.40, 26.46)
REAL_50 = (28.98, 85.46, 27.13)
REAL_50_SYNTHETIC_250_SFT = (30.51, 87.33, 35.61)
REAL_50_SYNTHETIC_250_RL = (32.25, 89.10, 40.63)

METRICS = ("Exact Match", "Action Type Accuracy", "Action Type F1")
SERIES_LABELS = (
    "Base",
    "Synth",
    "Real",
    "Real + Synth (SFT)",
    "Real + Synth (SFT + RL)",
)

# Color identifies the training method, while hatching identifies the data:
# left diagonal = synthetic, right diagonal = real, crosshatch = both.
# The three SFT-trained variants therefore share the same muted blue.
COLORS = ("#A8B5A2", "#D1C2A6", "#D1C2A6", "#D1C2A6", "#6F8195")
HATCHES = ("", "////", r"\\\\", "xxxx", "xxxx")


def placeholder_rl_results(seed: int) -> tuple[float, ...]:
    """Return deterministic placeholder RL scores near the SFT results."""
    rng = random.Random(seed)
    return tuple(
        round(max(0.0, min(100.0, value + rng.uniform(-1.5, 2.5))), 2)
        for value in REAL_50_SYNTHETIC_250_SFT
    )


def validate_data(results: list[tuple[float, ...]]) -> None:
    """Validate that each training setting has one percentage per metric."""
    if any(len(row) != len(METRICS) for row in results):
        raise ValueError("Every result row must contain exactly three metrics")
    if any(not 0 <= value <= 100 for row in results for value in row):
        raise ValueError("Metric values must be percentages in [0, 100]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="scripts/low_resource_effectiveness_grouped",
        help="Output basename; writes <basename>.png and <basename>.pdf",
    )
    parser.add_argument(
        "--rl-seed",
        type=int,
        default=42,
        help="Random seed used for the temporary RL placeholder scores",
    )
    args = parser.parse_args()

    results = [
        BASE_MODEL,
        SYNTHETIC_250,
        REAL_50,
        REAL_50_SYNTHETIC_250_SFT,
        REAL_50_SYNTHETIC_250_RL,
    ]
    validate_data(results)

    x = np.arange(len(METRICS))
    bar_width = 0.15
    offsets = (np.arange(len(results)) - (len(results) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    for offset, label, color, hatch, values in zip(
        offsets, SERIES_LABELS, COLORS, HATCHES, results
    ):
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="#555555",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=14)

    # ax.set_title("Low-Resource Effectiveness", fontsize=14, fontweight="semibold")
    ax.set_ylabel("Performance (%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#DDD8CF", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.90)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
