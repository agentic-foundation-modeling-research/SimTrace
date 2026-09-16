"""Plot synthetic-gain deltas from the data-mix experiments.

Renders the spreadsheet table of synthetic gain Delta = (Real + Synth) - Real
with 95% bootstrap CIs, one errorbar line per metric, across real-to-synthetic
training-data ratios.

Usage:
    python scripts/plot_data_mix_gains.py [--output scripts/data_mix_gains]
"""

import argparse

import matplotlib.pyplot as plt

RATIOS = ["1:1", "1:2", "1:5", "1:10"]

# Transcribed from the data-mix results spreadsheet (2026-07). Values are
# percentage points; "ci" is the 95% bootstrap interval of the delta.
METRICS = {
    "Exact Match": {
        "real": 28.98,
        "delta": [1.45, 3.13, 2.34, 2.99],
        "ci": [(-0.12, 3.16), (1.19, 5.03), (0.79, 3.64), (0.87, 4.91)],
    },
    "Action Type F1": {
        "real": 27.13,
        "delta": [6.06, 7.85, 9.18, 9.10],
        "ci": [(3.63, 8.36), (5.66, 10.27), (7.05, 11.52), (6.89, 10.93)],
    },
    "Action Type Acc": {
        "real": 85.46,
        "delta": [1.25, 1.37, 2.29, 2.33],
        "ci": [(-1.19, 3.25), (0.00, 2.82), (0.63, 3.94), (0.00, 4.84)],
    },
}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="scripts/data_mix_gains",
        help="Output basename; writes <basename>.png and <basename>.pdf",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    positions = range(len(RATIOS))
    dodge = 0.06

    for i, (name, m) in enumerate(METRICS.items()):
        x = [p + (i - 1) * dodge for p in positions]
        lo = [d - c[0] for d, c in zip(m["delta"], m["ci"])]
        hi = [c[1] - d for d, c in zip(m["delta"], m["ci"])]
        ax.errorbar(
            x,
            m["delta"],
            yerr=[lo, hi],
            label=name,
            color=COLORS[i],
            marker=MARKERS[i],
            markersize=6,
            linewidth=2,
            capsize=3,
            capthick=1.5,
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.5, zorder=0)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(RATIOS)
    ax.set_xlabel("Real : synthetic ratio")
    ax.set_ylabel("Synthetic gain Δ (points)")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.output}.{ext}", dpi=300)
        print(f"wrote {args.output}.{ext}")


if __name__ == "__main__":
    main()
