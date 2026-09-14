"""Pareto figure for LLaVA-OV-7B @ 1 fps: accuracy against RLT prune rate, all benchmarks.

The 7B companion to plot_pareto.py (0.5B, OVO-Bench only). Prune rate on x rather than
--prune_threshold, because equal steps in the threshold are not equal steps in tokens
dropped. The axis is linear from 0 to 1 (prune rate as a fraction). `--threshold` plots
against the threshold instead, which needs no measured prune rates.

The blind run (no video at all) is a special case, drawn at x = 1 -- every visual token
dropped, or equivalently threshold 1 -- as a dot inside a dotted circle in the benchmark's
colour. It is the language-prior floor and deliberately not joined to the curve.

Numbers are transcribed, not read from results/. `prune` is the measured prune rate,
100 * (1 - tokens_kept / tokens_seen) pooled over each run's videos (last row per video
in results.csv, see BaseVQA.reduction_stats). None means "not filled in yet": a point
missing its accuracy is left out, and a point missing its prune rate is left out of the
prune-rate plot only.

Usage:
  python plot_pareto_7b.py
  python plot_pareto_7b.py --threshold --out figures/pareto_7b_threshold.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple


MODEL = "LLaVA-OV-7B"
SAMPLE_FPS = 1

# arms: (threshold, prune rate %, accuracy %). Threshold None is the unpruned baseline,
# which prunes nothing by definition.
BENCHMARKS = {
    "OVO-Bench (Real-Time)": {
        "blind": 32.00,
        "arms": [
            (None,   0.0, 63.00),
            (0.25,  None, 63.61),   # prune rate: 64-1.0-rlt0.25cosine is not on disk
            (0.5,   72.9, 61.00),
            (0.6,   84.5, 60.42),
            (0.7,   92.3, 58.75),
            (0.8,   96.5, 56.02),
            (0.9,   98.4, 53.54),
        ],
    },
    "OVO-Bench (Backward)": {
        "blind": 38.53,
        "arms": [
            (None,   0.0, 45.70),
            (0.25,  None, 45.80),   # prune rate: 64-1.0-rlt0.25cosine is not on disk
            (0.5,   74.5, 45.44),
            (0.6,   86.3, 44.79),
            (0.7,   93.5, 44.64),
            (0.8,   97.0, 41.88),
            (0.9,   98.6, 41.59),
        ],
    },
    "StreamingBench (Real)": {
        "blind": 50.56,
        "arms": [
            (None,   0.0, 70.30),
            (0.25,  46.8, 70.62),
            (0.5,   76.8, 70.82),
            (0.6,   86.7, 68.38),
            (0.7,   93.4, 68.02),
            (0.8,   97.0, 65.29),
            (0.9,   98.7, 62.32),
        ],
    },
    "ODV-Bench": {
        "blind": 46.19,
        "arms": [
            (None,   0.0, 52.00),
            (0.25,  51.8, 51.72),
            (0.5,   79.1, 51.32),
            (0.6,   86.4, 51.20),
            (0.7,   91.1, 51.31),
            (0.8,   93.7, 51.95),
            (0.9,   94.8, 51.58),
        ],
    },
    "OVBench": {
        "blind": 36.73,
        "arms": [
            (None,   0.0, 44.77),
            (0.25,  49.5, 44.36),
            (0.5,   76.7, 44.32),
            (0.6,   86.0, 44.56),
            (0.7,   92.6, 45.08),
            (0.8,   96.4, 44.61),
            (0.9,   98.2, 44.53),
        ],
    },
    "StreamBench": {
        "blind": 32.75,
        "arms": [
            (None,   0.0, 66.05),
            (0.25,  42.1, 65.60),
            (0.5,   73.6, 65.10),
            (0.6,   84.8, 63.80),
            (0.7,   92.4, 60.40),
            (0.8,   96.4, 58.20),
            (0.9,   98.2, 52.40),
        ],
    },
    # Accuracies to fill in once the GPT judge finishes. The 0.25-0.7 prune rates are
    # from partial runs (846 of 1465 questions).
    "RVS-Ego": {
        "blind": None,
        "arms": [
            (None,   0.0, None),
            (0.25,  51.8, None),
            (0.5,   82.4, None),
            (0.6,   90.9, None),
            (0.7,   95.9, None),
            (0.8,   98.1, None),
            (0.9,   99.2, None),
        ],
    },
    "RVS-Movie": {
        "blind": None,
        "arms": [
            (None,   0.0, None),
            (0.25,  10.5, None),
            (0.5,   54.1, None),
            (0.6,   76.1, None),
            (0.7,   89.7, None),
            (0.8,   95.9, None),
            (0.9,   98.4, None),
        ],
    },
}

# Blind marker: a small solid dot inside a larger dotted ring.
DOT_SIZE = 28
RING_SIZE = 220


def blind_marker(ax, y, color, label=None):
    ax.scatter([1.0], [y], s=DOT_SIZE, color=color, zorder=4, label=label)
    ax.scatter([1.0], [y], s=RING_SIZE, facecolors="none", edgecolors=color,
               linestyles=":", linewidths=1.4, zorder=4)


def make_room_for_legend(fig, ax, leg, drawn, margin=0.02):
    """Lower the y-axis floor until the bottom-left legend sits under every curve it spans.

    A curve counts over the legend's x-extent plus its first point past the right edge,
    so the segment leaving the box is cleared too, not just the markers inside it.
    """
    for _ in range(30):
        fig.canvas.draw()
        box = leg.get_window_extent().transformed(ax.transData.inverted())
        ys = []
        for pts in drawn:
            for x, y in pts:
                ys.append(y)
                if x > box.x1:
                    break
        lo, hi = ax.get_ylim()
        if not ys or box.y1 < min(ys) - margin * (hi - lo):
            return
        ax.set_ylim(lo - 0.05 * (hi - lo), hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/pareto_7b.png")
    ap.add_argument("--threshold", action="store_true",
                    help="Plot against --prune_threshold (baseline at 0) instead of the "
                         "measured prune rate.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    colors = plt.get_cmap("tab10").colors
    drawn = []  # every plotted point set, sorted by x, for placing the legend below them

    for (name, data), color in zip(BENCHMARKS.items(), colors):
        pts = []
        for thr, prune, acc in data["arms"]:
            if acc is None:
                continue
            if args.threshold:
                x = 0.0 if thr is None else thr
            elif prune is None:
                print(f"skipped {name} @ {thr}: no prune rate")
                continue
            else:
                x = prune / 100.0
            pts.append((x, acc))
        if not pts and data["blind"] is None:
            print(f"skipped {name}: no numbers yet")
            continue

        if pts:
            xs, ys = zip(*pts)
            drawn.append(pts)
            ax.plot(xs, ys, "-o", color=color, label=name, markersize=5, linewidth=1.8,
                    zorder=3)
        if data["blind"] is not None:
            # Not joined to the curve: blind is a separate control, not a pruning arm.
            blind_marker(ax, data["blind"], color,
                         label=None if pts else f"{name} (blind)")
            drawn.append([(1.0, data["blind"])])

    if args.threshold:
        ax.set_xlabel("RLT prune threshold (0 = unpruned baseline, 1 = blind)")
    else:
        ax.set_xlabel("Prune rate (fraction of visual tokens dropped; 1 = blind)")
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    # One legend entry for the blind marker, instead of one per benchmark: the dot and
    # its dotted ring drawn on top of each other.
    dot = ax.scatter([], [], s=DOT_SIZE, color="grey")
    ring = ax.scatter([], [], s=RING_SIZE, facecolors="none", edgecolors="grey",
                      linestyles=":", linewidths=1.4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append((dot, ring))
    labels.append("blind (no video)")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{MODEL} @ {SAMPLE_FPS} fps: accuracy vs. RLT token pruning")
    ax.grid(alpha=0.25, zorder=0)
    leg = ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
                    loc="lower left", ncol=2, fontsize=9, labelspacing=0.9,
                    framealpha=0.95, edgecolor="lightgrey")
    fig.tight_layout()
    make_room_for_legend(fig, ax, leg, drawn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
