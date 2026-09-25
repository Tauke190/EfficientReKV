"""Pareto figure for LLaVA-OV-7B @ 1 fps: accuracy against RLT prune rate, all benchmarks.

The 7B companion to plot_pareto.py (0.5B, OVO-Bench only). Prune rate on x rather than
--prune_threshold, because equal steps in the threshold are not equal steps in tokens
dropped. The axis runs from 0 to 1 (prune rate as a fraction), linear above 0.4 and
squeezed 4x below it, where only the baselines sit (marked with an axis break). `--threshold` plots
against the threshold instead, which needs no measured prune rates; each arm gets an
equal-width slot there (baseline, 0.25, 0.5, ..., 0.9, blind), since the prune-rate axis
crowds the high thresholds into its right edge.

The blind run (no video at all) is off by default (`--blind` adds it). It is drawn at x = 1 -- every visual token
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
from matplotlib import transforms
from matplotlib.legend_handler import HandlerTuple


MODEL = "LLaVA-OV-7B"
SAMPLE_FPS = 1

# arms: (threshold, prune rate %, accuracy %). Threshold None is the unpruned baseline,
# which prunes nothing by definition. Pruned arms are --prune_method rlt_ref
# (results/llava_ov_7b/<dataset>/64-1.0-rlt_ref<thr>cosine). Accuracy is strict-letter
# for OVO-Bench / StreamingBench, micro average for ODV-Bench / OVBench / StreamBench, and
# gpt-3.5-turbo-0125 judge "yes" rate for RVS.
BENCHMARKS = {
    "OVO-Bench (Real-Time)": {
        "blind": 32.19,
        "arms": [
            (None,   0.0, 63.10),
            (0.25,  40.3, 63.61),
            (0.5,   72.9, 61.60),
            (0.6,   84.5, 60.42),
            (0.7,   92.3, 58.75),
            (0.8,   96.5, 56.02),
            (0.9,   98.4, 53.54),
        ],
    },
    "OVO-Bench (Backward)": {
        "blind": 38.53,
        "arms": [
            (None,   0.0, 45.73),
            (0.25,  38.7, 45.82),
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
            (0.6,   86.7, 69.38),
            (0.7,   93.4, 68.02),
            (0.8,   97.0, 65.29),
            (0.9,   98.7, 62.32),
        ],
    },
    "ODV-Bench": {
        "blind": 46.19,
        "arms": [
            (None,   0.0, 51.53),
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
        "blind": 32.75,   # 64-1.0-blind has no streambench_scores.json on disk to check against
        "arms": [
            (None,   0.0, 65.83),
            (0.25,  42.1, 65.61),
            (0.5,   73.6, 65.13),
            (0.6,   84.8, 63.76),
            (0.7,   92.4, 60.45),
            (0.8,   96.4, 58.22),
            (0.9,   98.2, 52.39),
        ],
    },
    # 0.25-0.7 are partial runs (846 of 1465 questions); baseline, blind, 0.8 and 0.9 are full.
    "RVS-Ego": {
        "blind": 36.31,
        "arms": [
            (None,   0.0, 59.52),
            (0.25,  51.8, 59.57),
            (0.5,   82.4, 61.70),
            (0.6,   90.9, 60.99),
            (0.7,   95.9, 62.06),
            (0.8,   98.1, 61.84),
            (0.9,   99.2, 61.50),
        ],
    },
    "RVS-Movie": {
        "blind": 38.06,
        "arms": [
            (None,   0.0, 47.61),
            (0.25,  10.5, 48.40),
            (0.5,   54.1, 49.03),
            (0.6,   76.1, 51.50),
            (0.7,   89.7, 51.76),
            (0.8,   95.9, 51.71),
            (0.9,   98.4, 48.29),
        ],
    },
}

# Eight benchmarks is the most one axes can carry, so identity is never colour alone:
# each benchmark gets a hue AND its own marker, in the order BENCHMARKS is declared.
# The hues were picked against the light-mode gates (OKLCH L in 0.43-0.77, C >= 0.11,
# >= 3:1 on white) and checked for protan/deutan separation; neighbouring entries in the
# legend are the ones held furthest apart. Red-vs-olive style pairs still collapse under
# simulated colour blindness at this series count -- the markers are what carry those,
# which is also what keeps the figure readable printed in greyscale.
PALETTE = [
    "#a6761d",  # gold
    "#7b2d8e",  # purple
    "#1a9641",  # green
    "#1f5fbf",  # blue
    "#d6301f",  # red
    "#12a3b5",  # teal
    "#7f3b08",  # brown
    "#d84f9c",  # pink
]
# Shapes are paired with the hues so that the pairs colour blindness collapses -- gold /
# green / red, and teal / pink -- are the ones furthest apart in outline.
MARKERS = ["o", "P", "^", "s", "X", "v", "D", "*"]

# Blind marker: a small solid dot inside a larger dotted ring.
# Prune-rate axis: below X_BREAK there is only the baseline and a stray point or two, so
# that stretch is squeezed to X_SQUEEZE of its linear width and marked with an axis break.
X_BREAK = 0.4
X_SQUEEZE = 0.25


def squeeze_x(x):
    x = np.asarray(x, dtype=float)
    return np.where(x < X_BREAK, x * X_SQUEEZE, X_BREAK * X_SQUEEZE + (x - X_BREAK))


def unsqueeze_x(u):
    u = np.asarray(u, dtype=float)
    cut = X_BREAK * X_SQUEEZE
    return np.where(u < cut, u / X_SQUEEZE, X_BREAK + (u - cut))


def axis_break(ax, x):
    """Double-slash break marks on the top and bottom spines at data x."""
    slash = dict(marker=[(-1, -2), (1, 2)], markersize=11, linestyle="none",
                 color="k", mec="k", mew=1.2, clip_on=False, zorder=5)
    base = ax.get_xaxis_transform()  # x in data, y in axes fraction
    for dx in (-2.5, 2.5):
        tr = transforms.offset_copy(base, fig=ax.figure, x=dx, units="points")
        ax.plot([x, x], [0, 1], transform=tr, **slash)


DOT_SIZE = 28
RING_SIZE = 220

# Sized for the ICLR PDF: one text size for axis labels, tick labels and legend.
AXIS_FONT_SIZE = 17
LEGEND_FONT_SIZE = AXIS_FONT_SIZE


def blind_marker(ax, x, y, color, label=None):
    ax.scatter([x], [y], s=DOT_SIZE, color=color, zorder=4, label=label)
    ax.scatter([x], [y], s=RING_SIZE, facecolors="none", edgecolors=color,
               linestyles=":", linewidths=1.4, zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/pareto_7b.png")
    ap.add_argument("--threshold", action="store_true",
                    help="Plot against --prune_threshold (baseline at 0) instead of the "
                         "measured prune rate.")
    ap.add_argument("--blind", action="store_true",
                    help="Also draw the blind (no video) control at x = 1.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    # ~1.4x wider than tall: the curves bunch up near
    # prune rate 1, and the extra width is what separates them there. The legend sits
    # above the axes, so the y-range no longer has to stretch down to make room for it.
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    colors = PALETTE

    # Threshold mode: one evenly spaced slot per arm, baseline first and blind last.
    thresholds = sorted({thr for data in BENCHMARKS.values()
                         for thr, _, _ in data["arms"] if thr is not None})
    slot = {thr: i + 1 for i, thr in enumerate(thresholds)}
    slot[None] = 0
    blind_x = len(thresholds) + 1 if args.threshold else 1.0

    for (name, data), color, marker in zip(BENCHMARKS.items(), colors, MARKERS):
        pts = []
        for thr, prune, acc in data["arms"]:
            if acc is None:
                continue
            if args.threshold:
                x = slot[thr]
            elif prune is None:
                print(f"skipped {name} @ {thr}: no prune rate")
                continue
            else:
                x = prune / 100.0
            pts.append((x, acc))
        blind = data["blind"] if args.blind else None
        if not pts and blind is None:
            print(f"skipped {name}: no numbers yet")
            continue

        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", marker=marker, color=color, label=name,
                    markersize=7 if marker != "*" else 10, markeredgecolor="white",
                    markeredgewidth=0.6, linewidth=2.0, zorder=3)
        if blind is not None:
            # Not joined to the curve: blind is a separate control, not a pruning arm.
            blind_marker(ax, blind_x, blind, color,
                         label=None if pts else f"{name} (blind)")

    if args.threshold:
        ax.set_xlabel("Prune threshold")
        last = blind_x if args.blind else len(thresholds)
        ax.set_xlim(-0.4, last + 0.4)
        ax.set_xticks(range(last + 1))
        # The unpruned baseline is threshold 0 -- it keeps every token by definition.
        ax.set_xticklabels(["0"] + [f"{t:g}" for t in thresholds]
                           + (["blind"] if args.blind else []))
    else:
        ax.set_xlabel("Prune rate (fraction of visual tokens dropped)"
                      + ("; 1 = blind" if args.blind else ""))
        ax.set_xscale("function", functions=(squeeze_x, unsqueeze_x))
        # Without blind the curves end just short of 1; stop the axis right after them.
        ax.set_xlim(-0.03, 1.02 if args.blind else 1.005)
        ax.set_xticks([0.0] + list(np.arange(X_BREAK, 1.01, 0.1)))
        axis_break(ax, X_BREAK / 2)

    handles, labels = ax.get_legend_handles_labels()
    if args.blind:
        # One legend entry for the blind marker, instead of one per benchmark: the dot
        # and its dotted ring drawn on top of each other.
        dot = ax.scatter([], [], s=DOT_SIZE, color="black")
        ring = ax.scatter([], [], s=RING_SIZE, facecolors="none", edgecolors="black",
                          linestyles=":", linewidths=1.4)
        handles.append((dot, ring))
        labels.append("blind (no video)")

    ax.set_ylabel("Accuracy (%)")
    ax.xaxis.label.set_size(AXIS_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
    ax.margins(y=0.04)
    # No title: the caption carries "{MODEL} @ {SAMPLE_FPS} fps" in the paper.
    ax.grid(color="#9e9e9e", alpha=0.55, linewidth=0.7, zorder=0)
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=LEGEND_FONT_SIZE, columnspacing=0.8, handlelength=1.6, handletextpad=0.4,
              borderaxespad=0.2, borderpad=0, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Tight bbox with a hairline pad: the legend lives outside the axes, and any blank
    # margin left here is whitespace in the paper. The PDF is the vector copy for LaTeX.
    for out in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    # The title moved out of the figure, so the run prints what the caption has to say.
    print(f"wrote {args.out} -- caption it {MODEL} @ {SAMPLE_FPS} fps, accuracy vs. token pruning")


if __name__ == "__main__":
    main()
