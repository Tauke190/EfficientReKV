"""Pareto figure for stage-2 token pruning: accuracy against how much was pruned.

Why prune rate on x and not --prune_threshold. The threshold is an arbitrary dial
position on a cosine distance; equal steps in it are not equal steps in anything
physical. Measured on llava_ov_0.5b @ 1 fps, the four thresholds 0.6/0.7/0.8/0.9 --
evenly spaced -- drop token budgets of 8.2/2.8/0.92/0.47%, i.e. most of the range
is spent in the first step. Plotting against the threshold would draw that collapse
as a straight line. Prune rate is the resource actually being spent (it is what sets
KV-Cache size, prefill FLOPs and throughput), so it is the cost axis a Pareto plot
needs. Each point is annotated with the threshold that produced it, so the knob is
still readable off the figure.

Scale. The points span 0 to 99.53% pruned and four of the seven sit above 91%, so a
linear prune-rate axis packs everything interesting into its last centimetre. The
axis is therefore linear in log10(tokens kept) -- which spreads the high-pruning end
evenly -- but ticked and labelled in prune rate, the quantity being reported.

The blind floor. Dashed horizontal lines mark the language-prior score, measured with
--blind: no video is ingested at all, so it is what the answer distribution plus the
language prior alone are worth. It is the y-value that "the video stopped
contributing" converges to, and the reason 36.82% at a 99.53% prune rate reads as
collapse rather than as a respectable number.

Numbers are transcribed from the sweep's results, not read from results/ -- the runs
live on the cluster. Update ARMS when a sweep finishes; keep_pct is the aggregate
token_keep_rate the run reported (see BaseVQA.reduction_stats).

Usage:
  python plot_pareto.py
  python plot_pareto.py --out figures/pareto.png --linear
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL = "LLaVA-OV-0.5B"
SAMPLE_FPS = 1

# threshold -> (tokens kept %, accuracy %). threshold None is the unpruned baseline.
# A baseline keeps 100% by definition; it is the left anchor of every curve.
BENCHMARKS = {
    "OVO-Bench (Real-Time)": {
        "blind": 27.87,
        "arms": [
            (None,  100.00, 47.43),
            (0.25,   56.20, 50.18),
            (0.5,    18.50, 50.48),
            (0.6,     8.20, 48.70),
            (0.7,     2.80, 46.31),
            (0.8,     0.92, 41.40),
            (0.9,     0.47, 36.82),
        ],
    },
    "OVO-Bench (Backward)": {
        "blind": 27.00,
        "arms": [
            (None,  100.00, 33.88),
            (0.25,   56.20, 33.04),
            (0.5,    18.50, 32.27),
            (0.6,     8.20, 31.71),
            (0.7,     2.80, 28.90),
            (0.8,     0.92, 29.83),
            (0.9,     0.47, 28.22),
        ],
    },
}

COLORS = ["#2b6cb0", "#c05621"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/pareto_0.5b.png")
    ap.add_argument("--linear", action="store_true",
                    help="Linear prune-rate axis instead of the default log-in-tokens-kept "
                         "one. Faithful to the raw quantity, but packs the four points "
                         "above 91%% pruned into the right edge.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for (name, data), color in zip(BENCHMARKS.items(), COLORS):
        arms = data["arms"]
        keep = np.array([k for _, k, _ in arms], dtype=float)
        acc = np.array([a for _, _, a in arms], dtype=float)
        # The reported axis. Prune rate and tokens kept are the same number mirrored;
        # x below is still derived from keep so the log option has something positive
        # to work on (a 100% prune rate would be log(0)).
        x = keep if not args.linear else 100.0 - keep
        if not args.linear:
            x = np.log10(keep)

        ax.plot(x, acc, "-o", color=color, label=name, markersize=6, linewidth=1.8, zorder=3)
        ax.axhline(data["blind"], color=color, linestyle="--", linewidth=1.1, alpha=0.55,
                   zorder=1)
        # Placed over the empty mid-right span rather than at either end: the left end
        # carries the legend and the far right carries the last arm's marker, which the
        # backward floor label would otherwise sit on top of.
        ax.annotate(f"blind floor {data['blind']:.1f}%", xy=(x[-3], data["blind"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=color, alpha=0.85)

        # Label each point with the threshold that produced it: the axis reports the
        # cost, but the threshold is the knob someone reproducing this has to set.
        for (thr, k, a), xi in zip(arms, x):
            label = "baseline" if thr is None else f"{thr:g}"
            ax.annotate(label, xy=(xi, a), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=color)

    if args.linear:
        ax.set_xlabel("Prune rate (% of visual tokens dropped)")
    else:
        # Tick in prune rate while the underlying axis is log10(tokens kept), so the
        # spacing spreads the high-pruning end without relabelling the quantity.
        ticks_keep = [100, 56.2, 18.5, 8.2, 2.8, 0.92, 0.47]
        ax.set_xticks([np.log10(t) for t in ticks_keep])
        ax.set_xticklabels([f"{100 - t:.4g}" for t in ticks_keep])
        ax.set_xlabel("Prune rate (% of visual tokens dropped)   [log-spaced]")
        ax.invert_xaxis()  # log10(keep) decreases as pruning rises; put 0% pruned left

    # Headroom below the lowest floor. Without it matplotlib fits the y-axis to the
    # data, the dashed floors land on the bottom spine, and the gap between the last
    # arm and the language prior -- the thing the floors exist to show -- is unreadable.
    floors = [d["blind"] for d in BENCHMARKS.values()]
    accs = [a for d in BENCHMARKS.values() for _, _, a in d["arms"]]
    lo, hi = min(floors), max(accs)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - 2.2 * pad, hi + 2.0 * pad)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{MODEL} @ {SAMPLE_FPS} fps — accuracy vs. stage-2 token pruning\n"
                 "point labels are --prune_threshold", fontsize=11)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")

    # The knee, stated rather than left to the eye: the last arm whose accuracy is
    # still within 1 point of the baseline.
    for name, data in BENCHMARKS.items():
        base = data["arms"][0][2]
        knee = None
        for thr, k, a in data["arms"][1:]:
            if a >= base - 1.0:
                knee = (thr, k, a)
        if knee:
            print(f"{name}: within 1 pt of baseline ({base:.2f}%) down to "
                  f"threshold {knee[0]:g} — {100 - knee[1]:.2f}% pruned, {knee[2]:.2f}%")


if __name__ == "__main__":
    main()
