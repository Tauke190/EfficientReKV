"""Spider (radar) plots of per-task accuracy for streaming video methods, one figure per benchmark.

Numbers are transcribed from the paper tables, not read from results/. All axes share one
radial scale in steps of 10, from the lowest score rounded down to the highest rounded up,
so rings are equally spaced and radius is comparable across axes. The average is shown in
the legend rather than as a spoke, since it is derived from the spokes already drawn.

Usage:
  python plot_radar.py --bench streamingbench
  python plot_radar.py --bench streambench --out figures/radar_streambench.pdf
  python plot_radar.py --bench all          # every benchmark, default output paths
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OURS = "StreamPrune (Ours)"

# Line styles. "ref" is an uncompressed reference: neutral, dashed, no tint.
REF = dict(color="#8a8a86", ls=(0, (4, 2)), lw=1.6, alpha=0.0)
OURS_STYLE = dict(color="#2a78d6", ls="-", lw=2.6, alpha=0.14)
# baseline colours, assigned in table order (categorical slots 2-6)
BASELINE_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Each method row: per-task scores..., avg. Rows are drawn and listed in table order.
BENCHMARKS = {
    "streamingbench": {
        "title": "StreamingBench",
        "tasks": ["CS", "OP", "ATP", "PR", "ACP", "SU", "EU", "CT", "TR", "CR"],
        "ref": "None",
        "rows": {
            "None":               [79.2, 77.5, 75.6, 66.0, 62.2, 60.3, 72.3, 43.6, 69.7, 79.5, 69.1],
            "ToMe":               [67.5, 64.6, 66.3, 63.0, 58.4, 53.3, 65.2, 19.7, 57.3, 76.6, 59.4],
            "VisionZip":          [69.5, 66.4, 69.3, 50.7, 52.9, 57.2, 64.1, 33.3, 54.8, 73.4, 60.4],
            "VidCom²":            [76.0, 68.1, 71.6, 62.0, 58.6, 52.0, 64.0, 42.5, 60.4, 76.6, 63.6],
            "STC-Pruner":         [75.4, 66.8, 71.2, 63.9, 57.8, 51.2, 64.0, 45.1, 63.2, 76.6, 63.7],
            "STC-Cacher & Prune": [74.4, 71.3, 72.9, 69.4, 58.9, 52.4, 60.9, 44.0, 66.9, 77.3, 65.2],
            OURS:                 [74.8, 77.92, 79.32, 68.52, 64.2, 59.76, 70.7, 50.0, 70.72, 82.03, 70.82],
        },
    },
    "streambench": {
        "title": "StreamBench",
        "tasks": ["OS", "LM", "SM", "CI", "KG", "SF"],
        "ref": None,
        "rows": {
            "Video-online":    [41.4, 48.8, 52.9, 62.7, 69.2, 64.1, 56.4],
            "FlashVStream":    [37.1, 44.5, 48.6, 58.1, 66.4, 59.2, 52.1],
            "StreamChat-Slow": [51.7, 53.9, 57.8, 68.5, 88.1, 69.3, 64.7],
            "StreamChat-Base": [50.5, 52.9, 56.1, 67.6, 87.9, 68.3, 63.8],
            "StreamChat-Fast": [48.1, 49.5, 53.5, 65.2, 86.7, 67.6, 61.7],
            OURS:              [50.84, 61.39, 59.0, 73.06, 76.17, 71.19, 65.1],
        },
    },
}

STEP = 10
FONT = 2.0  # scale on every font size


def styles(bench):
    out, i = {}, 0
    for m in bench["rows"]:
        if m == OURS:
            out[m] = OURS_STYLE
        elif m == bench.get("ref"):
            out[m] = REF
        else:
            out[m] = dict(color=BASELINE_COLORS[i], ls="-", lw=1.5, alpha=0.045)
            i += 1
    return out


def draw(ax, bench):
    tasks = bench["tasks"]
    n = len(tasks)
    vals = {m: np.array(v[:n], dtype=float) for m, v in bench["rows"].items()}
    avg = {m: v[n] for m, v in bench["rows"].items()}
    style = styles(bench)

    allv = np.stack(list(vals.values()))
    lo = STEP * np.floor(allv.min() / STEP)
    hi = STEP * np.ceil(allv.max() / STEP)
    rings = np.arange(lo, hi + 1e-9, STEP)

    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(ang, ang[0])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(lo, hi)
    ax.set_yticks(rings)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#d9d8d2", lw=0.7)
    ax.xaxis.grid(True, color="#d9d8d2", lw=0.7)
    ax.spines["polar"].set_color("#c3c2b7")
    ax.spines["polar"].set_linewidth(0.8)

    # ring values, along the gap between the first two axes
    for v in rings[1:]:
        ax.text(np.pi / n, v, f"{v:.0f}", ha="center", va="center", fontsize=8 * FONT,
                color="#6b6a64", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    ax.set_xticks(ang)
    ax.set_xticklabels(tasks, fontsize=12 * FONT, fontweight="bold", color="#1f1f1d")
    ax.tick_params(axis="x", pad=10 * FONT)

    # table order; ours is drawn on top via zorder
    for m, v in vals.items():
        s = style[m]
        r = np.append(v, v[0])
        ax.plot(closed, r, color=s["color"], ls=s["ls"], lw=s["lw"],
                zorder=5 if m == OURS else 3, label=f"{m} ({avg[m]:.1f})")
        if s["alpha"]:
            ax.fill(closed, r, color=s["color"], alpha=s["alpha"], zorder=2)

    ax.set_title(bench["title"], fontsize=15 * FONT, fontweight="bold", pad=30 * FONT)


def plot(bench, out):
    # a square for the plot (with room around it for task labels and title),
    # plus a strip below for the legend
    W, H_LEG = 9.0, 1.7
    size, below, above = 5.6, 0.8, 1.75  # circle diameter; room for labels below / labels+title above
    H_PLOT = below + size + above
    H = H_PLOT + H_LEG
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([(W - size) / 2 / W, (H_LEG + below) / H, size / W, size / H], polar=True)
    draw(ax, bench)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=9 * FONT, handlelength=2.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, (H_LEG + 0.05) / H))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200)  # no tight bbox: it ignores the figure legend and clips it
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=list(BENCHMARKS) + ["all"], default="all")
    p.add_argument("--out", default=None,
                   help="output path (default figures/radar_<bench>.png); not allowed with --bench all")
    args = p.parse_args()

    if args.bench == "all":
        if args.out:
            p.error("--out needs a single --bench")
        for key, bench in BENCHMARKS.items():
            plot(bench, f"figures/radar_{key}.png")
    else:
        plot(BENCHMARKS[args.bench], args.out or f"figures/radar_{args.bench}.png")


if __name__ == "__main__":
    main()
