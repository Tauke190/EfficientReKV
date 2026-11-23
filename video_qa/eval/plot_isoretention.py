"""Iso-retention figure: accuracy vs token keep rate, token-level vs frame-level pruning.

Single CVPR column (3.3 in), 12 pt text. Keep rate is the pooled tokens_kept/tokens_seen
of each run; accuracy is the StreamBench judge micro-average from streambench_scores.json.

    python video_qa/eval/plot_isoretention.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, BASE_C = '#2a78d6', '#eb6834', '#9a9893'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

METHODS = [  # (label, colour, run suffixes from low to high keep rate)
    ('Token (ours)', BLUE, ['rlt_ref0.7cosine', 'rlt_ref0.6cosine', 'rlt_ref0.5cosine']),
    ('Frame', ORANGE, ['rlt_frame0.52cosine', 'rlt_frame0.49cosine', 'rlt_frame0.44cosine']),
]


def point(root, suffix):
    d = os.path.join(root, f'64-1.0-{suffix}' if suffix else '64-1.0')
    s = json.load(open(os.path.join(d, 'streambench_scores.json')))
    acc = s['micro_average']
    acc = acc['accuracy'] if isinstance(acc, dict) else acc
    df = pd.read_csv(os.path.join(d, 'results.csv'))
    keep = df.tokens_kept.sum() / df.tokens_seen.sum() if 'tokens_kept' in df else 1.0
    return 100 * keep, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default='results/llava_ov_7b/streambench')
    ap.add_argument('--out_dir', default='results/analysis/isoretention')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _, base = point(args.root, '')
    pts = {lab: [point(args.root, s) for s in sfx] for lab, _, sfx in METHODS}

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.axhline(base, color=BASE_C, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(31.5, base + 0.35, 'Unpruned', ha='right', va='bottom', fontsize=11, color=INK2)

    for lab, col, _ in METHODS:
        x, y = zip(*pts[lab])
        ax.plot(x, y, '-o', color=col, lw=2, ms=7, mec='white', mew=1.3, zorder=3)
    # direct labels at the low-budget end, where the lines are furthest apart
    (xt, yt), (xf, yf) = pts['Token (ours)'][0], pts['Frame'][0]
    ax.text(6.3, 63.6, 'Token (ours)', color=BLUE, fontsize=11, va='bottom')
    ax.text(xf * 1.08, yf - 0.3, 'Frame', color=ORANGE, fontsize=11, va='center')

    # gap brackets at the two budgets where the methods split
    for i, (shift, side) in enumerate([(0.86, 'right'), (1.12, 'left')]):
        (x1, y1), (x2, y2) = pts['Token (ours)'][i], pts['Frame'][i]
        xm = (x1 + x2) / 2 * shift
        ax.annotate('', (xm, y1), (xm, y2),
                    arrowprops=dict(arrowstyle='<->', color=INK, lw=1, shrinkA=0, shrinkB=0))
        ax.text(xm * (0.97 if side == 'right' else 1.03), (y1 + y2) / 2, f'+{y1 - y2:.1f}',
                ha=side, va='center', fontsize=11, color=INK)

    ax.set_xscale('log')
    ax.set_xlim(6, 33)
    ax.set_xticks([8, 16, 28])
    ax.set_xticklabels(['8', '16', '28'])
    ax.minorticks_off()
    ax.set_ylim(53, 68)
    ax.set_yticks([55, 60, 65])
    ax.set_xlabel('Tokens kept (%)')
    ax.set_ylabel('Accuracy (%)')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'fig_isoretention.{ext}'), dpi=200,
                    bbox_inches='tight')
    for lab in pts:
        print(lab, [(round(k, 1), a) for k, a in pts[lab]])
    print('unpruned', base, '->', args.out_dir)


if __name__ == '__main__':
    main()
