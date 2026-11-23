"""Figures for the CAC freshness audit (reviewer item E1), from the `-fresh` runs.

Reads the per-question fresh_* columns plus the judge's verdicts (via freshness_audit's
loader) and writes, to --out_dir:

  fig_freshness_cdf      CDF of (a) cache age without the query-time flush (newest fully
                         committed frame) and (b) age of the newest frame that kept >=1
                         of its own tokens, per threshold.
  fig_freshness_acc      accuracy by age bin (0-2 s, 2-5 s, >5 s) with denominators,
                         pruned vs the unpruned run on the same questions, binned by
                         (a) block age and (b) kept age.
  fig_pending_vs_discarded
                         tokens from the frames after the last committed block, split into
                         retained-but-pending (made visible by the flush) and discarded
                         by the pruner.
  freshness_summary.csv  every number behind the figures, with n per cell.

Not covered: a per-video time series of kept tokens / filled blocks. The runs store one
snapshot per question, not the per-frame trace, so that figure needs a re-run with
per-frame logging.

Usage:
    python video_qa/eval/plot_freshness.py \
        --run 0.5=results/llava_ov_7b/streambench/64-1.0-rlt_ref0.5cosine-fresh/results.csv \
        --run 0.8=results/llava_ov_7b/streambench/64-1.0-rlt_ref0.8cosine-fresh/results.csv \
        --baseline results/llava_ov_7b/streambench/64-1.0/results.csv \
        --out_dir results/analysis/freshness
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freshness_audit import load_results

# Reference palette (dataviz skill), categorical slots in fixed order; control is neutral.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
BASE_C = '#9a9893'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
P = 196          # tokens per frame (llava_ov, 14x14 pooled)
BINS = [(0, 2, '0–2 s'), (2, 5, '2–5 s'), (5, np.inf, '>5 s')]

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42,
})


def load(runs, baseline):
    base = load_results(baseline)
    base = (base.assign(question_id=base['question_id'].astype(str))
                .groupby('question_id', as_index=False)['qa_acc'].mean()
                .rename(columns={'qa_acc': 'base_acc'}))
    out = {}
    for tau, path in runs:
        df = load_results(path).dropna(subset=['fresh_pending_tokens'])
        df = df.assign(question_id=df['question_id'].astype(str)).merge(
            base, on='question_id', how='left')
        out[tau] = df
    return out


def save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def ecdf(ax, x, color, label):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.r_[0, x], np.r_[0, y], where='post', color=color, lw=2, label=label)


def fig_cdf(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.32))
    for i, (tau, df) in enumerate(data.items()):
        c = SERIES[i]
        b, k = df['fresh_block_age_s'], df['fresh_kept_age_s']
        ecdf(axes[0], b, c, f'τ = {tau}: median {b.median():.0f} s, '
                            f'p90 {b.quantile(.9):.0f} s, max {b.max():.0f} s')
        ecdf(axes[1], k, c, f'τ = {tau}: {(k == 0).mean() * 100:.1f}% at 0 s')
    axes[0].set_xscale('symlog', linthresh=1)
    axes[0].set_xlim(0, 100)
    axes[0].set_xticks([0, 1, 2, 5, 10, 20, 50, 100])
    axes[0].set_xticklabels(['0', '1', '2', '5', '10', '20', '50', '100'])
    axes[0].set_xlabel('age of newest committed frame at query (s)')
    axes[0].set_title('(a) Cache age if the query did not flush', loc='left')
    axes[1].set_xlim(-0.5, 15)
    axes[1].set_ylim(0.9, 1.003)
    axes[1].set_xlabel('age of newest frame with ≥1 retained token (s)')
    axes[1].set_title('(b) Retained-evidence age (note y from 0.9)', loc='left')
    axes[0].set_ylabel('fraction of questions')
    axes[0].legend(loc='lower right', fontsize=7)
    axes[1].legend(loc='lower right', fontsize=7)
    n = len(next(iter(data.values())))
    fig.suptitle(f'StreamBench, ReKV-7B + TTS+CAC, 1 fps, n = {n} questions per τ',
                 x=0.01, ha='left', fontsize=8, color=INK2, y=1.02)
    save(fig, out_dir, 'fig_freshness_cdf')


def binned(df, col):
    rows = []
    for lo, hi, lab in BINS:
        sub = df[(df[col] >= lo) & (df[col] < hi)].dropna(subset=['qa_acc'])
        rows.append(dict(bin=lab, n=len(sub),
                         acc=sub['qa_acc'].mean() if len(sub) else np.nan,
                         base_acc=sub['base_acc'].mean() if len(sub) else np.nan))
    return pd.DataFrame(rows)


def fig_acc(data, out_dir, summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    specs = [('fresh_block_age_s', '(a) by cache age without flush'),
             ('fresh_kept_age_s', '(b) by retained-evidence age')]
    taus = list(data)
    w = 0.8 / (2 * len(taus))
    for ax, (col, title) in zip(axes, specs):
        x = np.arange(len(BINS))
        ns = []
        for i, tau in enumerate(taus):
            t = binned(data[tau], col)
            for _, r in t.iterrows():
                summary.append(dict(tau=tau, binned_by=col, **r.to_dict()))
            off = (2 * i - len(taus) + 0.5) * w
            ax.bar(x + off, t['base_acc'], w * 0.92, color=BASE_C,
                   label='unpruned, same questions' if i == 0 else None)
            ax.bar(x + off + w, t['acc'], w * 0.92, color=SERIES[i], label=f'τ = {tau}')
            ns.append(t['n'].astype(int).tolist())
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b[2]}\n' + '\n'.join(f'n={n[j]}' for n in ns)
                            for j, b in enumerate(BINS)], fontsize=8)
        ax.set_title(title, loc='left')
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('judge accuracy (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), fontsize=7.5,
               bbox_to_anchor=(0.5, 1.06))
    fig.text(0.01, -0.14, 'n = questions per bin, top to bottom ' +
             ' / '.join(f'τ = {t}' for t in taus) +
             '. Bins with n < 30 are too small to compare.', fontsize=7, color=INK2)
    save(fig, out_dir, 'fig_freshness_acc')


def fig_pending(data, out_dir, summary):
    """Tokens from the frames after the last committed block, at query time."""
    taus = list(data)
    pend, disc, seen = [], [], []
    for tau in taus:
        df = data[tau]
        s = df['fresh_block_age_frames'] * P                 # tokens observed in the tail
        p = df['fresh_pending_tokens'].clip(upper=s)
        pend.append(p.sum() / len(df))
        disc.append((s - p).sum() / len(df))
        seen.append(s.sum() / len(df))
        summary.append(dict(tau=tau, binned_by='tail_tokens', bin='per question (mean)',
                            n=len(df), tail_frames=df['fresh_block_age_frames'].mean(),
                            tail_tokens_seen=seen[-1], retained_pending=pend[-1],
                            discarded=disc[-1]))
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    y = np.arange(len(taus))
    for i, tau in enumerate(taus):
        ax.barh(i, pend[i], 0.55, color=SERIES[i])
        ax.barh(i, disc[i], 0.55, left=pend[i] + 2 * 0, color='white',
                edgecolor=SERIES[i], hatch='////', lw=0.8)
        ax.text(pend[i] + disc[i] + 60, i,
                f'{pend[i]:.0f} pending ({pend[i] / seen[i] * 100:.0f}%) + '
                f'{disc[i]:.0f} discarded ({disc[i] / seen[i] * 100:.0f}%)\n'
                f'over {data[tau]["fresh_block_age_frames"].mean():.1f} frames on average',
                va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f'τ = {t}' for t in taus])
    ax.invert_yaxis()
    ax.set_xlim(0, max(p + d for p, d in zip(pend, disc)) * 1.75)
    ax.set_xlabel('tokens per question from frames after the last committed block (mean)')
    ax.grid(axis='y', visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, label='retained, pending (shown by the flush)'),
                       Patch(facecolor='white', edgecolor=INK2, hatch='////',
                             label='discarded by the pruner')],
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, out_dir, 'fig_pending_vs_discarded')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', action='append', required=True,
                    help='tau=path/to/results.csv; repeat per threshold')
    ap.add_argument('--baseline', required=True, help='unpruned results.csv')
    ap.add_argument('--out_dir', default='results/analysis/freshness')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = [tuple(r.split('=', 1)) for r in args.run]
    data = load(runs, args.baseline)
    summary = []
    fig_cdf(data, args.out_dir)
    fig_acc(data, args.out_dir, summary)
    fig_pending(data, args.out_dir, summary)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, 'freshness_summary.csv'),
                                 index=False)
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    print(f'wrote figures to {args.out_dir}')


if __name__ == '__main__':
    main()
