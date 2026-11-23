"""CAC freshness audit: how fresh is memory when a question arrives?

Reads the `fresh_*` columns that every streaming solver writes per question (see
`Abstract_ReKV.freshness_snapshot` and `BaseVQA.reduction_stats`) and prints, per run:

  1. Pending tail     how many tokens were still waiting for a full block when the
                      question came, and the padding the question-time flush added.
  2. Block age        how stale the cache would have been *without* the flush.
  3. Retained         how old the newest frame that kept its own tokens was, and how much
                      of the last 1/5/10 s the pruner kept.
  4. Accuracy by age  qa_acc binned by kept age (seconds), the number that says whether
                      staleness actually costs answers.

Usage:
    python video_qa/eval/freshness_audit.py \
        results/llava_ov_0.5b/ovobench_realtime/64-1.0/results.csv \
        results/llava_ov_0.5b/ovobench_realtime/64-1.0-rlt_ref0.5cosine/results.csv

    --by task          also break accuracy-by-age down per column value (e.g. task)
    --bins 0,1,3,10    kept-age bin edges in seconds (last bin is open-ended)
    --out audit.csv    also write the accuracy-by-age table, for the figure later
    --baseline CSV     unpruned run on the same questions (matched by question_id): adds
                       its accuracy on each kept-age bin, so a drop in the stale bins can
                       be told apart from those questions simply being harder

Runs from before the audit was added have no `fresh_*` columns; they are reported as such
and skipped, so re-run them to get the numbers.
"""
import argparse
import os

import numpy as np
import pandas as pd


def run_name(path):
    parts = os.path.normpath(path).split(os.sep)
    return '/'.join(parts[-4:-1]) if len(parts) >= 4 else path


def pct(s, q):
    return float(np.percentile(s, q)) if len(s) else float('nan')


def describe(s, unit=''):
    s = s.dropna()
    return (f'mean {s.mean():.2f}{unit}  median {pct(s, 50):.2f}{unit}  '
            f'p90 {pct(s, 90):.2f}{unit}  max {s.max():.2f}{unit}')


def age_bins(df, edges):
    labels = [f'[{a:g},{b:g})s' for a, b in zip(edges[:-1], edges[1:])] + [f'>={edges[-1]:g}s']
    cut = pd.cut(df['fresh_kept_age_s'], bins=list(edges) + [np.inf], right=False,
                 labels=labels)
    return cut


def load_results(path):
    """results.csv, with `qa_acc` filled from the judge cache for open-ended runs.

    Multiple-choice solvers write qa_acc (0/100) themselves. Open-ended ones (StreamBench)
    do not: the LLM judge's verdicts live in tmp_*.jsonl next to results.csv, keyed per
    row. Reuse eval_streambench's own join so the two can never disagree; unscored rows
    get NaN and drop out of the means.
    """
    df = pd.read_csv(path)
    if 'qa_acc' in df.columns:
        return df
    save_dir = os.path.dirname(path) or '.'
    if not any(f.startswith('tmp') and f.endswith('.jsonl') for f in os.listdir(save_dir)):
        return df
    import eval_streambench as esb   # sibling module; this file runs from video_qa/eval/
    keys = esb.judge_keys(df)
    verdicts = esb.load_verdicts(esb.find_cache_dir(save_dir), keys)
    df['qa_acc'] = [(100.0 if verdicts[k]['pred'] == 'yes' else 0.0) if k in verdicts
                    else np.nan for k in keys]
    print(f'  ({os.path.basename(save_dir)}: qa_acc from judge cache, '
          f'{len(verdicts)}/{len(keys)} rows scored)')
    return df


def audit(path, edges, by, base=None):
    df = load_results(path)
    name = run_name(path)
    print(f'\n=== {name}  ({len(df)} questions)')
    if 'fresh_pending_tokens' not in df.columns:
        print('  no fresh_* columns: run predates the audit, re-run it')
        return None

    df = df.dropna(subset=['fresh_pending_tokens'])
    has_pend = df['fresh_pending_tokens'] > 0
    print('  [1] pending tail at query time')
    print(f'      questions with tokens pending: {has_pend.mean() * 100:.1f}%')
    print(f'      pending tokens:   {describe(df["fresh_pending_tokens"])}')
    print(f'      flush padding:    {describe(df["fresh_flush_pad_tokens"])}  '
          f'(tokens added per question by the flush)')
    print('  [2] cache age without the flush (newest fully committed frame)')
    print(f'      block age:        {describe(df["fresh_block_age_s"], "s")}')
    print('  [3] retained evidence (pruner)')
    print(f'      kept age:         {describe(df["fresh_kept_age_s"], "s")}  '
          f'(0 = query frame kept its own tokens)')
    print(f'      query frame kept >=1 token: {(df["fresh_kept_age_frames"] == 0).mean() * 100:.1f}%')
    recent = [c for c in df.columns
              if c.startswith('fresh_recent_') and c.endswith('_keep_rate')]
    for col in sorted(recent, key=lambda c: float(''.join(ch for ch in c[13:-10]
                                                         if ch.isdigit() or ch == '.') or 0)):
        w = col[len('fresh_recent_'):-len('_keep_rate')]
        fw = df[f'fresh_recent_{w}_frames_with_tokens']
        print(f'      last {w:>4}: token keep {df[col].mean() * 100:5.1f}%   '
              f'frames with tokens {fw.mean() * 100:5.1f}%')
    if 'fresh_local_span_frames' in df.columns and 'sample_fps' in df.columns:
        span = df['fresh_local_span_frames'] / df['sample_fps']
        print(f'      local window covers last: {describe(span, "s")}  (attended directly)')

    if 'qa_acc' not in df.columns:
        return None
    df = df.assign(age_bin=age_bins(df, edges))
    groups = ['age_bin'] + ([by] if by and by in df.columns else [])
    paired = False
    if base is not None and 'question_id' in df.columns:
        df = df.assign(question_id=df['question_id'].astype(str)).merge(
            base, on='question_id', how='left')
        paired = df['base_acc'].notna().any()
        n_miss = int(df['base_acc'].isna().sum())
        if n_miss:
            print(f'  (baseline missing for {n_miss} questions; they count only in acc)')
    agg = {'n': ('qa_acc', 'size'), 'acc': ('qa_acc', 'mean')}
    if paired:
        agg['base_acc'] = ('base_acc', 'mean')
    tab = df.groupby(groups, observed=True).agg(**agg).reset_index()
    if paired:
        tab['delta'] = tab['acc'] - tab['base_acc']
    print('  [4] accuracy by kept age' + ('  (base = unpruned, same questions)' if paired else ''))
    line = f'      overall acc {df["qa_acc"].mean():.2f}'
    if paired:
        line += f'   base {df["base_acc"].mean():.2f}'
    print(line)
    for _, r in tab.iterrows():
        key = '  '.join(str(r[g]) for g in groups)
        line = f'      {key:<32} n={int(r["n"]):5d}  acc={r["acc"]:6.2f}'
        if paired:
            line += f'  base={r["base_acc"]:6.2f}  delta={r["delta"]:+6.2f}'
        print(line)
    tab.insert(0, 'run', name)
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csvs', nargs='+', help='results.csv files from streaming runs')
    ap.add_argument('--bins', default='0,1,3,10',
                    help='kept-age bin edges in seconds, ascending; last bin is open')
    ap.add_argument('--by', default=None, help='extra column to split accuracy by, e.g. task')
    ap.add_argument('--out', default=None, help='write the accuracy-by-age table here')
    ap.add_argument('--baseline', default=None,
                    help='unpruned results.csv on the same questions, matched by question_id')
    args = ap.parse_args()

    base = None
    if args.baseline:
        b = load_results(args.baseline)
        if 'qa_acc' not in b.columns:
            print(f'WARNING: {args.baseline} has no qa_acc and no judge cache; '
                  f'auditing without the paired control')
        else:
            base = (b.assign(question_id=b['question_id'].astype(str))
                     .groupby('question_id', as_index=False)['qa_acc'].mean()
                     .rename(columns={'qa_acc': 'base_acc'}))
    edges = [float(x) for x in args.bins.split(',')]
    tabs = [t for t in (audit(p, edges, args.by, base) for p in args.csvs) if t is not None]
    if args.out and tabs:
        pd.concat(tabs, ignore_index=True).to_csv(args.out, index=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
