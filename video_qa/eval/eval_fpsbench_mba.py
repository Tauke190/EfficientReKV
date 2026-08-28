"""Score an FPSBench --mba run: Binary Accuracy, Multiple Binary Accuracy, and the
diagnostics that say whether a number is real.

Why this metric exists here. FPSBench is five-way multiple choice, so its chance floor is
0.200, and a blind ReKV control (no video at all) measures 0.230 because "None of the
above" is never the key and is easy to rule out from the question alone. Measured video
arms land at 0.278-0.295 across a 1-30 fps sweep. With N=990 the standard error is about
0.014, so the entire frame-rate sweep spans less than one standard error of its own
endpoints: the metric cannot tell those arms apart, and it cannot tell any of them from a
model that has learned the answer distribution.

MBA (TemporalBench, arXiv 2410.10818) fixes the floor rather than the model. Each item is
re-asked as K independent binaries -- one per candidate, exactly one of which is the key --
and counts as correct only when every binary is right. Three properties follow:

* the uniform-guessing floor drops from 0.200 to 1/2**K, i.e. 0.0625 at K=4;
* every constant strategy scores exactly 0 -- always-Yes fails the K-1 negatives,
  always-No fails the positive;
* a model can no longer be right by elimination, because it never sees the other options.

So a model that was riding the floor lands near zero and one that understands the video
does not. That separation is the measurement.

What is reported, and why each line is needed to read the others:

  Binary Accuracy (BA)     mean over rows. Floors at 0.500, and is *not* the headline:
                           a model answering No to everything scores (K-1)/K = 0.750 here.
                           It is reported because MBA alone cannot distinguish "wrong about
                           the video" from "answered No to everything", and BA plus the
                           yes-rate can.
  Multiple Binary Acc      the metric. Fraction of items with every binary correct.
  yes-rate                 fraction of binaries answered Yes. A calibrated model sits near
                           1/K (0.250 at K=4). Far above means Yes-bias, far below No-bias,
                           and either sends MBA toward 0 for a reason that is about
                           response bias rather than about video understanding.
  positive / negative BA   BA split by whether the candidate was the key. The gap between
                           them is the yes-rate expressed as an error asymmetry, and it is
                           what a reviewer will ask for first.
  chance / always-No       the two floors, computed from this run's own K rather than
                           quoted, since --mba_include_none changes K.
  parse failures           --mba_scoring generate only: responses that committed to
                           neither verdict. Scored wrong, counted separately, because a
                           hedging model and a wrong model are different findings.

Usage:
    python video_qa/eval/eval_fpsbench_mba.py --save_dir results/llava_ov_7b/fpsbench_stream_small/64-8.0-mba
"""

import os
import sys
import argparse

import pandas as pd

REQUIRED = ['video_id', 'question', 'is_positive', 'binary_correct', 'binary_pred']


def load(save_dir):
    path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(path):
        sys.exit(f'no results.csv in {save_dir}')
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f'{path} is not an --mba run: missing {missing}. Re-run the solver with '
                 f'--mba, or score this one with video_qa/eval/export_fpsbench.py.')
    return df


def group_key(df):
    """One group per item. video_id is 1:1 with the question on FPSBench, but grouping on
    the pair keeps this correct if a future annotation file puts several questions on a
    clip -- which the long-stream arm already does."""
    return [df.video_id, df.question]


def summarize(df, label=''):
    groups = df.groupby(group_key(df), sort=False)
    k = groups.size()
    if k.nunique() != 1:
        print(f'  WARNING: groups have {sorted(k.unique())} binaries; a partially written '
              f'run scores as if the missing binaries were correct')
    K = int(k.mode().iloc[0])

    n_items = len(k)
    n_rows = len(df)
    ba = df.binary_correct.mean()
    mba = groups.binary_correct.min().mean()   # all-correct == min over the group
    yes_rate = df.binary_pred.mean()
    pos = df[df.is_positive == 1].binary_correct.mean()
    neg = df[df.is_positive == 0].binary_correct.mean()

    head = f'{label}  ' if label else ''
    print(f'{head}items {n_items}   binaries {n_rows}   K={K}')
    print(f'  Binary Accuracy (BA)        {100*ba:6.2f}%     [floor 50.00%, '
          f'always-No {100*(K-1)/K:.2f}%]')
    print(f'  Multiple Binary Acc (MBA)   {100*mba:6.2f}%     [chance {100/2**K:.2f}%, '
          f'any constant answer 0.00%]')
    print(f'  yes-rate                    {100*yes_rate:6.2f}%     [calibrated {100/K:.2f}%]')
    print(f'    BA on the positive        {100*pos:6.2f}%   (n={int((df.is_positive==1).sum())})')
    print(f'    BA on the negatives       {100*neg:6.2f}%   (n={int((df.is_positive==0).sum())})')
    if 'parse_failed' in df.columns and df.parse_failed.fillna(0).sum():
        n = int(df.parse_failed.fillna(0).sum())
        print(f'  parse failures              {n} binaries ({100*n/n_rows:.2f}%) committed to '
              f'neither verdict; scored wrong')
    return {'items': n_items, 'K': K, 'BA': ba, 'MBA': mba, 'yes_rate': yes_rate}


def by_column(df, col, label):
    if col not in df.columns or df[col].isna().all():
        return
    print(f'\n  MBA by {label}:')
    print(f'    {label:<28} {"items":>6} {"BA%":>7} {"MBA%":>7} {"yes%":>7}')
    for value, sub in df.groupby(col, sort=True):
        g = sub.groupby(group_key(sub), sort=False)
        print(f'    {str(value):<28} {len(g):>6} {100*sub.binary_correct.mean():>7.2f} '
              f'{100*g.binary_correct.min().mean():>7.2f} {100*sub.binary_pred.mean():>7.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--no_breakdown', action='store_true',
                        help='Skip the per-task and per-min_fps tables.')
    args = parser.parse_args()

    df = load(args.save_dir)
    print(f'=== MBA | {args.save_dir} ===')
    summarize(df)

    if not args.no_breakdown:
        # task is the question type; min_fps is FPSBench's own claim about the frame rate a
        # question needs, so the split against it is the sampling-rate question asked at
        # the level MBA can actually resolve.
        by_column(df, 'task', 'task')
        if 'min_fps' in df.columns and 'sample_fps' in df.columns:
            df = df.copy()
            df['resolvable'] = df.sample_fps >= df.min_fps
            by_column(df, 'resolvable', 'sample_fps >= min_fps')


if __name__ == '__main__':
    main()
