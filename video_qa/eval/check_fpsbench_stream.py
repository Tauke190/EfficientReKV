"""Audit an FPS-Bench-Stream run before its accuracy is believed.

`eval_fpsbench_stream.py` reports the numbers; this says whether they mean anything. Every
failure it looks for produces a plausible-looking accuracy rather than a crash, which is
why it is a separate step run on every eval:

1. **Lookahead.** A question may only be answered from frames at or before its trigger, so
   every row must satisfy `n_frames_seen <= floor(trigger_time_sec * sample_fps) + 1`. A
   violation does not look like a bug in the CSV -- it looks like a better model.
2. **Rows that never had the evidence.** Under `--trigger end` the needle finishes long
   before the question, so `needle_last_frame < n_frames_seen` must hold; a row where it
   does not was asked before its own evidence had arrived and cannot be answered from the
   video at all. Under `--trigger query` that is the protocol (the needle runs a few
   seconds past the certificate end), so those rows are reported, not failed.
3. **Duplicates and gaps.** A merge across chunks that lost or doubled a chunk leaves a
   file that still scores; the question_id count is what catches it.
4. **Unparsable answers.** Those score 0 and are indistinguishable from wrong answers in
   the accuracy, but they are a prompt problem rather than a retrieval one.

Exits non-zero on (1) alone: it is the only one that invalidates the run outright.

Usage:
    python video_qa/eval/check_fpsbench_stream.py \
        --save_dir results/llava_ov_7b/fpsbench_stream/64-1.0
"""

import os
import math
import argparse

import pandas as pd


def check_no_lookahead(df):
    """Abort unless every question saw only frames at or before its trigger."""
    needed = {'n_frames_seen', 'trigger_time_sec', 'sample_fps'}
    if not needed <= set(df.columns):
        print('WARNING: results.csv predates the lookahead columns '
              '(n_frames_seen/trigger_time_sec/sample_fps); skipping the audit.')
        return
    allowed = (df['trigger_time_sec'] * df['sample_fps']).apply(math.floor) + 1
    violations = df[df['n_frames_seen'] > allowed]
    if len(violations):
        cols = [c for c in ['video_id', 'question_id', 'trigger_time_sec', 'n_frames_seen']
                if c in violations.columns]
        raise SystemExit(
            f'ABORT: {len(violations)} questions were answered against frames from beyond '
            f'their own trigger. These results are invalid -- accuracy is inflated by '
            f'future information. First few:\n'
            f'{violations[cols].head(5).to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'lookahead    : OK ({len(df)} questions, none saw frames past their trigger)'
          + (f'; {n_trunc} hit the end of a short stream' if n_trunc else ''))


def check_evidence_arrived(df):
    """Rows asked before their needle had finished arriving."""
    if not {'needle_last_frame', 'n_frames_seen'} <= set(df.columns):
        return
    early = df[df['needle_last_frame'] >= df['n_frames_seen']]
    if not len(early):
        print('evidence     : OK (every needle had fully arrived when its question fired)')
        return
    # Expected under the control arm, where the question fires at the certificate end and
    # the needle clip runs past it; a defect anywhere else.
    query_arm = ('trigger' in df.columns
                 and (df.loc[early.index, 'trigger'] == 'query').all())
    note = ' -- expected under --trigger query' if query_arm else \
           ' -- NOT expected under --trigger end; check the annotation timeline'
    print(f'evidence     : {len(early)}/{len(df)} questions fired before their needle '
          f'finished arriving{note}')


def check_rows(df, expected):
    """Duplicate or missing questions, which a bad chunk merge produces silently."""
    if 'question_id' not in df.columns:
        return
    dupes = int(df['question_id'].duplicated().sum())
    line = f'rows         : {len(df)} questions, {df["question_id"].nunique()} distinct'
    if expected:
        line += f', {expected} expected'
    print(line)
    if dupes:
        print(f'  WARNING: {dupes} duplicate question_id -- the per-chunk CSVs were '
              f'merged more than once, and every mean above is weighted by the overlap')
    if expected and len(df) != expected:
        print(f'  WARNING: {abs(expected - len(df))} rows {"missing" if len(df) < expected else "extra"} '
              f'against --expected: a chunk died before writing, and the missing questions '
              f'are not a random subset')


def check_parsing(df):
    """Responses with no option letter in them."""
    if 'pred_choice' not in df.columns:
        return
    unparsed = int(df['pred_choice'].isna().sum())
    if unparsed:
        print(f'parsing      : {unparsed}/{len(df)} responses had no parsable option '
              f'letter -- those score 0, but as a prompt failure rather than a retrieval one')
    else:
        print(f'parsing      : OK (every response committed to a letter)')


def check_retrieval(df):
    """Whether retrieval ran at all -- the premise of the whole benchmark."""
    if 'retrieval_fired' not in df.columns:
        return
    fired = df['retrieval_fired'].fillna(False).astype(bool)
    if fired.all():
        print(f'retrieval    : OK (ran on all {len(df)} questions)')
    elif not fired.any():
        print('retrieval    : never ran -- every stream fit inside n_local, so this run '
              'measures the backbone over a long context, not ReKV\'s memory. Lower '
              '--n_local or raise --sample_fps.')
    else:
        print(f'retrieval    : ran on {int(fired.sum())}/{len(df)} questions; the rest fit '
              f'inside n_local and say nothing about retrieval')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None,
                        help='Override the input CSV (default: <save_dir>/results.csv).')
    parser.add_argument('--expected', type=int, default=0,
                        help='Question count this run should have produced (990 for the '
                             'full release). 0 skips the count check.')
    args = parser.parse_args()

    path = args.results_path or os.path.join(args.save_dir, 'results.csv')
    df = pd.read_csv(path)
    print(f'{path}')
    check_no_lookahead(df)
    check_evidence_arrived(df)
    check_rows(df, args.expected)
    check_parsing(df)
    check_retrieval(df)


if __name__ == '__main__':
    main()
