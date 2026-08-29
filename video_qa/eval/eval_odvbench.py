"""Score an ODV-Bench run: per-subtask accuracy, plus the streaming no-leak audit.

Two things this does that `eval_multiple_choice.py` cannot:

1. **It verifies the no-leak invariant before reporting anything.** ODV-Bench is an online
   benchmark -- every question carries an `end_time` past which the model may not see, and
   62% of the questions ask what happens *after* that point ("What will the position box of
   the pedestrian be", "Will there be significant traffic risks in the future"). A solver
   that shows the model the whole clip still produces a plausible-looking accuracy; it just
   goes up. So every row must satisfy `n_frames_seen <= floor(end_time * sample_fps) + 1`,
   and scoring aborts rather than reporting a number that came from future frames.
2. **It reports the 12 subtasks separately.** They are wildly unbalanced (Distance
   Prediction 1488 questions, Key Information Extraction 53) and measure different things,
   so a pooled mean is dominated by one subtask. Both the pooled micro-average and the
   unweighted macro-average over subtasks are printed; the macro is the one that will not
   move just because the balance shifts.

The 2-vs-4-option split matters when reading these: 2472 questions are binary, so chance is
50% on those and 25% on the rest (~34% overall). A subtask sitting near its own chance line
is not a small deficit.
"""

import os
import json
import math
import argparse

import pandas as pd

# The three collections, and the subtasks each contributes. Fixed by the benchmark: a
# subtask never appears under more than one collection.
COLLECTIONS = [
    ('TS_Retrieval', ['Real-time Traffic Perception', 'Past Traffic Memory',
                      'Driving Decision-Making', 'Hallucination detection',
                      'Traffic Change Detection', 'Key Information Extraction']),
    ('TOI_Recognition', ['Distance Prediction', 'Location Prediction', 'Action Prediction']),
    ('TR_Analysis', ['Risk Prediction', 'Risk Analysis', 'Accident Reason Answering']),
]


def check_no_leak(df):
    """Abort unless every question saw only frames at or before its own end_time."""
    needed = {'n_frames_seen', 'realtime', 'sample_fps'}
    if not needed <= set(df.columns):
        raise SystemExit(
            'ABORT: results.csv has no leak-check columns (n_frames_seen/realtime/'
            'sample_fps). That means it was produced by an offline solver, which shows '
            'the model the whole clip -- these results are not an ODV-Bench score. '
            'Re-run with --dataset odvbench, which uses rekv_ovobench_vqa.')
    allowed = (df['realtime'] * df['sample_fps']).apply(math.floor) + 1
    violations = df[df['n_frames_seen'] > allowed]
    if len(violations):
        rows = violations[['video_id', 'question_id', 'realtime', 'n_frames_seen']].head(5)
        raise SystemExit(
            f'ABORT: {len(violations)} questions were answered against frames from beyond '
            f'their own end_time. These results are invalid -- accuracy is inflated by '
            f'future information. First few:\n{rows.to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'leak check: OK ({len(df)} questions, none saw future frames)'
          + (f'; {n_trunc} hit the end of a short video' if n_trunc else ''))


def summarise(df):
    """Per-subtask accuracy, grouped by collection, with both averages."""
    per_task = df.groupby('task')['qa_acc'].mean()
    counts = df['task'].value_counts()

    summary, macro = {}, []
    print(f"\n{'subtask':<32} {'acc':>7} {'n':>6}")
    print('-' * 47)
    for collection, tasks in COLLECTIONS:
        present = [t for t in tasks if t in per_task.index]
        if not present:
            continue
        print(f'{collection}')
        for t in present:
            print(f'  {t:<30} {per_task[t]:6.2f} {counts[t]:6d}')
            summary[t] = round(float(per_task[t]), 2)
            macro.append(float(per_task[t]))
        sub = df[df['task'].isin(present)]
        summary[f'{collection}_average'] = round(float(sub['qa_acc'].mean()), 2)
        print(f"  {'-> pooled':<30} {summary[f'{collection}_average']:6.2f} {len(sub):6d}")

    # Any subtask the hardcoded map above does not know about -- a sign the annotation
    # changed under this scorer rather than something to quietly average in.
    known = {t for _, ts in COLLECTIONS for t in ts}
    for t in sorted(set(per_task.index) - known):
        print(f'  UNKNOWN SUBTASK {t:<20} {per_task[t]:6.2f} {counts[t]:6d}')
        summary[t] = round(float(per_task[t]), 2)
        macro.append(float(per_task[t]))

    summary['micro_average'] = round(float(df['qa_acc'].mean()), 2)
    summary['macro_average'] = round(sum(macro) / len(macro), 2) if macro else None
    print('-' * 47)
    print(f"{'micro-average (pooled questions)':<32} {summary['micro_average']:6.2f} "
          f"{len(df):6d}")
    print(f"{'macro-average (over subtasks)':<32} {summary['macro_average']:6.2f} "
          f"{len(macro):6d}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()

    results_path = args.results_path or os.path.join(args.save_dir, 'results.csv')
    save_dir = os.path.dirname(results_path) if args.results_path else args.save_dir
    df = pd.read_csv(results_path)

    # A blind CSV scores through the same path on purpose -- the comparison is only
    # meaningful if both numbers are computed identically -- but it must never be read as
    # a real result, including when it has been moved out of its '-blind' directory.
    blind = 'blind' in df.columns and bool(df['blind'].all())
    if blind:
        print('*' * 72)
        print('BLIND CONTROL: no video was shown. This is the language-prior floor,')
        print('not an ODV-Bench score. Compare a sighted run against it; do not report it.')
        print('*' * 72)

    check_no_leak(df)
    summary = summarise(df)

    # Unparseable generations. get_prompt(mc=True) primes 'Best option: (', so this should
    # be ~0; a non-trivial rate means neither number above is measuring what it claims.
    letters = set('ABCDEFGH')
    n_bad = sum(1 for a in df['pred_answer']
                if not isinstance(a, str) or not a.strip() or a.strip()[0] not in letters)
    print(f'\nunparseable responses: {n_bad}/{len(df)} ({100 * n_bad / len(df):.2f}%)')

    # Chance level for this exact row set: 2-option questions are a coin flip, 4-option
    # ones are not, and the mix differs per subtask. Printed so an accuracy can be read
    # against the floor it has to clear.
    if 'choices' in df.columns:
        n_opts = df['choices'].apply(lambda c: len(eval(c)) if isinstance(c, str) else None)
        chance = float((1.0 / n_opts).mean() * 100)
        print(f'chance level for this question mix: {chance:.2f}%')
        summary['chance'] = round(chance, 2)

    scores_path = os.path.join(save_dir, 'odvbench_scores.json')
    with open(scores_path, 'w') as f:
        json.dump({**summary, 'n_questions': len(df), 'unparseable': n_bad,
                   'blind': blind}, f, indent=2)
    print(f'\nwrote {scores_path}')


if __name__ == '__main__':
    main()
