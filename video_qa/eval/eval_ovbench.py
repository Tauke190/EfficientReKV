"""Score an OVBench run: per-subtask accuracy, plus the streaming no-leak audit.

Two things this does that `eval_multiple_choice.py` cannot:

1. **It verifies the no-leak invariant before reporting anything.** OVBench is an online
   benchmark -- every question carries a `middle_frame_timestamp` past which the model may
   not see, and whole task families ask about what is *not* on screen at that moment: Past
   Memory (1714 questions) asks what happened earlier, Future Prediction (1083) asks what
   happens next. A solver that shows the model the whole clip still produces a
   plausible-looking accuracy; it just goes up, and Future Prediction goes up the most
   because the answer is literally in the frames it was not supposed to see. So every row
   must satisfy `n_frames_seen <= floor(realtime * sample_fps) + 1`, and scoring aborts
   rather than reporting a number that came from future frames.
2. **It reports the 16 sub-tasks separately, grouped under their 6 parent types.** They are
   badly unbalanced -- Procedure Recall 1068 questions against Action Anticipation 78, a
   14x spread -- so a pooled mean is dominated by a handful of them. Both the pooled
   micro-average and the unweighted macro-average over sub-tasks are printed; the macro is
   the one that will not move just because the balance shifts.

The option-count mix matters when reading these: 1774 of the 7090 questions are binary and
72 have three options, so chance is ~31.3% overall rather than 25%. Temporal Hallucination
Verification is where the binary questions concentrate, so its floor is far higher than the
others' -- a 55% there and a 55% on Object Trajectory are not the same result.

Mirrors video_qa/eval/eval_odvbench.py; the two differ only in the task map and in reading
`task` as the fine-grained axis.
"""

import os
import json
import math
import argparse

import pandas as pd

# answer_type -> its sub_answer_types, the same map convert_ovbench.py asserts against.
# The solver records the sub-task in `task`; the parent is recovered here, so one column
# carries both levels of the report.
TASK_GROUPS = [
    ('Temporal Perception', ['Action Sequence', 'Object Existence State', 'Step Localization']),
    ('Spatio Perception', ['Action Location', 'Object Position']),
    ('Spatio Temporal Perception', ['Action Trajectory', 'Object Trajectory']),
    ('Past Memory', ['Action Retrieval', 'Procedure Recall', 'Trajectory Retrieval']),
    ('Future Prediction', ['Action Anticipation', 'Goal/Step Prediction', 'Movement Prediction']),
    ('Temporal Hallucination Verification', ['Action Persistence', 'Object Presence',
                                             'Step Verification']),
]


def check_no_leak(df):
    """Abort unless every question saw only frames at or before its own timestamp."""
    needed = {'n_frames_seen', 'realtime', 'sample_fps'}
    if not needed <= set(df.columns):
        raise SystemExit(
            'ABORT: results.csv has no leak-check columns (n_frames_seen/realtime/'
            'sample_fps). That means it was produced by an offline solver, which shows '
            'the model the whole clip -- these results are not an OVBench score. '
            'Re-run with --dataset ovbench, which uses rekv_ovobench_vqa.')
    allowed = (df['realtime'] * df['sample_fps']).apply(math.floor) + 1
    violations = df[df['n_frames_seen'] > allowed]
    if len(violations):
        rows = violations[['video_id', 'question_id', 'realtime', 'n_frames_seen']].head(5)
        raise SystemExit(
            f'ABORT: {len(violations)} questions were answered against frames from beyond '
            f'their own middle_frame_timestamp. These results are invalid -- accuracy is '
            f'inflated by future information. First few:\n{rows.to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'leak check: OK ({len(df)} questions, none saw future frames)'
          + (f'; {n_trunc} hit the end of a short video' if n_trunc else ''))


def summarise(df):
    """Per-sub-task accuracy, grouped by answer_type, with both averages."""
    per_task = df.groupby('task')['qa_acc'].mean()
    counts = df['task'].value_counts()

    summary, macro = {}, []
    print(f"\n{'sub-task':<40} {'acc':>7} {'n':>6}")
    print('-' * 55)
    for group, tasks in TASK_GROUPS:
        present = [t for t in tasks if t in per_task.index]
        if not present:
            continue
        print(f'{group}')
        for t in present:
            print(f'  {t:<38} {per_task[t]:6.2f} {counts[t]:6d}')
            summary[t] = round(float(per_task[t]), 2)
            macro.append(float(per_task[t]))
        sub = df[df['task'].isin(present)]
        summary[f'{group}_average'] = round(float(sub['qa_acc'].mean()), 2)
        print(f"  {'-> pooled':<38} {summary[f'{group}_average']:6.2f} {len(sub):6d}")

    # Any sub-task the map above does not know about -- a sign the annotation changed under
    # this scorer rather than something to quietly average in.
    known = {t for _, ts in TASK_GROUPS for t in ts}
    for t in sorted(set(per_task.index) - known):
        print(f'  UNKNOWN SUB-TASK {t:<28} {per_task[t]:6.2f} {counts[t]:6d}')
        summary[t] = round(float(per_task[t]), 2)
        macro.append(float(per_task[t]))

    summary['micro_average'] = round(float(df['qa_acc'].mean()), 2)
    summary['macro_average'] = round(sum(macro) / len(macro), 2) if macro else None
    print('-' * 55)
    print(f"{'micro-average (pooled questions)':<40} {summary['micro_average']:6.2f} "
          f"{len(df):6d}")
    print(f"{'macro-average (over sub-tasks)':<40} {summary['macro_average']:6.2f} "
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
        print('not an OVBench score. Compare a sighted run against it; do not report it.')
        print('*' * 72)

    check_no_leak(df)
    summary = summarise(df)

    # Unparseable generations. get_prompt(mc=True) primes 'Best option: (', so this should
    # be ~0; a non-trivial rate means neither number above is measuring what it claims.
    letters = set('ABCDEFGH')
    n_bad = sum(1 for a in df['pred_answer']
                if not isinstance(a, str) or not a.strip() or a.strip()[0] not in letters)
    print(f'\nunparseable responses: {n_bad}/{len(df)} ({100 * n_bad / len(df):.2f}%)')

    # Chance level for this exact row set. OVBench mixes 2-, 3- and 4-option questions
    # (1774 / 72 / 5244 in the full set, ~31.3% overall), and the mix differs sharply per
    # sub-task, so an accuracy has to be read against the floor it actually has to clear.
    if 'choices' in df.columns:
        n_opts = df['choices'].apply(lambda c: len(eval(c)) if isinstance(c, str) else None)
        chance = float((1.0 / n_opts).mean() * 100)
        print(f'chance level for this question mix: {chance:.2f}%')
        summary['chance'] = round(chance, 2)

        by_task = (1.0 / n_opts).groupby(df['task']).mean() * 100
        spread = by_task.max() - by_task.min()
        if spread > 5:
            print(f'  per-sub-task chance ranges {by_task.min():.1f}%-{by_task.max():.1f}% '
                  f'({by_task.idxmin()} .. {by_task.idxmax()}) -- compare each sub-task '
                  f'against its own floor, not the pooled one')
            summary['chance_per_task'] = {k: round(float(v), 2) for k, v in by_task.items()}

    scores_path = os.path.join(save_dir, 'ovbench_scores.json')
    with open(scores_path, 'w') as f:
        json.dump({**summary, 'n_questions': len(df), 'unparseable': n_bad,
                   'blind': blind}, f, indent=2)
    print(f'\nwrote {scores_path}')


if __name__ == '__main__':
    main()
