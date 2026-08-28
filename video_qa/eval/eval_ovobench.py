"""Score an OVO-Bench run and export it in OVO-Bench's own result format.

Two things this does that `eval_multiple_choice.py` cannot:

1. **OVO-Bench's aggregation is not a micro-average.** The headline number is the mean of
   the per-task accuracies within a mode -- so OCR (149 queries) and FPD (101) count
   equally. Pooling all rows instead would silently reweight the benchmark.
2. **It verifies the no-leak invariant before reporting anything.** Every row must satisfy
   `n_frames_seen <= floor(realtime * sample_fps) + 1`. If a conversion or a solver change
   ever lets a query see past its own timestamp, the accuracy still looks plausible -- it
   just goes up. Scoring aborts instead.

Both a strict-letter accuracy (ReKV's convention) and OVO-Bench's own substring rule
(`int(gt_letter in response)`, utils/OVOBenchScore.py) are reported. They should agree;
where they diverge, the substring rule is the looser of the two -- it credits a verbose
answer that happens to contain the letter -- so a gap between them is worth reading as a
prompt-adherence problem, not as free accuracy.

The exported `ovo_results.json` is exactly what OVO-Bench's scorer consumes, so the number
below can be reproduced with their code rather than trusted from ours.
"""

import os
import json
import math
import argparse

import pandas as pd

BACKWARD_TASKS = ['EPM', 'ASI', 'HLD']
REALTIME_TASKS = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
MODES = [('backward', BACKWARD_TASKS), ('realtime', REALTIME_TASKS)]


def check_no_leak(df):
    """Abort unless every query saw only frames at or before its own timestamp."""
    if not {'n_frames_seen', 'realtime', 'sample_fps'} <= set(df.columns):
        print('WARNING: results.csv predates the leak-check columns '
              '(n_frames_seen/realtime/sample_fps); skipping the audit.')
        return
    allowed = (df['realtime'] * df['sample_fps']).apply(math.floor) + 1
    violations = df[df['n_frames_seen'] > allowed]
    if len(violations):
        rows = violations[['video_id', 'question_id', 'realtime', 'n_frames_seen']].head(5)
        raise SystemExit(
            f'ABORT: {len(violations)} queries were answered against frames from beyond '
            f'their own realtime timestamp. These results are invalid -- accuracy is '
            f'inflated by future information. First few:\n{rows.to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'leak check: OK ({len(df)} queries, none saw future frames)'
          + (f'; {n_trunc} queries hit the end of a short video' if n_trunc else ''))


def ovo_substring_score(row):
    """OVO-Bench's own rule: the gold letter appearing anywhere in the raw response."""
    response = row['pred_answer']
    if not isinstance(response, str):
        return 0
    return int(str(row['correct_choice']) in response)


def summarise(df, label, score_col):
    """Per-task accuracy, then the unweighted per-mode mean OVO-Bench reports."""
    per_task = (df.groupby('task')[score_col].mean() * (100 if score_col == 'ovo_score' else 1))
    print(f'\n=== {label} ===')
    summary = {}
    mode_means = []
    for mode, tasks in MODES:
        present = [t for t in tasks if t in per_task.index]
        if not present:
            continue
        for t in present:
            n = int((df['task'] == t).sum())
            print(f'  {mode:<9} {t:<4} {per_task[t]:6.2f}  (n={n})')
            summary[t] = round(float(per_task[t]), 2)
        mode_mean = float(sum(per_task[t] for t in present) / len(present))
        summary[f'{mode}_average'] = round(mode_mean, 2)
        mode_means.append(mode_mean)
        print(f'  {mode:<9} AVG  {mode_mean:6.2f}  (unweighted mean of {len(present)} tasks)')
    if len(mode_means) > 1:
        summary['average'] = round(sum(mode_means) / len(mode_means), 2)
        print(f'  overall        {summary["average"]:6.2f}  (mean of mode averages)')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()

    results_path = args.results_path or os.path.join(args.save_dir, 'results.csv')
    save_dir = os.path.dirname(results_path) if args.results_path else args.save_dir
    df = pd.read_csv(results_path)

    check_no_leak(df)

    df['ovo_score'] = df.apply(ovo_substring_score, axis=1)
    strict = summarise(df, 'strict letter match (ReKV convention)', 'qa_acc')
    loose = summarise(df, "OVO-Bench substring rule (gt letter in response)", 'ovo_score')

    # Unparseable generations. get_prompt(mc=True) primes 'Best option: (', so this should
    # be ~0; a non-trivial rate means the answers are not being read the way either scorer
    # assumes and both numbers above are suspect.
    letters = set('ABCDEFGH')
    n_bad = sum(1 for a in df['pred_answer']
                if not isinstance(a, str) or not a.strip() or a.strip()[0] not in letters)
    print(f'\nunparseable responses: {n_bad}/{len(df)} ({100 * n_bad / len(df):.2f}%)')

    scores_path = os.path.join(save_dir, 'ovobench_scores.json')
    with open(scores_path, 'w') as f:
        json.dump({'strict_letter': strict, 'ovo_substring': loose,
                   'n_queries': len(df), 'unparseable': n_bad}, f, indent=2)

    # OVO-Bench's own result format, so their scorer can be run over this verbatim:
    #   python -c "from utils.OVOBenchScore import calculate_score_backward_realtime as f; \
    #              import json; print(f(json.load(open('ovo_results.json')))[1])"
    export_path = os.path.join(save_dir, 'ovo_results.json')
    with open(export_path, 'w') as f:
        json.dump([{'id': int(r['question_id']), 'task': r['task'], 'question': r['question'],
                    'response': r['pred_answer'] if isinstance(r['pred_answer'], str) else None,
                    'ground_truth': r['correct_choice']}
                   for _, r in df.iterrows()], f, indent=2)
    print(f'\nwrote {scores_path}\nwrote {export_path}')


if __name__ == '__main__':
    main()
