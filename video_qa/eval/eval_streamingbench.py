"""Score a StreamingBench run the way StreamingBench scores it.

The reference scorer (src/data/count.py) is three lines of substance: bucket by
`task_type`, compare `response[0]` -- the first character of the generation -- against the
gold letter, and report `correct/total` per task plus one pooled `total` bucket. That
pooled number is the headline, so it is a **micro**-average: Object Perception (369
queries) outweighs Prospective Reasoning (108). This differs from OVO-Bench, whose
headline is the unweighted mean of its per-task accuracies, which is why this scorer
exists instead of reusing video_qa/eval/eval_ovobench.py.

The macro mean is printed alongside it, unlabelled as the headline. It is the number to
read when comparing arms of a pruning sweep on a subset whose task sizes are lopsided --
a micro-average can move because one big task moved -- but it is not what the leaderboard
reports, so do not put it in a table next to published figures.

Three accuracies are reported per task, and they should agree:

    strict     ReKV's own extract_characters_regex, already in the CSV as qa_acc
    first_char count.py's rule: response[0] == gold letter
    substring  the gold letter appearing anywhere in the response

A gap between strict/first_char and substring is a prompt-adherence problem -- the model
is answering in prose that happens to contain the letter -- not free accuracy.

The no-leak invariant is checked before anything is reported: every query must satisfy
`n_frames_seen <= floor(realtime * sample_fps) + 1`. A solver change that let a query see
past its own timestamp would not look wrong, it would just score higher.
"""

import os
import json
import math
import argparse

import pandas as pd

# The paper's reporting groups. `context` and `sqa` are run separately (SQA carries its
# conversation history) but are averaged together as Contextual Understanding.
SUBSET_TASKS = {
    'real': ['Object Perception', 'Action Perception', 'Attribute Perception',
             'Text-Rich Understanding', 'Clips Summarize', 'Spatial Understanding',
             'Counting', 'Event Understanding', 'Causal Reasoning',
             'Prospective Reasoning'],
    'omni': ['Emotion Recognition', 'Scene Understanding', 'Source Discrimination',
             'Multimodal Alignment'],
    'context': ['Misleading Context Recognition', 'Anomaly Context Understanding'],
    'sqa': ['Sequential Question Answering'],
}


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
            f'their own timestamp. These results are invalid -- accuracy is inflated by '
            f'future information. First few:\n{rows.to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'leak check: OK ({len(df)} queries, none saw future frames)'
          + (f'; {n_trunc} queries hit the end of a short video' if n_trunc else ''))


def first_char_score(row):
    """count.py's rule: the first character of the response is the gold letter."""
    r = row['pred_answer']
    return int(isinstance(r, str) and bool(r) and r[0] == str(row['correct_choice']))


def substring_score(row):
    r = row['pred_answer']
    return int(isinstance(r, str) and str(row['correct_choice']) in r)


def summarise(df, score_col, label, scale=100):
    """Per-task accuracy, then StreamingBench's pooled headline and the macro mean.

    `scale` is 1 for qa_acc, which the solver already writes as a percentage, and 100 for
    the 0/1 columns computed here.
    """
    print(f'\n=== {label} ===')
    per_task = df.groupby('task')[score_col].mean() * scale
    summary = {}
    known = [t for tasks in SUBSET_TASKS.values() for t in tasks]
    # Declared order first so a table lines up with the paper's; anything unrecognised
    # after it, rather than dropped -- a new or misspelled task_type must stay visible.
    ordered = [t for t in known if t in per_task.index]
    ordered += [t for t in per_task.index if t not in known]
    for t in ordered:
        n = int((df['task'] == t).sum())
        print(f'  {t:<32} {per_task[t]:6.2f}  (n={n})')
        summary[t] = round(float(per_task[t]), 2)
    summary['overall'] = round(float(df[score_col].mean() * scale), 2)
    summary['macro'] = round(float(per_task.mean()), 2)
    print(f'  {"OVERALL (micro, count.py)":<32} {summary["overall"]:6.2f}  (n={len(df)})')
    print(f'  {"macro mean of tasks":<32} {summary["macro"]:6.2f}')
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

    df['first_char'] = df.apply(first_char_score, axis=1)
    df['substring'] = df.apply(substring_score, axis=1)
    scores = {
        'strict_letter': summarise(df, 'qa_acc', 'strict letter match (ReKV convention)',
                                   scale=1),  # already a percentage in the CSV
        'first_char': summarise(df, 'first_char', 'first character (StreamingBench count.py)'),
        'substring': summarise(df, 'substring', 'gold letter anywhere in the response'),
    }

    # get_prompt(mc=True) primes 'Best option: (', so this should be ~0. A non-trivial
    # rate means the answers are not being read the way any of the three scorers assumes.
    letters = set('ABCD')
    n_bad = sum(1 for a in df['pred_answer']
                if not isinstance(a, str) or not a.strip() or a.strip()[0] not in letters)
    print(f'\nunparseable responses: {n_bad}/{len(df)} ({100 * n_bad / len(df):.2f}%)')

    scores_path = os.path.join(save_dir, 'streamingbench_scores.json')
    with open(scores_path, 'w') as f:
        json.dump({**scores, 'n_queries': len(df), 'unparseable': n_bad}, f, indent=2)

    # count.py's own input format, so the number above can be reproduced with their code:
    #   python src/data/count.py --model ReKV --task real --src streamingbench_results.json
    export_path = os.path.join(save_dir, 'streamingbench_results.json')
    grouped = {}
    for _, r in df.iterrows():
        grouped.setdefault(r['video_id'], []).append({
            'question_id': r['question_id'],
            'task_type': r['task'],
            'question': r['question'],
            'answer': r['correct_choice'],
            'ReKV': r['pred_answer'] if isinstance(r['pred_answer'], str) else '',
        })
    with open(export_path, 'w') as f:
        json.dump([{'video_id': k, 'questions': v} for k, v in grouped.items()], f, indent=2)
    print(f'\nwrote {scores_path}\nwrote {export_path}')


if __name__ == '__main__':
    main()
