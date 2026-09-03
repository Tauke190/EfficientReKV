"""Break a judged StreamBench run down by question class and source, and audit the stream.

StreamBench's answers are free-form, so the *grading* is done by an LLM judge
(eval_open_ended_local.py, `--judge_preset streambench` = the Meta-Llama-3-8B-Instruct
judge upstream uses). This script runs after it and does the two things that pooled
accuracy cannot:

1. **It verifies the no-leak invariant.** Every breakpoint carries a `time` past which the
   model may not see, and Long-term / Short-term Memory Search and Object Search are all
   about what is no longer on screen. A solver that shows the whole video still produces a
   plausible accuracy; it just goes up. So every row must satisfy
   `n_frames_seen <= floor(realtime * sample_fps) + 1`, and this refuses to report
   otherwise.
2. **It splits the six classes.** They are near-evenly balanced (~300 each) and are not
   the same task. KG -- Knowledge-based QA -- is world knowledge with no visual content
   ("From which ingredient is sesame oil extracted?"), answerable with the video switched
   off; it is 16% of the set and will sit far above the rest, pulling the pooled figure up
   for a reason that has nothing to do with streaming. Reading KG separately from the
   memory classes is the whole point.

Class names are upstream's (StreamChat README): OS Object Search, LM Long-term Memory
Search, SM Short-term Memory Search, CI Conversational Interaction, KG Knowledge-based
Question Answering, SF Simple Factual.

The judge writes one verdict file per item into its cache directory, keyed
`{video_id}_{n}` where n counts occurrences of that video_id **in results.csv row order**
(eval_open_ended_local.build_prediction_set). This rebuilds the same keys from the same
CSV to rejoin verdicts to rows, so nothing here depends on the judge emitting metadata it
does not carry.

Usage (run_eval does this for you):
    python video_qa/eval/eval_streambench.py --save_dir results/<model>/streambench/64-1.0
"""

import os
import ast
import json
import math
import argparse
from collections import Counter, defaultdict

import pandas as pd

# Upstream's abbreviations, expanded. Order is deliberate: the memory/perception classes
# the benchmark exists to test first, KG last because it is the one answerable blind.
CLASS_NAMES = [
    ('SF', 'Simple Factual'),
    ('OS', 'Object Search'),
    ('SM', 'Short-term Memory Search'),
    ('LM', 'Long-term Memory Search'),
    ('CI', 'Conversational Interaction'),
    ('KG', 'Knowledge-based QA (answerable blind)'),
]


def check_no_leak(df):
    """Abort unless every question saw only frames at or before its own timestamp."""
    needed = {'n_frames_seen', 'realtime', 'sample_fps'}
    if not needed <= set(df.columns):
        raise SystemExit(
            'ABORT: results.csv has no leak-check columns (n_frames_seen/realtime/'
            'sample_fps). That means it was produced by a solver that does not gate on '
            'the breakpoint time -- these results are not a StreamBench score. Re-run '
            'with --dataset streambench, which uses rekv_streambench_vqa.')
    allowed = (df['realtime'] * df['sample_fps']).apply(math.floor) + 1
    violations = df[df['n_frames_seen'] > allowed]
    if len(violations):
        rows = violations[['video_id', 'question_id', 'realtime', 'n_frames_seen']].head(5)
        raise SystemExit(
            f'ABORT: {len(violations)} questions were answered against frames from beyond '
            f'their own breakpoint time. These results are invalid -- accuracy is inflated '
            f'by future information. First few:\n{rows.to_string(index=False)}')
    n_trunc = int(df['truncated'].sum()) if 'truncated' in df.columns else 0
    print(f'leak check: OK ({len(df)} questions, none saw future frames)'
          + (f'; {n_trunc} hit the end of a short video' if n_trunc else ''))


def judge_keys(df):
    """Rebuild eval_open_ended_local's per-item cache keys, in CSV row order."""
    counts, keys = Counter(), []
    for vid in df['video_id']:
        keys.append(f'{vid}_{counts[vid]}')
        counts[vid] += 1
    return keys


def load_verdicts(cache_dir, keys):
    """key -> {'pred','score'} for the items the judge managed to score."""
    out = {}
    for key in keys:
        path = os.path.join(cache_dir, f'{key}.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
        except (ValueError, OSError):
            continue
        verdict = payload[0] if isinstance(payload, list) and payload else payload
        if isinstance(verdict, dict) and verdict.get('pred') is not None \
                and verdict.get('score') is not None:
            out[key] = verdict
    return out


def find_cache_dir(save_dir, explicit=None):
    """The judge's per-item cache. `tmp_local_streambench/` by default (judges.PRESETS)."""
    if explicit:
        return explicit
    candidates = sorted(d for d in os.listdir(save_dir)
                        if d.startswith('tmp') and os.path.isdir(os.path.join(save_dir, d)))
    if not candidates:
        raise SystemExit(
            f'ABORT: no judge cache directory under {save_dir}. Run the judge first '
            f'(run_eval does this automatically); expected e.g. tmp_local_streambench/.')
    # Prefer the StreamBench judge's own directory when several judges have run here.
    for c in candidates:
        if 'streambench' in c:
            return os.path.join(save_dir, c)
    return os.path.join(save_dir, candidates[0])


def summarise(df, label, order=None):
    """Accuracy and mean score per group, printed and returned."""
    groups = order or sorted(df[label].dropna().unique())
    out = {}
    width = max(len(str(g)) for g in groups) if groups else 10
    print(f"\n{label:<{width}} {'acc':>7} {'score':>7} {'n':>6}")
    print('-' * (width + 23))
    for g in groups:
        sub = df[df[label] == g]
        if not len(sub):
            continue
        acc = 100.0 * sub['correct'].mean()
        out[str(g)] = {'accuracy': round(float(acc), 2),
                       'score': round(float(sub['score'].mean()), 3),
                       'n': int(len(sub))}
        print(f'{str(g):<{width}} {acc:7.2f} {sub["score"].mean():7.3f} {len(sub):6d}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="The judge's per-item verdict directory. Auto-detected.")
    args = parser.parse_args()

    results_path = args.results_path or os.path.join(args.save_dir, 'results.csv')
    save_dir = os.path.dirname(results_path) if args.results_path else args.save_dir
    df = pd.read_csv(results_path)

    blind = 'blind' in df.columns and bool(df['blind'].all())
    if blind:
        print('*' * 72)
        print('BLIND CONTROL: no video was shown. This is the language-prior floor,')
        print('not a StreamBench score. Compare a sighted run against it; do not report it.')
        print('*' * 72)
        print('Expect KG to be near its sighted value -- it needs no video -- while the')
        print('memory and search classes should collapse. A class that does not is one')
        print('the video was not contributing to.')
    else:
        check_no_leak(df)

    cache_dir = find_cache_dir(save_dir, args.cache_dir)
    keys = judge_keys(df)
    verdicts = load_verdicts(cache_dir, keys)
    print(f'judge cache: {cache_dir}  ({len(verdicts)}/{len(keys)} items scored)')
    if not verdicts:
        raise SystemExit(
            'ABORT: the judge cache holds no usable verdicts. Nothing to break down.')

    df = df.assign(_key=keys)
    df['correct'] = df['_key'].map(lambda k: verdicts[k]['pred'] == 'yes'
                                   if k in verdicts else None)
    df['score'] = df['_key'].map(lambda k: verdicts[k]['score'] if k in verdicts else None)
    n_unscored = int(df['correct'].isna().sum())
    scored = df.dropna(subset=['correct']).copy()
    scored['correct'] = scored['correct'].astype(bool)

    summary = {}
    if 'task' in scored.columns:
        order = [c for c, _ in CLASS_NAMES if c in set(scored['task'])]
        summary['per_class'] = summarise(scored, 'task', order)
        for code, name in CLASS_NAMES:
            if code in summary['per_class']:
                summary['per_class'][code]['name'] = name
    if 'source' in scored.columns and scored['source'].notna().any():
        summary['per_source'] = summarise(scored, 'source')

    micro = 100.0 * scored['correct'].mean()
    per_class = summary.get('per_class', {})
    # Macro over classes, and the same excluding KG. KG is 16% of the set and needs no
    # video, so a sighted-vs-blind delta computed on the pooled figure is diluted by it.
    macro = (sum(v['accuracy'] for v in per_class.values()) / len(per_class)
             if per_class else None)
    no_kg = [v['accuracy'] for k, v in per_class.items() if k != 'KG']
    macro_no_kg = sum(no_kg) / len(no_kg) if no_kg else None

    print('\n' + '-' * 44)
    print(f"{'micro-average (pooled questions)':<32} {micro:7.2f} {len(scored):6d}")
    if macro is not None:
        print(f"{'macro-average (over classes)':<32} {macro:7.2f} {len(per_class):6d}")
        print(f"{'macro-average excluding KG':<32} {macro_no_kg:7.2f} {len(no_kg):6d}")
    print(f"{'mean judge score':<32} {scored['score'].mean():7.3f}")
    if n_unscored:
        print(f'\n{n_unscored}/{len(df)} items have no usable verdict and are excluded '
              f'from every figure above (the judge failed to parse, not the model).')

    summary.update({
        'micro_average': round(float(micro), 2),
        'macro_average': round(float(macro), 2) if macro is not None else None,
        'macro_average_no_kg': round(float(macro_no_kg), 2) if macro_no_kg is not None else None,
        'mean_score': round(float(scored['score'].mean()), 3),
        'n_questions': int(len(df)),
        'n_scored': int(len(scored)),
        'n_unscored': n_unscored,
        'judge_cache': os.path.basename(cache_dir),
        'blind': blind,
    })

    out_path = os.path.join(save_dir, 'streambench_scores.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
