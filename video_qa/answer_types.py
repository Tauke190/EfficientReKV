"""Official VStream-QA `answer_type` labels, restored.

`data/rvs/*_oe.json` is Flash-VStream's benchmark with exactly one field dropped. The QA
content is untouched -- verified identical as multisets against the official release
(rvs_ego 1465, rvs_movie 1905) -- but the conversion to ReKV's `conversations` layout keeps
only question/answer/start_time/end_time and discards `answer_type`:

    official : answer, answer_type, duration, end_time, gt_duration, id,
               question, start_time, video_id, video_name
    local    : answer,              end_time,                question, start_time

That field is the benchmark's own defence against exactly the problem a blind control
exposes. Two of its five categories are marked Y/N in the name -- "Order Judging(Y/N)" and
"Whether Something Happened(Y/N)" -- and together they are **50.6%** of RVS-Ego and 35.7%
of RVS-Movie. A Y/N item has a ~54% majority-class floor no matter how well the model sees
the video, so a single pooled accuracy averages a coin-flip with a real task. The labels
exist so results can be read per category; dropping them is what makes the pooled number
misleading, and restoring them costs nothing because the QA pairs still match one-to-one.

Resolution is by (question, answer) rather than by index: the local file regroups the
official 99 ego clips under their 10 source videos, so positions do not correspond. That
pair is unique enough to be unambiguous -- measured 0 collisions mapping to two different
types, and 0 unmatched rows, on both datasets.

The map is cached next to this module on first use so later runs need no network.

Lives in `video_qa/` rather than `blind/` because two callers need it: the blind-control
comparison, and `video_qa/eval/eval_open_ended_local.py`, which is what
`scripts/score_open_ended.sh` invokes and where most people will first see an RVS number.
A breakdown that only existed in the blind harness would leave the default scoring path
printing the pooled figure alone -- which is the reporting problem, not a fix for it.
"""

import os
import json

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'answer_types.json')

# Dataset name as used in results paths -> file in the IVGSZ/VStream-QA dataset repo.
OFFICIAL = {
    'rvs_ego': 'vstream-realtime/test_qa_ego4d.json',
    'rvs_movie': 'vstream-realtime/test_qa_movienet.json',
}

# Display order: Y/N categories first, since they are the ones a pooled number hides.
TYPE_ORDER = [
    'Order Judging(Y/N)',
    'Whether Something Happened(Y/N)',
    'Scene Summary',
    'Action Caption',
    'What event order',
]


def is_yes_no(answer_type):
    """True for the categories the benchmark itself marks as binary."""
    return '(Y/N)' in (answer_type or '')


def _key(question, answer):
    return f'{str(question).strip()}\x00{str(answer).strip()}'


def _build():
    """Fetch the official annotations and index answer_type by (question, answer)."""
    from huggingface_hub import hf_hub_download

    out = {}
    for dataset, path in OFFICIAL.items():
        local = hf_hub_download(repo_id='IVGSZ/VStream-QA', filename=path, repo_type='dataset')
        with open(local) as f:
            official = json.load(f)
        # Collisions would mean the same (question, answer) carries two types, which would
        # make the join ambiguous. Measured zero on both datasets; assert rather than pick,
        # because silently choosing one would put items in the wrong category.
        table = {}
        for item in official:
            k = _key(item['question'], item['answer'])
            prev = table.get(k)
            assert prev in (None, item['answer_type']), \
                f'{dataset}: {k!r} maps to both {prev!r} and {item["answer_type"]!r}'
            table[k] = item['answer_type']
        out[dataset] = table
    return out


def load(dataset):
    """{(question, answer) key: answer_type} for one dataset, cached on disk.

    Raises if the dataset is unknown or the labels cannot be obtained -- a silent fallback
    to "no categories" would quietly restore the pooled-only reporting this module exists
    to replace.
    """
    if dataset not in OFFICIAL:
        raise KeyError(f'no official answer_type source for {dataset!r}; '
                       f'known: {sorted(OFFICIAL)}')

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
    if dataset not in cache:
        cache = _build()
        with open(CACHE, 'w') as f:
            json.dump(cache, f)
    return cache[dataset]


def detect_dataset(pairs, min_coverage=0.9):
    """Which RVS dataset a set of (question, answer) pairs belongs to, or None.

    Resolves by content rather than by path, so a results.csv scored from an unusual
    directory still gets its breakdown, and a non-RVS benchmark (qaego4d, activitynet_qa)
    is correctly recognised as having no answer_type and skipped rather than mislabelled.
    Requires `min_coverage` of the pairs to match, so a chance overlap cannot select the
    wrong table.
    """
    pairs = list(pairs)
    if not pairs:
        return None
    best, best_cov = None, 0.0
    for dataset in OFFICIAL:
        try:
            table = load(dataset)
        except Exception:
            continue
        cov = sum(1 for q, a in pairs if _key(q, a) in table) / len(pairs)
        if cov > best_cov:
            best, best_cov = dataset, cov
    return best if best_cov >= min_coverage else None


def summarize(pairs, correct, min_coverage=0.9):
    """Per-answer_type accuracy for one arm, as printable lines. [] when not applicable.

    `pairs` and `correct` are parallel sequences of (question, answer) and bools. Returns
    [] -- silently -- for benchmarks that have no answer_type, so this can be called
    unconditionally from a scorer shared with non-RVS datasets.
    """
    pairs, correct = list(pairs), list(correct)
    dataset = detect_dataset(pairs, min_coverage)
    if dataset is None:
        return []
    types, _ = classify(dataset, pairs)

    def acc(sel):
        hits = [c for t, c in zip(types, correct) if sel(t)]
        return (len(hits), 100 * sum(hits) / len(hits)) if hits else (0, 0.0)

    lines = [f'Breakdown by official answer_type ({dataset}, from IVGSZ/VStream-QA):',
             f'  {"answer_type":<34} {"n":>5} {"share":>7} {"acc%":>7}']
    total = len(types)
    for t in TYPE_ORDER + ['UNMATCHED']:
        n, a = acc(lambda x, t=t: x == t)
        if n:
            lines.append(f'  {t:<34} {n:>5} {100 * n / total:>6.1f}% {a:>7.1f}')
    for label, sel in [('-- Y/N subtotal', is_yes_no),
                       ('-- non-Y/N subtotal', lambda x: not is_yes_no(x)),
                       ('-- perception only (Summary+Caption)',
                        lambda x: x in ('Scene Summary', 'Action Caption'))]:
        n, a = acc(sel)
        if n:
            lines.append(f'  {label:<34} {n:>5} {100 * n / total:>6.1f}% {a:>7.1f}')
    lines.append('  Y/N categories carry a ~54% majority-class floor; the pooled Accuracy '
                 'above averages them with the rest.')
    return lines


def classify(dataset, pairs):
    """Map an iterable of (question, answer) to answer_type.

    Returns (types, n_unmatched). Unmatched items are labelled 'UNMATCHED' and counted
    rather than dropped: a category table that quietly omits rows would not sum to the
    pooled number, and reconciling against the pooled number is the point.
    """
    table = load(dataset)
    types, unmatched = [], 0
    for question, answer in pairs:
        t = table.get(_key(question, answer))
        if t is None:
            t, unmatched = 'UNMATCHED', unmatched + 1
        types.append(t)
    return types, unmatched


if __name__ == '__main__':
    for ds in OFFICIAL:
        table = load(ds)
        counts = {}
        for t in table.values():
            counts[t] = counts.get(t, 0) + 1
        total = sum(counts.values())
        print(f'{ds}: {total} unique (question, answer) pairs')
        for t in TYPE_ORDER:
            if t in counts:
                print(f'   {counts[t]:5d}  ({100 * counts[t] / total:4.1f}%)  {t}')
