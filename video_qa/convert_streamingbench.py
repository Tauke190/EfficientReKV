"""Convert StreamingBench's per-question CSVs into ReKV's video-level schema.

StreamingBench is the same shape as OVO-Bench's realtime split -- a query carries a
`time_stamp` and may only be answered from frames at or before it -- so the output here
is deliberately the schema video_qa/rekv_ovobench_vqa.py already consumes: one record per
*video*, questions sorted by timestamp, `gt_index` decided at conversion time. That solver
then runs StreamingBench unmodified, ingesting each video once and questioning it five
times instead of re-encoding a truncated clip per query (4250 encodes -> 850).

Four subsets, the same split StreamingBench's own runner uses (scripts/stats.sh takes
--task real/omni/sqa/proactive). Proactive Output is deliberately left out: it asks the
model to emit a string at the right *moment*, scored against a ground-truth timestamp,
which is not multiple choice and needs its own solver.

    real     Real_Time_Visual_Understanding.csv   2500 q / 500 videos / 10 tasks
    omni     Omni_Source_Understanding.csv        1000 q / 200 videos /  4 tasks
    context  Contextual_Understanding.csv          500 q / 100 videos /  2 tasks (ACU, MCR)
    sqa      Sequential_Question_Answering.csv     250 q /  50 videos /  1 task

`context` and `sqa` together are the paper's Contextual Understanding group; they are kept
apart because they are run under different protocols -- SQA hands the model the previous
questions and their gold answers (StreamingBenchSQA.py), the others do not.

Four properties of the source CSVs that this script has to repair, all verified against
the 4250 rows:

* **`answer` is a letter, and the options carry their own letter prefixes.** "A. FIFA and
  La Liga." would render as "(A) A. FIFA and La Liga." through base.py's
  `format_mcqa_prompt`, so the prefix is stripped and `gt_index` comes from the letter.
  100 Sequential-QA rows and 18 Omni rows carry no prefix at all; those are left alone.
* **9 Omni rows have an option split across two list entries** by a newline in the source
  ("C. A leopard lies in wait in a ravine as an antelope grazes" / "above it. The narrator
  says ..."). An entry that does not open with a letter prefix is a continuation of the
  previous one and is rejoined, which is why the repair is by prefix and not by position.
* **2 Omni rows repeat the whole option list** (7 and 6 entries, letters A,B,C,D,B,C,D).
  First occurrence of each letter wins.
* **14 Real-Time rows write the timestamp as MM:SS** rather than HH:MM:SS. Read as MM:SS
  they land inside their videos; read as HH:MM they would not.

`Proactive Output` rows in Proactive_Output.csv sometimes use the prefix `Active Output`
for the same videos, and three PO sample directories name their file `Active Output_N.mp4`
instead of `video.mp4`. Neither affects the three subsets here; the video lookup globs for
an mp4 anyway so it would not trip over the latter.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_streamingbench.py --subset real \
        --out data/StreamingBench/real.json
"""

import os
import re
import csv
import ast
import glob
import json
import argparse
from collections import Counter, defaultdict

LETTERS = 'ABCDEFGH'

SUBSETS = {
    'real': ['Real_Time_Visual_Understanding.csv'],
    'omni': ['Omni_Source_Understanding.csv'],
    'context': ['Contextual_Understanding.csv'],
    'sqa': ['Sequential_Question_Answering.csv'],
}

# question_id looks like "<category>_sample_<N>_<q>"; <category> is also the directory
# scripts/dataset_prep/setup_streamingbench.py extracted that zip into.
QID_RE = re.compile(r'^(?P<category>.+)_sample_(?P<sample>\d+)_(?P<q>\d+)$')
PREFIX_RE = re.compile(r'^\s*([A-H])[.)]\s*')


def parse_timestamp(text):
    """Seconds from 'HH:MM:SS' or the 14 rows written 'MM:SS'."""
    parts = [p.strip() for p in str(text).strip().split(':')]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    else:
        raise ValueError(f'unparseable time_stamp {text!r}')
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalise_options(raw):
    """Return four option strings with their letter prefixes removed.

    A four-entry list is read positionally -- entry i *is* option i -- and a prefix is
    stripped from each entry only if it has one. Position is the only trustworthy signal
    there, because the prefixes themselves are unreliable: one Omni row labels two options
    'B.', one Sequential-QA row labels two 'B.', and another prefixes just its first entry
    ('A.Lebrun.', then three bare names). Keying on the letters instead would collapse
    those lists.

    Only a list longer than four can have been split by a stray newline, and only there is
    the prefix worth trusting: an entry that does not open with one is glued back onto the
    entry it was severed from, and a letter seen twice is a repeated copy of the whole
    list, so the first occurrence wins.
    """
    items = [str(x) for x in raw]
    if len(items) <= 4:
        return [PREFIX_RE.sub('', x, count=1).strip() for x in items]

    by_letter = {}
    order = []
    current = None
    for item in items:
        m = PREFIX_RE.match(item)
        if m:
            current = m.group(1)
            if current in by_letter:  # a repeated list: keep the first copy
                current = None
                continue
            by_letter[current] = PREFIX_RE.sub('', item, count=1).strip()
            order.append(current)
        elif current is not None:
            # A continuation line: glue it back onto the option it was split from.
            by_letter[current] = (by_letter[current] + ' ' + item.strip()).strip()
    return [by_letter[l] for l in order]


def find_video(video_root, category, sample):
    """`<category>/sample_<N>/video.mp4`, or the single mp4 there if it is named oddly."""
    d = os.path.join(video_root, category, f'sample_{sample}')
    default = os.path.join(d, 'video.mp4')
    if os.path.exists(default):
        return default
    others = sorted(glob.glob(os.path.join(d, '*.mp4')))
    return others[0] if others else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, required=True, choices=sorted(SUBSETS))
    parser.add_argument('--csv_root', type=str, default='data/StreamingBench',
                        help='Directory holding the six StreamingBench CSVs.')
    parser.add_argument('--video_root', type=str, default='data/StreamingBench/videos',
                        help='Video tree written by scripts/dataset_prep/setup_streamingbench.py.')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose video file is absent instead of dropping '
                             'them. For inspecting the conversion before the zips are '
                             'unpacked; an eval run over these would crash on load.')
    args = parser.parse_args()

    rows = []
    for name in SUBSETS[args.subset]:
        path = os.path.join(args.csv_root, name)
        with open(path, newline='') as f:
            rows.extend(csv.DictReader(f))
    print(f'subset {args.subset}: {len(rows)} rows from {SUBSETS[args.subset]}')

    by_video = defaultdict(list)
    for r in rows:
        m = QID_RE.match(r['question_id'].strip())
        if m is None:
            raise SystemExit(f'unparseable question_id {r["question_id"]!r}')
        by_video[(m.group('category'), int(m.group('sample')))].append((r, int(m.group('q'))))

    anno, missing, n_queries = [], [], 0
    for (category, sample), group in sorted(by_video.items()):
        video_path = find_video(args.video_root, category, sample)
        if not os.path.exists(video_path):
            missing.append(f'{category}/sample_{sample}')
            if not args.allow_missing:
                continue

        # Ascending timestamp is a correctness requirement, not a nicety: the streaming
        # solver only moves its ingestion cursor forward, so a question placed out of order
        # would be answered against frames from beyond its own timestamp -- a future-
        # information leak that inflates accuracy while looking perfectly normal. The
        # in-file question index breaks ties, keeping the conversion deterministic and
        # Sequential-QA's intended order intact where two questions share a second.
        group = sorted(group, key=lambda rq: (parse_timestamp(rq[0]['time_stamp']), rq[1]))

        conversations = []
        for r, qidx in group:
            options = normalise_options(ast.literal_eval(r['options']))
            letter = r['answer'].strip()
            if letter not in LETTERS:
                raise SystemExit(f'{r["question_id"]}: answer {letter!r} is not a letter')
            gt = LETTERS.index(letter)
            if gt >= len(options):
                raise SystemExit(f'{r["question_id"]}: answer {letter} but only '
                                 f'{len(options)} options survived normalisation')
            t = parse_timestamp(r['time_stamp'])
            conversations.append({
                'question_id': r['question_id'],
                'question': r['question'],
                'choices': options,
                # Kept consistent with gt_index so a code path that recovers the letter by
                # text lookup agrees with one that uses the index -- options are not
                # unique across the benchmark (two Sequential-QA rows repeat a string).
                'answer': options[gt],
                'gt_index': gt,
                'reference_answer': letter,
                'question_type': r['task_type'].strip(),
                'realtime': t,
                # The solver gates ingestion on end_time alone; start_time is carried only
                # to match the schema of the other datasets.
                'start_time': 0,
                'end_time': t,
                # Upstream metadata, passed through for slicing results afterwards:
                # whether the answer needs one frame or several, and whether the evidence
                # precedes the query ('Prior') or the model must wait for it.
                'frames_required': r.get('frames_required'),
                'temporal_clue_type': r.get('temporal_clue_type'),
            })
        n_queries += len(conversations)
        anno.append({
            'video_id': f'{category}__sample_{sample}'.replace(' ', '_'),
            'video_path': video_path,
            'source_video': os.path.relpath(video_path, args.video_root),
            'conversations': conversations,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(anno, f, indent=1)

    print(f'{n_queries} queries over {len(anno)} videos -> {args.out}')
    print('per-task:', dict(sorted(Counter(
        c['question_type'] for v in anno for c in v['conversations']).items())))
    n_choices = Counter(len(c['choices']) for v in anno for c in v['conversations'])
    print('options per question:', dict(sorted(n_choices.items())))
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'DROPPED'
        print(f'{len(missing)} videos not found under {args.video_root} and {verb}; '
              f'e.g. {missing[:3]}')


if __name__ == '__main__':
    main()
