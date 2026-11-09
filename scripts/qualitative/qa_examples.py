"""Token retention on one question TTS got right and one it got wrong (same benchmark).

Companion to token_retention.py: same overlay, but the clips are chosen by outcome
instead of by motion. Each row is one question, showing the frames leading up to the
moment it was asked, with pruned patches dimmed -- so a reader can see whether the
evidence the question needs was kept or dropped.

Default pair: StreamingBench (Real) sample_263, an onboard racing video, both questions
answered correctly by the unpruned baseline:
    correct    t=185s  "What is the current track length displayed on the screen?"
    incorrect  t=276s  "What color is the current helmet of the driver?"  (Green -> Yellow)
So the incorrect row is a failure pruning caused, not one the model had anyway.

The masks are the eval's own, not an approximation. StreamingBench is streamed from frame
0 with one pruner carried across the whole video (video_qa/rekv_streamingbench_vqa.py), so
this script ingests every frame from 0 up to the question on the same 1 fps grid and runs
the registered rlt_ref pruner over all of them. It then checks the cumulative kept-token
count against `tokens_kept` in the eval's results.csv for that question; a mismatch fails
the run instead of drawing a picture of a different pruner state (a few tokens of slack
are allowed for fp noise on a different GPU).

Output: <out>.png / <out>.pdf, and <out>_masks.npz with the masks for every frame up to
each question.

Usage (needs a GPU; see qa_examples.slurm):
    python scripts/qualitative/qa_examples.py
    python scripts/qualitative/qa_examples.py \\
        --correct "Real-Time Visual Understanding_sample_2_2" \\
        --incorrect "Real-Time Visual Understanding_sample_2_1"
    python scripts/qualitative/qa_examples.py --list      # CPU: print candidate pairs
"""

import os
import sys
import ast
import json
import argparse
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from token_retention import GRID, N_TOK, load_frames, token_features, keep_masks, overlay  # noqa: E402

RESULTS = 'results/llava_ov_7b/streamingbench_real'
LETTERS = 'ABCDEFGH'


def load_outcomes(results, threshold):
    """Pruned rows joined with the baseline's answer, one row per question."""
    base = pd.read_csv(f'{results}/64-1.0/results.csv')
    pruned = pd.read_csv(f'{results}/64-1.0-rlt_ref{threshold:g}cosine/results.csv')
    m = pruned.merge(base[['question_id', 'qa_acc', 'pred_choice']], on='question_id',
                     suffixes=('', '_base'))
    m['ok'] = m.qa_acc == 100
    m['ok_base'] = m.qa_acc_base == 100
    return m.set_index('question_id')


def list_pairs(m, top=15):
    """Videos with a question both runs got right and one only pruning got wrong."""
    good = m[m.ok & m.ok_base]
    bad = m[~m.ok & m.ok_base]
    vids = sorted(set(good.video_id) & set(bad.video_id))
    print(f'{len(vids)} videos with a correct and a pruning-caused incorrect question')
    for v in vids[:top]:
        print(f'== {v}')
        for qid, r in m[(m.video_id == v) & m.ok_base].iterrows():
            print(f'   {"OK " if r.ok else "BAD"} t={r.realtime:>4.0f}s  {qid!r}  [{r.task}] '
                  f'{r.question}')


def run_question(model, anno, r, qid, threshold):
    """Frames 0..question time, their keep masks, and the check against the eval."""
    n = int(r.n_frames_seen)
    frames, times = load_frames(anno[r.video_id], 0, n, 1.0)
    assert len(frames) == n, f'{qid}: loaded {len(frames)} frames, eval saw {n}'
    # One frame per forward pass, as the streaming eval does: batching changes the vision
    # tower's fp noise, which flips the few tokens sitting on the threshold.
    masks, _ = keep_masks(token_features(model, frames, batch=1), threshold)
    kept = int(masks.sum())
    # Bit-exact only on the eval's own GPU: another device's fp noise still flips a handful
    # of tokens on the threshold (measured: 2 of 36456). Anything past 0.1% is a real
    # mismatch -- wrong frames, wrong pruner -- not noise.
    assert masks.size == int(r.tokens_seen), \
        f'{qid}: {masks.size} tokens seen, eval logged {int(r.tokens_seen)}'
    diff = kept - int(r.tokens_kept)
    assert abs(diff) <= 1e-3 * masks.size, \
        f'{qid}: reproduced {kept}/{masks.size} tokens kept, eval logged {int(r.tokens_kept)}'
    print(f'{qid}: {kept}/{masks.size} tokens kept, eval logged {int(r.tokens_kept)} '
          f'({diff:+d})')
    return frames, times, masks


def row_title(r, threshold, width):
    choices = ast.literal_eval(r.choices)
    opt = lambda c: f'{c}) {choices[LETTERS.index(c)].rstrip(".")}' if c in LETTERS else str(c)
    mark = 'Correct' if r.ok else 'Incorrect'
    q = textwrap.shorten(r.question, width=width, placeholder='...')
    return (f'{mark}  [{r.task}, asked at t={r.realtime:.0f}s]  Q: {q}\n'
            f'Ground truth: {opt(r.correct_choice)}    Baseline: {opt(r.pred_choice_base)}    '
            f'TTS @ {threshold:g}: {opt(r.pred_choice)}')


def draw(rows, threshold, window, n_show, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = n_show + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.3 * ncol, 2.35 * len(rows)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.62), squeeze=False)
    for i, (r, frames, times, masks) in enumerate(rows):
        T = len(frames)
        lo = max(0, T - 1 - window)
        show = np.unique(np.linspace(lo, T - 1, n_show).round().astype(int))
        for k in range(ncol - 1):
            ax = axes[i, k]
            ax.set_xticks([]); ax.set_yticks([])
            if k >= len(show):
                ax.axis('off'); continue
            t = show[k]
            ax.imshow(overlay(frames[t], masks[t]))
            ax.set_title(f't={times[t]:.0f}s  kept {int(masks[t].sum())}/{N_TOK}',
                         fontsize=8, pad=2)
        ax = axes[i, -1]
        freq = masks[lo:].mean(0).reshape(GRID, GRID)
        fh, fw = frames.shape[1:3]
        im = ax.imshow(freq, cmap='magma', vmin=0, vmax=1, interpolation='nearest',
                       extent=[0, fw, fh, 0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'keep freq. ({100 * masks[lo:].mean():.1f}% kept)', fontsize=8, pad=2)
        color = '#1a7f37' if r.ok else '#c62828'
        # Title over the whole row, left-aligned with the first frame.
        box = axes[i, 0].get_position()
        fig.text(box.x0, box.y1 + 0.052, row_title(r, threshold, 110), fontsize=9,
                 va='bottom', ha='left', color=color, linespacing=1.4)
    cb = fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    for ext in ('.png', '.pdf'):
        fig.savefig(out + ext, dpi=200, bbox_inches='tight', pad_inches=0.05)
    print(f'wrote {out}.png / {out}.pdf')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', default='llava_ov_7b')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--results', default=RESULTS)
    p.add_argument('--anno', default='data/StreamingBench/real.json')
    p.add_argument('--correct', default='Real-Time Visual Understanding_sample_263_3')
    p.add_argument('--incorrect', default='Real-Time Visual Understanding_sample_263_4')
    p.add_argument('--window', type=int, default=10,
                   help='seconds before the question shown in each row')
    p.add_argument('--n_show', type=int, default=6)
    p.add_argument('--list', action='store_true', help='CPU: print candidate pairs and exit')
    p.add_argument('--out', default='figures/qa_examples')
    args = p.parse_args()

    m = load_outcomes(args.results, args.threshold)
    if args.list:
        list_pairs(m)
        return
    for qid, want in ((args.correct, True), (args.incorrect, False)):
        assert qid in m.index, f'no such question id: {qid!r} (try --list)'
        assert m.loc[qid, 'ok'] == want, \
            f'{qid!r} was {"correct" if m.loc[qid, "ok"] else "incorrect"} at {args.threshold:g}'

    anno = {x['video_id']: x['video_path'] for x in json.load(open(args.anno))}
    from video_qa.base import MODELS
    spec = MODELS[args.model]
    assert args.model.startswith('llava_ov'), 'the 14x14 geometry here is LLaVA-OV specific'
    model, _ = spec['load_func'](model_path=spec['model_path'], n_local=15000)

    rows = []
    for qid in (args.correct, args.incorrect):
        r = m.loc[qid]
        frames, times, masks = run_question(model, anno, r, qid, args.threshold)
        rows.append((r, frames, times, masks))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    draw(rows, args.threshold, args.window, args.n_show, args.out)
    np.savez_compressed(args.out + '_masks.npz',
                        **{f'{"correct" if r.ok else "incorrect"}_masks': mk
                           for r, _, _, mk in rows},
                        questions=np.asarray([args.correct, args.incorrect]))
    print(f'wrote {args.out}_masks.npz')


if __name__ == '__main__':
    main()
