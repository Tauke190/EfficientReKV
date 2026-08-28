"""Score an FPS-Bench-Stream run, and say whether retrieval was what decided it.

FPS-Bench-Stream ships an answer key, so unlike the short-clip FPSBench arm this is a real
scorer. But a single accuracy number is nearly uninterpretable on a needle-in-a-haystack
benchmark, because three different failures produce the same one:

1. the needle was never retrieved, so the model answered from the haystack and the prior;
2. it was retrieved, but the frame rate was below the question's `min_fps`, so the evidence
   was never resolvable however well retrieval worked;
3. it was retrieved and resolvable, and the model still got it wrong.

Only (3) is about the backbone; (1) is what this benchmark exists to measure and (2) is a
ceiling the sampling rate imposes before retrieval is even involved. So the report splits
them: accuracy overall, accuracy on the subset that was resolvable at all, and accuracy
conditioned on whether retrieval reached the needle -- alongside the retrieval rate itself,
which is the number a pruning sweep should be read on.

Chance is 20% (five options, 'E' always "None of the above"). A run near it has not
necessarily failed to retrieve; check `needle retrieved` before concluding anything.

Everything comes from the merged results.csv. A summary goes to `<save_dir>/results.json`
so a sweep can be assembled without re-parsing the CSVs.

Usage:
    python video_qa/eval/eval_fpsbench_stream.py \
        --save_dir results/llava_ov_7b/fpsbench_stream/64-1.0
"""

import os
import json
import argparse

import pandas as pd

CHANCE = 20.0  # five options


def pct(x):
    return f'{x:.1f}%'


def acc_by(df, col, min_rows=1):
    """Mean qa_acc per value of `col`, as an ordered list of (value, n, acc)."""
    out = []
    for value, group in df.groupby(col, dropna=False):
        if len(group) >= min_rows:
            out.append((value, len(group), group.qa_acc.mean()))
    return sorted(out, key=lambda r: (-r[1], str(r[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--results_path', type=str, default=None,
                        help='Override the input CSV (default: <save_dir>/results.csv).')
    args = parser.parse_args()

    pred_path = args.results_path or os.path.join(args.save_dir, 'results.csv')
    df = pd.read_csv(pred_path)
    n = len(df)
    summary = {'save_dir': args.save_dir, 'n_questions': n,
               'qa_acc': round(float(df.qa_acc.mean()), 2), 'chance': CHANCE}

    print(f'{pred_path}: {n} questions\n')
    print('accuracy')
    print(f'  overall                             : {pct(df.qa_acc.mean())}  '
          f'(chance {pct(CHANCE)})')

    # --- the ceiling the sampling rate imposes ------------------------------------
    # FPSBench's min_fps is a property of the needle, and the stream was normalised to its
    # own canvas rate, so what the model effectively got is the smaller of the two.
    if {'min_fps', 'stream_fps', 'sample_fps'} <= set(df.columns):
        effective = df[['sample_fps', 'stream_fps']].min(axis=1)
        resolvable = effective >= df.min_fps
        summary['n_resolvable'] = int(resolvable.sum())
        print(f'  on questions whose min_fps was met  : '
              f'{pct(df[resolvable].qa_acc.mean()) if resolvable.any() else "n/a"}  '
              f'({int(resolvable.sum())}/{n} rows)')
        if resolvable.any():
            summary['qa_acc_resolvable'] = round(float(df[resolvable].qa_acc.mean()), 2)
        if not resolvable.all():
            print(f'  -> {n - int(resolvable.sum())} questions were sampled below their own '
                  f'min_fps: no amount of retrieval makes those answerable, so read the '
                  f'overall number as a mixture')

    # --- was retrieval even needed --------------------------------------------------
    # A needle still inside the local window is attended directly, with true RoPE
    # positions; such a row measures the backbone over a long context and says nothing
    # about ReKV's memory. The `--trigger query` control is entirely these rows, and so is
    # part of the 'late' position bin at low frame rates. Everything below is computed on
    # the complement.
    needs = df
    if 'needle_in_local_window' in df.columns:
        local = df.needle_in_local_window.fillna(False).astype(bool)
        needs = df[~local]
        summary['n_needle_in_local_window'] = int(local.sum())
        print('\nwas retrieval needed')
        print(f'  needle already in the local window  : {int(local.sum())}/{n} rows'
              + (f'   {pct(df[local].qa_acc.mean())}' if local.any() else ''))
        print(f'  needle only reachable by retrieval  : {len(needs)}/{n} rows'
              + (f'   {pct(needs.qa_acc.mean())}' if len(needs) else ''))
        if local.any() and len(needs):
            summary['qa_acc_needle_local'] = round(float(df[local].qa_acc.mean()), 2)
            summary['qa_acc_needle_retrieved_only'] = round(float(needs.qa_acc.mean()), 2)
            print('  -> the gap between those two is what having to retrieve costs, within '
                  'this run and at this frame rate')

    # --- did retrieval reach the needle -------------------------------------------
    if 'layers_hit_frac' in needs.columns and needs.layers_hit_frac.notna().any():
        hit = needs[needs.layers_hit_frac.notna()]
        summary['n_rows_with_retrieval'] = int(len(hit))
        summary['layers_hit_frac_mean'] = round(float(hit.layers_hit_frac.mean()), 4)
        summary['blocks_hit_mean'] = round(float(hit.blocks_hit_mean.mean()), 4)
        summary['frac_needle_never_retrieved'] = round(
            float((hit.layers_hit_frac == 0).mean()), 4)
        print('\nneedle retrieved')
        print(f'  rows where retrieval ran            : {len(hit)}/{len(needs)}')
        print(f'  layers whose top-k held the needle  : {pct(100 * hit.layers_hit_frac.mean())} '
              f'(mean over rows)')
        print(f'  needle blocks per layer             : {hit.blocks_hit_mean.mean():.2f} of '
              f'{int(hit.n_blocks_per_layer.median())} retrieved')
        print(f'  needle reached by no layer at all   : {int((hit.layers_hit_frac == 0).sum())} rows')
        # How much of memory the top-k actually excluded. When k is a large fraction of the
        # blocks in memory, retrieval returns most of the video and a high hit rate is
        # arithmetic rather than a result -- which is what happens at low frame rates, where
        # a 600 s stream is only a few hundred blocks.
        if 'num_blocks' in hit.columns:
            frac = (hit.n_blocks_per_layer / hit.num_blocks.clip(lower=1)).mean()
            summary['retrieved_frac_of_memory'] = round(float(frac), 4)
            print(f'  top-k as a share of memory          : {pct(100 * frac)} of blocks')
            if frac > 0.5:
                print('  -> retrieval is barely selecting here: it returns most of the '
                      'stream, so the hit rate above is close to unconditional. Raise '
                      '--sample_fps or lower --retrieve_size before reading it as retrieval '
                      'quality')
        # The split that separates "could not find it" from "found it and still missed".
        found = hit[hit.layers_hit_frac >= 0.5]
        lost = hit[hit.layers_hit_frac < 0.5]
        if len(found):
            summary['qa_acc_needle_found'] = round(float(found.qa_acc.mean()), 2)
        if len(lost):
            summary['qa_acc_needle_lost'] = round(float(lost.qa_acc.mean()), 2)
        print(f'  accuracy | most layers found it     : '
              f'{pct(found.qa_acc.mean()) if len(found) else "n/a"}  ({len(found)} rows)')
        print(f'  accuracy | most layers did not      : '
              f'{pct(lost.qa_acc.mean()) if len(lost) else "n/a"}  ({len(lost)} rows)')
        print('  -> the gap between those two is retrieval\'s contribution; if it is near '
              'zero, accuracy here is not being decided by retrieval at all')
    elif 'retrieval_fired' in df.columns and not df.retrieval_fired.astype(bool).any():
        print('\nneedle retrieved')
        print('  retrieval never ran: every stream fit inside n_local, so this arm measures '
              'the backbone over a long context, not ReKV\'s memory')

    # --- how far back the answer was ----------------------------------------------
    if 'position_bin' in df.columns:
        print('\naccuracy by needle position (retrieval distance)')
        for value, count, acc in acc_by(df, 'position_bin'):
            dist = df[df.position_bin == value].retrieval_distance_sec.median() \
                if 'retrieval_distance_sec' in df.columns else float('nan')
            print(f'  {str(value):<8} {count:>4} rows   {pct(acc):>7}   '
                  f'median {dist:.0f}s back')
        summary['qa_acc_by_position'] = {
            str(v): round(float(a), 2) for v, _, a in acc_by(df, 'position_bin')}

    if 'task' in df.columns:
        print('\naccuracy by task')
        for value, count, acc in acc_by(df, 'task'):
            print(f'  {str(value):<28} {count:>4} rows   {pct(acc):>7}')
        summary['qa_acc_by_task'] = {
            str(v): round(float(a), 2) for v, _, a in acc_by(df, 'task')}

    # --- what the run cost ---------------------------------------------------------
    if 'token_keep_rate' in df.columns:
        summary['token_keep_rate'] = round(float(df.token_keep_rate.mean()), 4)
        print('\ntoken reduction')
        print(f'  tokens kept                         : {pct(100 * df.token_keep_rate.mean())}')
        print(f'  tokens fed per stream               : {df.n_tokens_fed.mean():.0f} mean')
    if 'pred_choice' in df.columns:
        unparsed = int(df.pred_choice.isna().sum())
        if unparsed:
            print(f'\n{unparsed} responses had no parsable option letter -- those score 0 '
                  f'and are a prompt problem, not a retrieval one')
            summary['n_unparsed'] = unparsed

    out = os.path.join(args.save_dir, 'results.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nsave_dir: {args.save_dir}')


if __name__ == '__main__':
    main()
