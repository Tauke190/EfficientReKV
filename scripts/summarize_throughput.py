"""Collapse the per-run throughput CSVs into one table.

Steady-state FPS is computed over the rows with local_window_full set, i.e. after the
sliding window has filled. Before that the model is prefilling into an empty cache and
runs faster than it can sustain, so an overall mean would overstate the streaming rate.
"""
import os
import glob
import argparse

import pandas as pd


MIN_STEADY_ROWS = 50


def steady_fps(df):
    """Sustained frames/s, or None if the run never got there.

    Returning None rather than falling back to the whole run is the point: the local
    window fills after 15000 / (196 * keep_rate) frames, so an aggressively pruned arm can
    spend an entire short run warming up. Averaging that in reports the cold start as the
    sustained rate, and it looks plausible -- it is simply too low.
    """
    if 'local_window_full' not in df:
        return None
    full = df[df['local_window_full'].astype(bool)]
    if len(full) < MIN_STEADY_ROWS:
        return None
    return full['num_frames'].sum() / full['seconds'].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/throughput')
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, '*.csv'))):
        df = pd.read_csv(path)
        if not len(df):
            continue
        fps = steady_fps(df)
        if fps is None:
            nfull = int(df['local_window_full'].astype(bool).sum()) if 'local_window_full' in df else 0
            print(f'SKIP {os.path.basename(path)}: only {nfull} steady-state rows '
                  f'(need {MIN_STEADY_ROWS}); raise NUM_FRAMES.')
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        model, vid, arm = name.rsplit('-', 2)
        rows.append({
            'model': model, 'video': vid, 'arm': arm,
            'fps': fps,
            'keep_rate': df['keep_rate'].dropna().iloc[-1] if df['keep_rate'].notna().any() else 1.0,
            'gpu_peak_gb': df['gpu_peak_bytes'].max() / 1e9,
        })
    if not rows:
        print(f'no CSVs in {args.dir}')
        return

    d = pd.DataFrame(rows)
    agg = d.groupby(['model', 'arm']).agg(
        fps=('fps', 'mean'), fps_sd=('fps', 'std'),
        keep=('keep_rate', 'mean'), gpu=('gpu_peak_gb', 'max'), n=('fps', 'size')
    ).reset_index()

    order = {'baseline': 0, 'rlt0.25': 1, 'rlt0.5': 2}
    agg = agg.sort_values(['model', 'arm'], key=lambda s: s.map(order).fillna(9) if s.name == 'arm' else s)

    print(f"{'model':15} {'arm':10} {'steady fps':>11} {'sd':>6} {'speedup':>8} "
          f"{'keep':>7} {'GPU GB':>7} {'n':>3}")
    for model, g in agg.groupby('model'):
        base = g[g['arm'] == 'baseline']['fps']
        base = float(base.iloc[0]) if len(base) else float('nan')
        for _, r in g.iterrows():
            sd = '' if pd.isna(r['fps_sd']) else f"{r['fps_sd']:6.2f}"
            print(f"{r['model']:15} {r['arm']:10} {r['fps']:11.2f} {sd:>6} "
                  f"{r['fps'] / base:7.2f}x {r['keep']:7.3f} {r['gpu']:7.2f} {int(r['n']):3d}")


if __name__ == '__main__':
    main()
