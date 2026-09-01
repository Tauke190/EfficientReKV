"""Collapse the per-run throughput CSVs into one table.

Which FPS to quote depends on whether the run's KV-Cache ever reached the regime it would
live in, and past a certain threshold it never does -- so this prints three columns rather
than picking one:

  overall   frames/s over the whole stream. The honest headline for a fixed-length stream:
            "this arm ingests a 10-minute clip at N frames/s". Always defined.
  tail      frames/s over the last TAIL_FRAMES frames. Per-frame cost grows with the
            KV-Cache the new frame attends over, so the tail is the slowest, most
            pessimistic rate the run actually sustained, and it is what the overall figure
            would converge to on a longer stream.
  window    frames/s after `local_window_full`, i.e. once the cache passed n_local and
            eviction/offload began. Only defined for arms that get there.

The old version reported `window` alone and dropped every run that never filled n_local.
That silently deletes the aggressive arms: the window fills after 15000 / (196 * keep)
frames, so at a keep rate of 0.5% it would take ~16k frames -- more than a 10-minute clip
at 2 FPS contains, and more than an hour of video at 1 FPS. For those arms "never fills
n_local" is not a warm-up artifact to be measured past, it is the arm's actual steady
behaviour: the cache stays under the local window for the whole stream, which is exactly
why they are fast. `window_full` below reports what fraction of each arm's frames were in
the post-eviction regime, so a reader can see which column applies.
"""
import os
import glob
import argparse

import pandas as pd


# Frames at the end of the run averaged into the `tail` column. 100 frames is enough to
# average out per-frame jitter (individual frames vary ~4x) while still being short enough
# that the cache size barely grows across it.
TAIL_FRAMES = 100


def rate(df):
    return df['num_frames'].sum() / df['seconds'].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/throughput')
    ap.add_argument('--tail', type=int, default=TAIL_FRAMES)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, '*.csv'))):
        df = pd.read_csv(path)
        if not len(df):
            continue
        full = df[df['local_window_full'].astype(bool)] if 'local_window_full' in df else df.iloc[:0]
        name = os.path.splitext(os.path.basename(path))[0]
        model, vid, arm = name.rsplit('-', 2)
        rows.append({
            'model': model, 'video': vid, 'arm': arm,
            'fps': rate(df),
            'tail_fps': rate(df.tail(args.tail)),
            'window_fps': rate(full) if len(full) else float('nan'),
            'window_frac': len(full) / len(df),
            'frames': int(df['num_frames'].sum()),
            'keep_rate': df['keep_rate'].dropna().iloc[-1] if df['keep_rate'].notna().any() else 1.0,
            'kv_gb': df['kv_cache_bytes'].max() / 1e9 if 'kv_cache_bytes' in df else float('nan'),
            'gpu_peak_gb': df['gpu_peak_bytes'].max() / 1e9,
        })
    if not rows:
        print(f'no CSVs in {args.dir}')
        return

    d = pd.DataFrame(rows)
    agg = d.groupby(['model', 'arm']).agg(
        fps=('fps', 'mean'), fps_sd=('fps', 'std'),
        tail=('tail_fps', 'mean'), window=('window_fps', 'mean'),
        wfrac=('window_frac', 'mean'), frames=('frames', 'mean'),
        keep=('keep_rate', 'mean'), gpu=('gpu_peak_gb', 'max'), kv=('kv_gb', 'max'),
        n=('fps', 'size'),
    ).reset_index()

    def arm_order(arm):
        """baseline first, then thresholds ascending -- so the table reads as a sweep."""
        return -1.0 if arm == 'baseline' else float(arm.replace('rlt', ''))

    agg = agg.sort_values(['model', 'arm'],
                          key=lambda s: s.map(arm_order) if s.name == 'arm' else s)

    print(f"{'model':15} {'arm':9} {'overall':>8} {'sd':>6} {'x':>6} {'tail':>7} {'x':>6} "
          f"{'window':>7} {'full%':>6} {'keep':>7} {'KV GB':>6} {'GPU GB':>7} {'n':>3}")
    for model, g in agg.groupby('model'):
        def base_of(col):
            b = g[g['arm'] == 'baseline'][col]
            return float(b.iloc[0]) if len(b) else float('nan')
        b_fps, b_tail = base_of('fps'), base_of('tail')
        for _, r in g.iterrows():
            sd = '' if pd.isna(r['fps_sd']) else f"{r['fps_sd']:6.2f}"
            win = '   n/a' if pd.isna(r['window']) else f"{r['window']:7.2f}"
            print(f"{r['model']:15} {r['arm']:9} {r['fps']:8.2f} {sd:>6} {r['fps'] / b_fps:5.2f}x "
                  f"{r['tail']:7.2f} {r['tail'] / b_tail:5.2f}x {win:>7} {r['wfrac'] * 100:5.0f}% "
                  f"{r['keep']:7.4f} {r['kv']:6.2f} {r['gpu']:7.2f} {int(r['n']):3d}")
    print(f"\n{int(agg['frames'].mean())} frames/run; tail = last {args.tail} frames. "
          "'full%' is the share of frames encoded after the cache passed n_local;\n"
          "where it is 0 the arm never reached that regime within the stream and 'window' "
          "is undefined -- read 'overall'/'tail'.")


if __name__ == '__main__':
    main()
