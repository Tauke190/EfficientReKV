"""Token-retention visualization: which spatial patches the pruner keeps vs. drops.

Qualitative companion to the aggregate keep rates. Two clips side by side -- one
near-static, one high-motion -- with every pruned patch dimmed on top of the actual
frame, so a reader can see that the pruner discards the unchanged background and keeps
the region that is moving, instead of taking the percentage on trust.

What is visualized is exactly what the eval does, not a re-implementation of it:

  * frames are sampled on the eval's 1 fps stride grid (video_qa/base.py load_video);
  * features come from the model's own `_get_video_features` after the processor's own
    video preprocessing, i.e. the projected + pooled 14x14 grid the pruner sees;
  * keep decisions come from the registered `rlt_ref` pruner (model/token_pruning.py),
    stateful across frames exactly as in streaming.

The pruner returns kept rows and per-frame counts, not a mask. The mask is recovered from
the carried reference instead: under `rlt_ref` a kept token overwrites ref[p] and a
dropped one leaves it untouched, and a kept token is by definition > threshold > 0 away
from the old ref[p], so "ref[p] changed" is exactly "p was kept". Every frame asserts the
recovered mask agrees with the pruner's own count.

Geometry: LLaVA-OV's video processor resizes straight to 384x384 (no crop), SigLIP makes
27x27 patches, and `apply_pooling` bilinearly resamples them to 14x14 in row-major order.
So token (i, j) covers row band i/14 and column band j/14 of the *full* original frame.
(Pooling is bilinear, so each token also draws a little on its neighbours; the cell is its
centre of mass, not a hard receptive field.)

Clip choice. By default the two clips are picked from StreamingBench (Real) using the
pixel-level optical flow in data/StreamingBench/motion/real_flow_frames.json -- a signal
independent of the pruner's own feature distance, so the choice is not circular:

  static   -- the lowest-mean-flow window that still has *some* motion (a frozen frame
              would just be all-grey, which shows nothing), with no shot cut in it;
  dynamic  -- the highest-mean-flow window with no shot cut in it (a cut makes every
              patch change at once, which is trivially "keep everything", not motion).

The top candidates for each are printed, so a better-looking clip is one flag away:
--static VIDEO_ID:START_SEC / --dynamic VIDEO_ID:START_SEC (video ids are StreamingBench's,
or pass a path to any video file instead of an id).

The pruner starts `--warmup` seconds before the shown window, so the first shown frame is
already in steady state rather than the trivially-all-kept first frame of a stream.

Outputs (few files on purpose -- home has an inode quota):
    <out>.png / <out>.pdf   the figure: one row per clip per threshold, the last column a
                            heatmap of how often each patch was kept over the window
    <out>.gif               the two clips side by side, every frame (--gif)
    <out>_masks.npz         the keep masks and distances, to restyle without the GPU

Usage (needs a GPU, except --preview; see token_retention.slurm):
    python scripts/qualitative/token_retention.py --preview 8      # CPU: pick clips by eye
    python scripts/qualitative/token_retention.py
    python scripts/qualitative/token_retention.py --thresholds 0.5 0.7 --gif
    python scripts/qualitative/token_retention.py \\
        --static Real-Time_Visual_Understanding__sample_228:120 \\
        --dynamic Real-Time_Visual_Understanding__sample_298:30
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

GRID = 14
N_TOK = GRID * GRID


# ---------------------------------------------------------------- clip selection

def _windows(v, span):
    """(start_slot, mean_flow, max_flow) for every cut-free window of `span` slots."""
    flow = np.asarray(v['flow'], dtype=float)
    cut = np.asarray(v['hist1s'], dtype=float) > 0.5
    n = min(len(flow), len(cut))
    out = []
    for s in range(0, n - span + 1):
        if cut[s:s + span].any():
            continue
        f = flow[s:s + span]
        out.append((s, float(f.mean()), float(f.max())))
    return out


def pick_clips(flow_json, span, top=5):
    """Ranked (static, dynamic) candidates as [(video_id, start_slot)], runners-up printed."""
    videos = json.load(open(flow_json))['videos']
    static, dynamic = [], []
    for vid, v in videos.items():
        w = _windows(v, span)
        if not w:
            continue
        # Static: least motion that is still motion -- something in the frame moves
        # (max flow above 1% of frame width/s), so the kept patches have somewhere to be.
        moving = [x for x in w if x[2] > 1.0]
        if moving:
            s, m, _ = min(moving, key=lambda x: x[1])
            static.append((m, vid, s))
        s, m, _ = max(w, key=lambda x: x[1])
        dynamic.append((m, vid, s))
    static.sort()
    dynamic.sort(reverse=True)
    for name, cands in (('static', static), ('dynamic', dynamic)):
        print(f'{name} candidates (mean flow, % frame width/s):')
        for m, vid, s in cands[:top]:
            print(f'    {vid}:{s}   flow={m:.2f}')
    return [(v, s) for _, v, s in static[:top]], [(v, s) for _, v, s in dynamic[:top]]


def preview(cands, anno, warmup, clip_seconds, out):
    """CPU only: a contact sheet of every candidate window (first / middle / last shown
    frame), so the clips can be chosen by eye before paying for the GPU run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(kind, v, s) for kind, cs in cands.items() for v, s in cs]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 1.5 * len(rows)), squeeze=False)
    for r, (kind, v, s) in enumerate(rows):
        start = s + warmup
        frames, times = load_frames(anno[v], start, clip_seconds, 1.0)
        for k, t in enumerate((0, len(frames) // 2, len(frames) - 1)):
            ax = axes[r, k]
            ax.imshow(frames[t]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f't={times[t]:.0f}s', fontsize=7, pad=1)
        axes[r, 0].set_ylabel(f'{kind}\n{v.split("__")[-1]}:{start}', fontsize=7)
    fig.tight_layout()
    fig.savefig(out + '_candidates.png', dpi=110)
    print(f'wrote {out}_candidates.png -- pass the one you like as --static/--dynamic '
          f'VIDEO_ID:START (row label, with the Real-Time_Visual_Understanding__ prefix)')


def resolve_video(spec, anno):
    """'VIDEO_ID:START' or 'path/to/video.mp4:START' -> (label, path, start_sec)."""
    ref, _, start = spec.rpartition(':')
    if not ref:                       # no ':START' given
        ref, start = spec, '0'
    path = anno.get(ref, ref)
    assert os.path.exists(path), f'no such video id or file: {ref}'
    return ref, path, float(start)


# ---------------------------------------------------------------- frames + features

def load_frames(path, start, n_frames, sample_fps):
    """`n_frames` frames from `start` s on the eval's stride grid, (T, H, W, 3) uint8."""
    from video_qa.base import open_video_reader
    vr = open_video_reader(path)
    fps = round(vr.get_avg_fps())
    stride = max(1, int(fps / sample_fps))       # same rule as QA base load_video
    first = int(round(start * sample_fps)) * stride
    idx = [i for i in range(first, len(vr), stride)][:n_frames]
    assert idx, f'{path}: start {start}s is past the end of the video'
    return vr.get_batch(idx).asnumpy(), [i / fps for i in idx]


def token_features(model, frames, batch=16):
    """(T, 196, D) projected + pooled tokens, straight from the model's encode path."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames), batch):
            chunk = torch.from_numpy(frames[i:i + batch])
            px = model.processor.video_processor(chunk, return_tensors='pt') \
                .pixel_values_videos.to(model.device, model.dtype)
            f = model._get_video_features(px)                       # (1, T*196, D)
            out.append(f.view(-1, N_TOK, f.shape[-1]))
    feats = torch.cat(out)
    assert feats.shape[1] == N_TOK, f'expected {N_TOK} tokens/frame, got {feats.shape[1]}'
    return feats


def keep_masks(feats, threshold, metric='cosine'):
    """Run the registered rlt_ref pruner frame by frame; (T, 196) bool masks + distances."""
    import torch
    from model.token_pruning import build_pruner
    pruner = build_pruner('rlt_ref', threshold=threshold, metric=metric)
    masks, dists = [], []
    for t in range(feats.shape[0]):
        before = None if pruner.ref is None else pruner.ref.clone()
        _, counts = pruner(feats[t:t + 1])
        if before is None:
            mask = torch.ones(N_TOK, dtype=torch.bool, device=feats.device)
            dist = torch.full((N_TOK,), float('nan'), device=feats.device)
        else:
            mask = (pruner.ref != before).any(-1)
            f = feats[t].float()
            dist = 1.0 - torch.nn.functional.cosine_similarity(f, before, dim=-1)
        assert int(mask.sum()) == counts[0], \
            f'frame {t}: recovered mask keeps {int(mask.sum())}, pruner kept {counts[0]}'
        masks.append(mask.cpu().numpy())
        dists.append(dist.cpu().numpy())
    return np.stack(masks), np.stack(dists)


# ---------------------------------------------------------------- drawing

def overlay(frame, mask, dim=0.28):
    """Frame with pruned cells darkened and desaturated; kept cells untouched."""
    h, w = frame.shape[:2]
    img = frame.astype(np.float32)
    grey = img.mean(-1, keepdims=True).repeat(3, -1)
    faded = dim * (0.35 * img + 0.65 * grey)
    rows = np.minimum((np.arange(h) * GRID) // h, GRID - 1)
    cols = np.minimum((np.arange(w) * GRID) // w, GRID - 1)
    keep = mask.reshape(GRID, GRID)[rows[:, None], cols[None, :]]
    img = np.where(keep[..., None], img, faded)
    # Hairline at every kept/pruned boundary, so small kept regions stay visible in print.
    edge = np.zeros((h, w), bool)
    edge[1:, :] |= keep[1:, :] != keep[:-1, :]
    edge[:, 1:] |= keep[:, 1:] != keep[:, :-1]
    img[edge] = (255, 214, 0)
    return img.clip(0, 255).astype(np.uint8)


def draw_figure(clips, thresholds, n_show, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = len(clips) * len(thresholds)
    ncol = n_show + 1
    fig, axes = plt.subplots(n_rows, ncol, figsize=(2.3 * ncol, 1.75 * n_rows + 0.3),
                             gridspec_kw=dict(wspace=0.04, hspace=0.32), squeeze=False)
    r = 0
    for c in clips:
        T = len(c['frames'])
        show = np.unique(np.linspace(c['warm'], T - 1, n_show).round().astype(int))
        for thr in thresholds:
            m = c['masks'][thr]
            for k in range(ncol - 1):
                ax = axes[r, k]
                ax.set_xticks([]); ax.set_yticks([])
                if k >= len(show):
                    ax.axis('off'); continue
                t = show[k]
                ax.imshow(overlay(c['frames'][t], m[t]))
                ax.set_title(f't={c["times"][t]:.0f}s  kept {int(m[t].sum())}/{N_TOK}',
                             fontsize=8, pad=2)
            # Keep frequency over the shown window (warm-up excluded).
            ax = axes[r, -1]
            freq = m[c['warm']:].mean(0).reshape(GRID, GRID)
            fh, fw = c['frames'].shape[1:3]   # stretch the 14x14 grid to the frame's aspect
            im = ax.imshow(freq, cmap='magma', vmin=0, vmax=1, interpolation='nearest',
                           extent=[0, fw, fh, 0])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'keep freq. ({100 * m[c["warm"]:].mean():.1f}% kept)',
                         fontsize=8, pad=2)
            axes[r, 0].set_ylabel(f'{c["kind"]}\nthr {thr:g}', fontsize=9)
            r += 1
    cb = fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    for ext in ('.png', '.pdf'):
        fig.savefig(out + ext, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out}.png / {out}.pdf')


def draw_gif(clips, threshold, out, height=270, ms=500):
    """Clips side by side, every frame (warm-up included, marked)."""
    from PIL import Image, ImageDraw
    panels = []
    for c in clips:
        m = c['masks'][threshold]
        seq = []
        for t, f in enumerate(c['frames']):
            im = Image.fromarray(overlay(f, m[t]))
            im = im.resize((int(im.width * height / im.height), height))
            d = ImageDraw.Draw(im)
            tag = ' (warm-up)' if t < c['warm'] else ''
            d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
            d.text((4, 2), f'{c["kind"]}  t={c["times"][t]:.0f}s  kept {int(m[t].sum())}/{N_TOK}{tag}',
                   fill=(255, 255, 255))
            seq.append(im)
        panels.append(seq)
    n = max(len(s) for s in panels)
    frames = []
    for i in range(n):
        ims = [s[min(i, len(s) - 1)] for s in panels]
        canvas = Image.new('RGB', (sum(im.width for im in ims) + 8 * (len(ims) - 1), height), 'white')
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0)); x += im.width + 8
        frames.append(canvas)
    frames[0].save(out + '.gif', save_all=True, append_images=frames[1:], duration=ms, loop=0)
    print(f'wrote {out}.gif')


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', default='llava_ov_7b')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.7])
    p.add_argument('--static', default=None, help='VIDEO_ID:START_SEC or path:START_SEC')
    p.add_argument('--dynamic', default=None, help='VIDEO_ID:START_SEC or path:START_SEC')
    p.add_argument('--anno', default='data/StreamingBench/real.json')
    p.add_argument('--flow', default='data/StreamingBench/motion/real_flow_frames.json')
    p.add_argument('--sample_fps', type=float, default=1.0)
    p.add_argument('--clip_seconds', type=int, default=12, help='length of the shown window')
    p.add_argument('--warmup', type=int, default=4,
                   help='seconds fed to the pruner before the shown window')
    p.add_argument('--n_show', type=int, default=6, help='frames per row in the figure')
    p.add_argument('--gif', action='store_true')
    p.add_argument('--preview', type=int, default=0, metavar='N',
                   help='CPU only: write a contact sheet of the top N auto-picked candidates '
                        'of each kind and exit, without loading the model')
    p.add_argument('--out', default='figures/token_retention')
    args = p.parse_args()

    anno = {x['video_id']: x['video_path'] for x in json.load(open(args.anno))}
    span = int((args.clip_seconds + args.warmup) * args.sample_fps)
    if args.preview:
        assert args.sample_fps == 1.0, 'flow slots are on the 1 fps grid'
        static, dynamic = pick_clips(args.flow, span, top=args.preview)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        preview({'static': static, 'dynamic': dynamic}, anno, args.warmup,
                args.clip_seconds, args.out)
        return
    if args.static is None or args.dynamic is None:
        static, dynamic = pick_clips(args.flow, span)
        # Flow slots are on the 1 fps grid; the window starts after the warm-up.
        args.static = args.static or f'{static[0][0]}:{static[0][1] + args.warmup}'
        args.dynamic = args.dynamic or f'{dynamic[0][0]}:{dynamic[0][1] + args.warmup}'
    print(f'static  = {args.static}\ndynamic = {args.dynamic}')

    from video_qa.base import MODELS
    spec = MODELS[args.model]
    assert args.model.startswith('llava_ov'), 'the 14x14 geometry here is LLaVA-OV specific'
    model, _ = spec['load_func'](model_path=spec['model_path'], n_local=15000)

    clips = []
    for kind, s in (('static', args.static), ('dynamic', args.dynamic)):
        label, path, start = resolve_video(s, anno)
        warm = int(min(args.warmup, start) * args.sample_fps)
        n = warm + int(args.clip_seconds * args.sample_fps)
        frames, times = load_frames(path, start - warm / args.sample_fps, n, args.sample_fps)
        feats = token_features(model, frames)
        masks, dists = {}, {}
        for thr in args.thresholds:
            masks[thr], dists[thr] = keep_masks(feats, thr)
            print(f'{kind:7s} {label}  thr {thr:g}: kept '
                  f'{100 * masks[thr][warm:].mean():.1f}% over the shown window')
        clips.append(dict(kind=kind, label=label, frames=frames, times=times, warm=warm,
                          masks=masks, dists=dists))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    draw_figure(clips, args.thresholds, args.n_show, args.out)
    if args.gif:
        draw_gif(clips, args.thresholds[-1], args.out)
    np.savez_compressed(
        args.out + '_masks.npz',
        **{f'{c["kind"]}_{k}_thr{thr:g}': c[k][thr] for c in clips
           for thr in args.thresholds for k in ('masks', 'dists')},
        **{f'{c["kind"]}_times': np.asarray(c['times']) for c in clips},
        clips=np.asarray([f'{c["kind"]}={c["label"]}' for c in clips]))
    print(f'wrote {args.out}_masks.npz')


if __name__ == '__main__':
    main()
