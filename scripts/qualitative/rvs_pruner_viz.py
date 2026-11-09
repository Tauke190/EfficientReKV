"""Pruner visualization for one RVS-Ego question: what rlt_ref keeps over its window.

Defaults to video 879dd163, question #59 (45:00-50:00):
Q: "Is there any noticeable movement or gesture in the video?"
GT: "The person raised their hands at some point."
Unpruned: correct (hand reaches toward the camera). rlt_ref @ 0.9: "No ... no movement".
Any other question: --video_id ID --q N (window and question read from the annotation).

Replays the eval exactly: 1 fps stride grid from 00:00, LLaVA-OV-7B's own vision path
(processor -> SigLIP -> projector -> pooling, fp16), and the registered rlt_ref pruner fed
one frame per call from slot 0, so the carried reference at 45:00 is the one the eval had.
Only the vision tower and projector are loaded -- the LM plays no part in the keep decision.

The replay is checked against the eval's own counts (cumulative tokens_kept in
results.csv at every question up to #59).

Outputs (few files -- inode quota), <out> = figures/rvs_<video[:8]>_q<N>_pruner:
    <out>.mp4       original | kept tokens | distance-to-reference, over the window
    <out>_keep.png  tokens kept and max distance per frame over the window
    <out>.npz       masks + distances for the window, per-frame counts for all
"""

import os
import sys
import json
import time
import subprocess

import numpy as np
import torch

ROOT = '/home/av354855/EfficientReKV'
sys.path.insert(0, ROOT)

MODEL = f'{ROOT}/model_zoo/llava-onevision-qwen2-7b-ov-hf'
GRID, N_TOK = 14, 196
BLOCK = 32
PANEL_H = 360


def load_vision(device):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
    cfg = AutoConfig.from_pretrained(MODEL)
    with init_empty_weights():
        m = LlavaOnevisionForConditionalGeneration(cfg)
    idx = json.load(open(f'{MODEL}/model.safetensors.index.json'))['weight_map']
    shards = {v for k, v in idx.items() if 'language_model' not in k}
    sd = {}
    for s in shards:
        sd.update({k: v for k, v in load_file(f'{MODEL}/{s}').items() if 'language_model' not in k})
    m.load_state_dict(sd, strict=False, assign=True)
    m.vision_tower.to(device, torch.float16).eval()
    m.multi_modal_projector.to(device, torch.float16).eval()
    return m, cfg, LlavaOnevisionProcessor.from_pretrained(MODEL)


@torch.inference_mode()
def features(m, cfg, proc, frames, device):
    """Same as LlavaOnevisionReKV._get_video_features without a vision reducer."""
    px = proc.video_processor(torch.from_numpy(frames), return_tensors='pt') \
        .pixel_values_videos.to(device, torch.float16)
    b, f, c, h, w = px.shape
    hs = m.vision_tower(px.view(b * f, c, h, w), output_hidden_states=True) \
        .hidden_states[cfg.vision_feature_layer]
    if cfg.vision_feature_select_strategy == 'default':
        hs = hs[:, 1:]
    feats = m.apply_pooling(m.multi_modal_projector(hs))
    return feats.view(-1, N_TOK, feats.shape[-1])


def overlay(frame, mask, dim=0.22):
    h, w = frame.shape[:2]
    img = frame.astype(np.float32)
    grey = img.mean(-1, keepdims=True).repeat(3, -1)
    faded = dim * (0.35 * img + 0.65 * grey)
    rows = np.minimum((np.arange(h) * GRID) // h, GRID - 1)
    cols = np.minimum((np.arange(w) * GRID) // w, GRID - 1)
    keep = mask.reshape(GRID, GRID)[rows[:, None], cols[None, :]]
    img = np.where(keep[..., None], img, faded)
    edge = np.zeros((h, w), bool)
    edge[1:, :] |= keep[1:, :] != keep[:-1, :]
    edge[:, 1:] |= keep[:, 1:] != keep[:, :-1]
    img[edge] = (255, 214, 0)
    return img.clip(0, 255).astype(np.uint8)


def heat(frame, dist, thr):
    """Distance to the carried reference per token, 0 -> thr mapped onto magma; >thr = kept."""
    import matplotlib
    cmap = matplotlib.colormaps['magma']
    h, w = frame.shape[:2]
    d = np.nan_to_num(dist, nan=1.0).reshape(GRID, GRID) / thr
    rgb = (cmap(np.clip(d, 0, 1))[..., :3] * 255).astype(np.float32)
    rows = np.minimum((np.arange(h) * GRID) // h, GRID - 1)
    cols = np.minimum((np.arange(w) * GRID) // w, GRID - 1)
    rgb = rgb[rows[:, None], cols[None, :]]
    base = frame.astype(np.float32).mean(-1, keepdims=True)
    return (0.75 * rgb + 0.25 * base).clip(0, 255).astype(np.uint8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_id', default='879dd163-7588-45d1-9466-a5deabc59167')
    ap.add_argument('--q', type=int, default=59)
    ap.add_argument('--threshold', type=float, default=0.9)
    args = ap.parse_args()
    global VIDEO_ID, THR
    VIDEO_ID, THR = args.video_id, args.threshold
    VIDEO = f'{ROOT}/data/rvs/ego/videos/{VIDEO_ID}.mp4'
    anno = [x for x in json.load(open(f'{ROOT}/data/rvs/ego/ego4d_oe.json')) if x['video_id'] == VIDEO_ID][0]
    q = anno['conversations'][args.q]
    # shown window in 1 fps slots; the question fires after slot W1 - 1, as in the eval
    W0, W1 = int(float(q['start_time'])), int(float(q['end_time']))
    OUT = f'{ROOT}/figures/rvs_{VIDEO_ID[:8]}_q{args.q}_pruner'
    print(f'#{args.q}: {q["question"]!r}  GT {q["answer"]!r}  window {W0}-{W1}s', flush=True)
    from PIL import Image, ImageDraw, ImageFont
    from video_qa.base import open_video_reader
    from model.token_pruning import build_pruner

    device = 'cuda'
    t0 = time.time()
    m, cfg, proc = load_vision(device)
    print(f'vision path loaded in {time.time() - t0:.0f}s', flush=True)

    vr = open_video_reader(VIDEO)
    fps = round(vr.get_avg_fps())
    stride = max(1, int(fps / 1.0))                     # eval's stride rule at 1 fps
    pruner = build_pruner('rlt_ref', threshold=THR, metric='cosine')

    counts = np.zeros(W1, np.int32)
    w_masks = np.zeros((W1 - W0, N_TOK), bool)
    w_dists = np.zeros((W1 - W0, N_TOK), np.float32)
    w_frames = []
    t0 = time.time()
    for s in range(0, W1, BLOCK):
        slots = list(range(s, min(s + BLOCK, W1)))
        frames = vr.get_batch([k * stride for k in slots]).asnumpy()
        feats = features(m, cfg, proc, frames, device)
        for j, k in enumerate(slots):
            before = None if pruner.ref is None else pruner.ref.clone()
            _, c = pruner(feats[j:j + 1])
            counts[k] = c[0]
            if W0 <= k < W1:
                i = k - W0
                if before is None:
                    w_masks[i] = True; w_dists[i] = np.nan
                else:
                    w_masks[i] = (pruner.ref != before).any(-1).cpu().numpy()
                    w_dists[i] = (1.0 - torch.nn.functional.cosine_similarity(
                        feats[j].float(), before, dim=-1)).cpu().numpy()
                assert w_masks[i].sum() == c[0]
                fr = frames[j]
                ph = PANEL_H; pw = int(fr.shape[1] * ph / fr.shape[0])
                w_frames.append(np.asarray(Image.fromarray(fr).resize((pw, ph), Image.BILINEAR)))
        if (s // BLOCK) % 10 == 0:
            el = time.time() - t0
            print(f'slot {slots[-1] + 1}/{W1}  kept so far {counts[:slots[-1] + 1].sum()}  '
                  f'{el:.0f}s elapsed, ~{el / (slots[-1] + 1) * (W1 - slots[-1] - 1):.0f}s left', flush=True)

    # ---- check against the eval's own counts
    import pandas as pd
    df = pd.read_csv(f'{ROOT}/results/llava_ov_7b/rvs_ego/64-1.0-rlt_ref{THR:g}cosine/results.csv')
    df = df[df.video_id == VIDEO_ID].reset_index(drop=True)
    cum = np.cumsum(counts)
    print(f'check vs eval up to #{args.q} (question, end slot, eval tokens_kept, replay):')
    worst = 0
    for qi, q in enumerate(anno['conversations']):
        n = int(float(q['end_time']))
        if n > W1:
            break
        ev, rp = int(df.tokens_kept[qi]), int(cum[n - 1])
        worst = max(worst, abs(ev - rp))
        if qi % 10 == 0 or qi == args.q:
            print(f'   #{qi:3d}  slot {n:4d}  eval {ev:5d}  replay {rp:5d}')
    print(f'max |eval - replay| over questions up to slot {W1}: {worst} tokens')

    # ---- video: original | kept tokens | distance to reference
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    ph = PANEL_H; pw = w_frames[0].shape[1]; gap = 6; bar = 26
    W = 3 * pw + 2 * gap; H = ph + bar
    W += W % 2; H += H % 2
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', '4', '-i', '-', '-c:v', 'mpeg4', '-q:v', '3',
                           OUT + '.mp4'], stdin=subprocess.PIPE)
    for i, fr in enumerate(w_frames):
        k = W0 + i
        canvas = Image.new('RGB', (W, H), (0, 0, 0))
        canvas.paste(Image.fromarray(fr), (0, bar))
        canvas.paste(Image.fromarray(overlay(fr, w_masks[i])), (pw + gap, bar))
        canvas.paste(Image.fromarray(heat(fr, w_dists[i], THR)), (2 * (pw + gap), bar))
        d = ImageDraw.Draw(canvas)
        t = f'{k // 60:02d}:{k % 60:02d}'
        d.text((6, 5), f'{t}  original', fill='white', font=font)
        d.text((pw + gap + 6, 5), f'rlt_ref@{THR}: kept {int(w_masks[i].sum())}/{N_TOK}', fill='white', font=font)
        d.text((2 * (pw + gap) + 6, 5), f'dist to ref (max {np.nanmax(w_dists[i]):.2f}, thr {THR})',
               fill='white', font=font)
        ff.stdin.write(np.asarray(canvas).tobytes())
    ff.stdin.close(); ff.wait()
    print(f'wrote {OUT}.mp4  ({len(w_frames)} frames at 4 fps = {len(w_frames) / 4:.0f}s)')

    # ---- per-frame plot over the window
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(W0, W1) / 60
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    ax[0].bar(x, w_masks.sum(1), width=1 / 60, color='#3b6ea5')
    ax[0].set_ylabel('tokens kept\n(of 196)')
    ax[0].set_title(f'RVS-Ego {VIDEO_ID[:8]}, Q{args.q} window {W0 // 60:02d}:{W0 % 60:02d}-{W1 // 60:02d}:{W1 % 60:02d}, rlt_ref@{THR}: '
                    f'{w_masks.sum()} of {w_masks.size} tokens kept ({100 * w_masks.mean():.2f}%)', fontsize=10)
    ax[1].plot(x, np.nanmax(w_dists, 1), lw=0.8, label='max over tokens')
    ax[1].plot(x, np.nanmean(w_dists, 1), lw=0.8, label='mean over tokens')
    ax[1].axhline(THR, color='crimson', ls='--', lw=0.8, label=f'threshold {THR}')
    ax[1].set_ylabel('cosine dist.\nto reference'); ax[1].set_xlabel('minute')
    ax[1].legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(OUT + '_keep.png', dpi=150)
    print(f'wrote {OUT}_keep.png')

    np.savez_compressed(OUT + '.npz', masks=w_masks, dists=w_dists, counts_all=counts,
                        window=np.array([W0, W1]), threshold=THR)
    print(f'wrote {OUT}.npz')


if __name__ == '__main__':
    main()
