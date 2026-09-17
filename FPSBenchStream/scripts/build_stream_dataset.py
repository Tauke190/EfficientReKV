#!/usr/bin/env python3
"""Build FPS-Bench-Stream: each FPS-Bench clip spliced into a long MLVU haystack.

FPS-Bench's own source videos are far too short to make a long-video benchmark
(median 3m58s, longest 10m23s), so length has to come from somewhere else. This
script takes the *needle-in-a-haystack* route: one FPS-Bench clip is inserted at
a controlled offset inside a long MLVU video, giving exactly two scene cuts and
leaving the question's answer untouched -- the padding footage does not contain
the action being asked about.

Two modes:

``plan``
    Deterministically assign every FPS-Bench question a haystack, an insertion
    offset, and a position bin, then write the annotations needed to rebuild the
    streams. No video is touched. This is the redistributable artifact: it names
    videos and timestamps, it does not contain them.

``assemble``
    Cut the streams with ffmpeg from a plan. Every part is normalised to one
    resolution and frame rate and audio is stripped, so the splice leaves no
    resolution/fps/audio discontinuity for a model to detect.

Note on reproducing a *published* release: run ``assemble`` against the shipped
``fpsbench_stream_v1.jsonl`` and do not re-run ``plan``. ``plan`` draws its
haystack assignments from an RNG whose sequence depends on which needle clips
are present in the clip cache, so a recipient with a different set of downloaded
clips gets a different -- equally valid, but not identical -- assignment.

Usage:
    # rebuild a published release (the normal path)
    python scripts/build_stream_dataset.py assemble \
        --plan fpsbench_stream_v1.jsonl --video-dir videos \
        --canvas haystack --clip-dir ~/.cache/fpsbench/clips/clip \
        --haystack-dir /path/to/mlvu

    # plan a new release from scratch
    python scripts/build_stream_dataset.py plan \
        --target-duration 600 --output-dir data/FPSBenchStream
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def resolve_ffmpeg() -> str:
    """First ffmpeg on the box that can actually encode H.264.

    The cluster default (anaconda3-2019.03) is built without libx264, so it
    fails on ``-preset`` with a misleading "option not found". Probe instead of
    trusting PATH.
    """
    import shutil
    cands = ["/usr/bin/ffmpeg", shutil.which("ffmpeg"), "/usr/local/bin/ffmpeg"]
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in cands:
        if not c or not Path(c).exists():
            continue
        enc = subprocess.run([c, "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        if "libx264" in enc:
            return c
    sys.exit("ERROR: no ffmpeg with libx264 found")


FFMPEG = None  # resolved lazily in assemble()
# Defaults only. Every one is overridable, because this script also ships inside
# the FPSBenchStream release, where there is no FPSBench checkout around it.
CLIP_CACHE = Path.home() / ".cache/fpsbench/clips/clip"
HAYSTACK_DIR = ROOT / "data/haystack/mlvu_ge10min"
ANNOTATIONS = ROOT / "annotations/fpsbench_v1.jsonl"

# MLVU's own `needle_*` subset already has clips spliced into it -- reusing those
# as padding would stack a second needle behind ours. Always dropped.
EXCLUDE_SUBSETS = ("needle",)
# MLVU's `count_*` videos are built around counting events, which collides with
# FPS-Bench's own counting questions ("how many times does X happen"). Rather
# than drop them, we just never pair them with a counting question.
COUNTING_TASKS = ("instance_count", "repetitive_motion")
# Keep the needle clear of the very start/end, where a cut is trivially spotted.
EDGE_GUARD_SEC = 30.0
POSITION_BINS = {"early": (0.10, 0.30), "middle": (0.40, 0.60), "late": (0.70, 0.90)}


def probe(path: Path, entries: str, stream: Optional[str] = None) -> str:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def load_haystacks(manifest: Optional[Path] = None) -> List[Dict]:
    """Read the >=10-minute MLVU manifest, dropping contaminated subsets."""
    man = Path(manifest) if manifest else HAYSTACK_DIR / "manifest.tsv"
    if not man.exists():
        sys.exit(f"ERROR: no haystack manifest at {man}")
    out = []
    for i, line in enumerate(man.read_text().splitlines()):
        if i == 0:
            continue
        f, dur, _hms, w, h, fps = line.split("\t")
        subset = f.split("_")[0] if "_" in f else "misc"
        if subset in EXCLUDE_SUBSETS:
            continue
        out.append({"file": f, "duration_sec": float(dur), "subset": subset,
                    "width": int(w), "height": int(h), "fps": float(fps)})
    return out


def load_clips(clip_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Map fpsbench id -> downloaded clip path and true measured duration."""
    clips = {}
    for p in sorted(Path(clip_dir or CLIP_CACHE).glob("fpsbench_*.mp4")):
        parts = p.stem.split("_")
        fid = f"{parts[0]}_{parts[1]}"
        clips[fid] = {"path": p, "duration_sec": None}
    return clips


def plan(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.annotations)]
    clips = load_clips(args.clip_dir)
    haystacks = load_haystacks(args.manifest)
    if not haystacks:
        sys.exit("ERROR: haystack pool is empty")

    # Measured clip durations (the annotation value is authoritative, but a clip
    # that downloaded short would silently shift every downstream timestamp).
    dur_cache = {}
    for fid, c in clips.items():
        d = probe(c["path"], "format=duration")
        dur_cache[fid] = float(d) if d else None

    target = float(args.target_duration)
    rng = random.Random(args.seed)
    bins = list(POSITION_BINS)

    plan_rows, skipped = [], []
    # Longest haystacks first so short ones are not wasted on long needles.
    pool = sorted(haystacks, key=lambda h: -h["duration_sec"])
    usage = {h["file"]: 0 for h in pool}

    for idx, rec in enumerate(records):
        fid = rec["id"]
        if fid not in clips:
            skipped.append({"id": fid, "reason": "clip not downloaded"})
            continue
        needle_dur = dur_cache.get(fid) or rec["time"]["clip_duration_sec"]
        if not needle_dur or needle_dur <= 0:
            skipped.append({"id": fid, "reason": "clip duration unreadable"})
            continue

        hay_needed = target - needle_dur
        task = rec["question"]["type"]
        cands = [h for h in pool if h["duration_sec"] >= hay_needed
                 and not (h["subset"] == "count" and task in COUNTING_TASKS)]
        if not cands:
            skipped.append({"id": fid, "reason": f"no haystack >= {hay_needed:.0f}s"})
            continue
        # Spread load: prefer the least-reused haystack, break ties randomly.
        least = min(usage[h["file"]] for h in cands)
        cands = [h for h in cands if usage[h["file"]] == least]
        hay = rng.choice(cands)
        usage[hay["file"]] += 1

        # Window of haystack actually used, and where inside it the needle lands.
        win_start = rng.uniform(0.0, max(0.0, hay["duration_sec"] - hay_needed))
        pos_bin = bins[idx % len(bins)]          # stratified, not random
        lo, hi = POSITION_BINS[pos_bin]
        lo_s = max(EDGE_GUARD_SEC, lo * hay_needed)
        hi_s = min(hay_needed - EDGE_GUARD_SEC, hi * hay_needed)
        if hi_s <= lo_s:
            lo_s, hi_s = hay_needed * 0.4, hay_needed * 0.6
        offset = rng.uniform(lo_s, hi_s)

        t = rec["time"]
        cert_off = t["temporal_certificate_start_sec"] - t["clip_start_sec"]
        cert_dur = t["temporal_certificate_duration_sec"]
        needle_start = offset
        needle_end = offset + needle_dur
        cert_start_stream = needle_start + max(0.0, cert_off)
        cert_end_stream = min(needle_end, cert_start_stream + cert_dur)

        plan_rows.append({
            "stream_id": f"fpsstream_{len(plan_rows):06d}",
            "source_question_id": fid,
            "question": rec["question"],
            "temporal_requirements": rec["temporal_requirements"],
            "categories": rec["categories"],
            "needle": {
                "fpsbench_id": fid,
                "source_video_id": rec["source"]["video_id"],
                "source_url": rec["source"]["url"],
                "clip_start_sec": t["clip_start_sec"],
                "clip_end_sec": t["clip_end_sec"],
                "clip_file": clips[fid]["path"].name,
                "measured_duration_sec": round(needle_dur, 3),
            },
            "haystack": {
                "dataset": "MLVU",
                "file": hay["file"],
                "subset": hay["subset"],
                "full_duration_sec": hay["duration_sec"],
                "window_start_sec": round(win_start, 3),
                "window_duration_sec": round(hay_needed, 3),
            },
            "insertion": {
                "offset_in_window_sec": round(offset, 3),
                "position_bin": pos_bin,
                "position_frac": round(offset / hay_needed, 4),
            },
            "stream": {
                "total_duration_sec": round(target, 3),
                "needle_start_sec": round(needle_start, 3),
                "needle_end_sec": round(needle_end, 3),
                "certificate_start_sec": round(cert_start_stream, 3),
                "certificate_end_sec": round(cert_end_stream, 3),
                "query_time_sec": round(cert_end_stream, 3),
            },
            "normalization": {
                "width": args.width, "height": args.height,
                "fps": args.fps, "audio": "stripped",
            },
        })

    out = Path(out_dir) / "fpsbench_stream_v1.jsonl"
    with open(out, "w") as fh:
        for r in plan_rows:
            fh.write(json.dumps(r) + "\n")
    if skipped:
        with open(Path(out_dir) / "skipped.jsonl", "w") as fh:
            for s in skipped:
                fh.write(json.dumps(s) + "\n")

    import collections
    stats = {
        "num_streams": len(plan_rows),
        "num_skipped": len(skipped),
        "target_duration_sec": target,
        "normalization": {"width": args.width, "height": args.height,
                          "fps": args.fps, "audio": "stripped"},
        "seed": args.seed,
        "haystack_pool_size": len(haystacks),
        "haystack_reuse": {
            "max": max(usage.values()) if usage else 0,
            "mean": round(sum(usage.values()) / max(1, sum(1 for v in usage.values() if v)), 2),
        },
        "position_bins": dict(collections.Counter(r["insertion"]["position_bin"] for r in plan_rows)),
        "haystack_subsets": dict(collections.Counter(r["haystack"]["subset"] for r in plan_rows)),
        "skipped_reasons": dict(collections.Counter(s["reason"] for s in skipped)),
    }
    (Path(out_dir) / "fpsbench_stream_v1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")


def assemble(args) -> None:
    global FFMPEG
    FFMPEG = args.ffmpeg or resolve_ffmpeg()
    print(f"using ffmpeg: {FFMPEG}  encoder: {args.encoder}")
    rows = [json.loads(l) for l in open(args.plan)]
    if args.limit:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        # Deterministic interleave: shard k takes every num_shards-th row. Keeps
        # each worker's mix of haystack lengths comparable so they finish together.
        rows = rows[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(rows)} streams")
    vid_dir = Path(args.video_dir)
    vid_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    built = []
    for r in rows:
        # Merged layout keeps every original FPS-Bench field at the top level and
        # nests the splice under "stream"; the older flat layout put it at the
        # top level. Accept both so an in-flight run is never invalidated.
        sp = r.get("stream", r)
        sid = sp.get("stream_id") or r.get("stream_id")
        tl = sp.get("timeline", sp)
        out = vid_dir / f"{sid}.mp4"
        if out.exists() and not args.overwrite:
            ok += 1
            continue
        n = sp["normalization"]
        hay = Path(args.haystack_dir) / sp["haystack"]["file"]
        needle = Path(args.clip_dir) / sp["needle"]["clip_file"]
        if args.canvas == "haystack":
            # Canvas follows the haystack. Costs less than a fixed 720p/1080p
            # canvas and the needle still clears the 384x384 the vision tower
            # resizes every frame to, so nothing the model can see is lost.
            hw = probe(hay, "stream=width,height,r_frame_rate", "v:0")
            if not hw:
                print(f"SKIP {sid}: cannot probe haystack", file=sys.stderr)
                fail += 1
                continue
            parts = hw.split(",")
            W, H = int(parts[0]), int(parts[1])
            num, den = parts[2].split("/")
            hay_fps = float(num) / float(den)
            # A needle whose min_fps exceeds the haystack's rate would lose the
            # very motion the question asks about, so raise the canvas instead.
            need_fps = r.get("temporal_requirements", {}).get("min_fps", 0) or 0
            F = int(round(max(hay_fps, need_fps)))
        else:
            W, H, F = n["width"], n["height"], n["fps"]
        if not hay.exists() or not needle.exists():
            print(f"SKIP {sid}: missing input", file=sys.stderr)
            fail += 1
            continue

        ws = sp["haystack"]["window_start_sec"]
        off = sp["insertion"]["offset_in_window_sec"]
        post = sp["haystack"]["window_duration_sec"] - off
        # Normalise every part identically: scale into the canvas preserving
        # aspect, pad, reset SAR, force one frame rate. Audio dropped entirely --
        # the FPS-Bench clips were fetched with --no-audio, so keeping the
        # haystack's track would make the splice audible.
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={F}")
        if args.encoder == "h264_nvenc":
            enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-cq", str(args.crf), "-b:v", "0"]
        else:
            enc = ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf)]
        cmd = [
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{ws:.3f}", "-t", f"{off:.3f}", "-i", str(hay),
            "-i", str(needle),
            "-ss", f"{ws + off:.3f}", "-t", f"{post:.3f}", "-i", str(hay),
            "-filter_complex",
            f"[0:v]{vf}[a];[1:v]{vf}[b];[2:v]{vf}[c];[a][b][c]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-an", *enc,
            "-pix_fmt", "yuv420p", str(out),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and out.exists():
            ok += 1
            built.append({"stream_id": sid, "width": W, "height": H, "fps": F,
                          "audio": "stripped", "bytes": out.stat().st_size})
            print(f"built {out.name}  {W}x{H}@{F}  ({out.stat().st_size/1e6:.1f} MB)")
        else:
            fail += 1
            print(f"FAIL {sid}: {p.stderr.strip()[:200]}", file=sys.stderr)
    if built:
        # Record the canvas actually used so the annotations can be updated to
        # match; with --canvas haystack it varies per stream.
        man = vid_dir / "built_canvas.jsonl"
        with open(man, "a") as fh:
            for b in built:
                fh.write(json.dumps(b) + "\n")
        print(f"canvas recorded in {man}")
    print(f"\nassembled {ok}, failed {fail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write annotations, touch no video")
    p.add_argument("--target-duration", type=float, default=600.0)
    p.add_argument("--output-dir", default=str(ROOT / "data/FPSBenchStream"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--annotations", default=str(ANNOTATIONS),
                   help="FPS-Bench v1 annotations to plan over")
    p.add_argument("--clip-dir", default=str(CLIP_CACHE),
                   help="directory of downloaded needle clips")
    p.add_argument("--haystack-dir", default=str(HAYSTACK_DIR),
                   help="directory of MLVU haystack videos")
    p.add_argument("--manifest", default=None,
                   help="haystack manifest.tsv (default: <haystack-dir>/manifest.tsv)")
    p.set_defaults(func=plan)

    a = sub.add_parser("assemble", help="cut streams with ffmpeg from a plan")
    a.add_argument("--plan", required=True)
    a.add_argument("--video-dir", default=str(ROOT / "data/FPSBenchStream/videos"))
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--crf", type=int, default=23)
    a.add_argument("--preset", default="veryfast", help="libx264 preset")
    a.add_argument("--encoder", default="libx264", choices=("libx264", "h264_nvenc"))
    a.add_argument("--ffmpeg", default=None, help="override ffmpeg binary")
    a.add_argument("--clip-dir", default=str(CLIP_CACHE),
                   help="directory of rebuilt needle clips")
    a.add_argument("--haystack-dir", default=str(HAYSTACK_DIR),
                   help="directory of MLVU haystack videos")
    a.add_argument("--shard", type=int, default=0, help="this worker's index")
    a.add_argument("--num-shards", type=int, default=1, help="total parallel workers")
    a.add_argument("--canvas", default="fixed", choices=("fixed", "haystack"),
                   help="'fixed' uses stream.normalization; 'haystack' matches the "
                        "haystack's own resolution and frame rate")
    a.set_defaults(func=assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
