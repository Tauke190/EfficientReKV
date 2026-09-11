"""Unpack OVBench into one video per clip, transcoding its frame directories on the fly.

OVBench (VideoChat-Online, MCG-NJU) ships ten per-source zips and addresses every clip
from `ovbench.json` as `<source>/<rest>`. Only three of the ten hold containers; the other
seven hold *directories of JPEGs*, which is upstream's own format -- there is no mp4
distribution of this benchmark, and its README says so ("the video data used in this
benchmark consists of both image sequences and video clips").

Extracting those as shipped costs **1,045,648 files**, or 345,360 for just the clips the
annotation references. That is an inode budget, not a disk budget, and it is what this
script exists to avoid: each needed frame directory is piped straight out of the zip into
ffmpeg and lands as a single .mp4. No JPEG is ever written to disk.

    frame sources  326 directories, 345,360 JPEGs, 40.2 GB  ->  326 .mp4, ~6 GB
    container srcs                   1,137 files,  73.5 GB  ->  1,137 files, unchanged
    ------------------------------------------------------------------------------
    total                          346,497 files            ->  1,463 files

Three things make the transcode safe to do unattended, all verified against the zips:

* **Frame order.** Within any one directory the basenames share a prefix and a width, so
  lexicographic order is numeric order -- checked here rather than assumed, because the
  widths differ *across* ArgoVerse directories (37/38/40 chars: `ring_side_left_` vs
  `ring_rear_right_`) and a naive global sort would look fine while silently shuffling.
* **Frame rate.** A directory carries no rate; `ovbench.json`'s per-video `fps` is the
  only record of it, and it varies from 8 to 60 even within one source. It is what maps a
  question's timestamp to a frame index, so it is passed to ffmpeg per clip. The six
  index-named sources are contiguous dumps (frame0001..frameN, no gaps) and ArgoVerse's
  nanosecond timestamps are uniform to within 0.5%, so a constant rate is right for both.
* **Naming.** `ovbench.json` says `HACS/Spinning_v_...` with no extension, so the output
  gets `.mp4` appended; video_qa/convert_ovbench.py resolves both spellings.

CRF 18 is visually lossless but not bit-exact (PSNR ~40 dB against the source JPEGs).
`--crf 0` gives a mathematically lossless H.264 at roughly 3x the JPEGs' size if a run
ever needs to be defended frame-for-frame.

Re-running is cheap and safe: a clip is skipped when its .mp4 is already there, and
encodes go to a .tmp that is renamed only on a clean exit, so an interrupted run never
leaves a half-written file that the next one would trust. Zips are never deleted.

Usage:
    python scripts/dataset_prep/setup_ovbench.py                    # everything, 4 parallel encodes
    python scripts/dataset_prep/setup_ovbench.py --jobs 8
    python scripts/dataset_prep/setup_ovbench.py --sources HACS     # one source
    python scripts/dataset_prep/setup_ovbench.py --verify_only      # check an earlier run
"""

import os
import re
import sys
import json
import time
import shutil
import zipfile
import argparse
import subprocess
import collections
import multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Zips holding real containers: source name in ovbench.json -> top-level folder to strip.
# The names disagree in two of three cases (COIN.zip holds `coin/`, HiREST.zip holds
# `HiREST/` but the annotation says `hirest/`), and all three are fully referenced by the
# annotation -- 601, 472 and 64 videos, nothing spare.
CONTAINER_SOURCES = {
    'COIN':    ('COIN',    'coin'),
    'HiREST':  ('hirest',  'HiREST'),
    'AVA_RAW': ('AVA_RAW', 'AVA_RAW'),
}

# Zips holding one directory of JPEGs per clip, dropped at the archive root. Every one is
# a superset of what the benchmark asks for (LaSOT ships 201 sequences for 71 questions),
# so only the referenced directories are transcoded.
FRAME_SOURCES = ['LaSOT', 'HACS', 'AVA', 'Charades', 'YFCC100M', 'BDD', 'ArgoVerse']

ALL_SOURCES = list(CONTAINER_SOURCES) + FRAME_SOURCES


def wanted_ids(anno_path):
    """source -> {rest of video_id: annotation entry}, for every clip the benchmark uses."""
    by_source = collections.defaultdict(dict)
    for entry in json.load(open(anno_path)):
        source, rest = entry['video_id'].split('/', 1)
        by_source[source][rest] = entry
    return by_source


def frame_order(names):
    """`names` in playback order, asserting that sorting them is unambiguous.

    Returns the lexicographic order, but only after confirming it agrees with ordering by
    the trailing integer -- the frame index for six sources, a nanosecond capture time for
    ArgoVerse. The two agree exactly when every basename is the same width, which is true
    inside a directory and false across ArgoVerse as a whole.
    """
    lex = sorted(names)
    num = sorted(names, key=lambda n: int(re.findall(r'\d+', n)[-1]))
    assert lex == num, (
        'frame names do not sort unambiguously -- lexicographic and numeric order '
        f'disagree (e.g. {lex[:2]} vs {num[:2]}). Refusing to guess playback order.')
    return lex


def encode_clip(job):
    """Transcode one frame directory straight from the zip into a single .mp4.

    Runs in a worker process, so it opens its own ZipFile handle. Frames are read one at a
    time and written to ffmpeg's stdin, which is the whole point: peak disk cost is the
    output file and peak memory is one JPEG.
    """
    zip_path, source, clip, names, fps, out_path, crf, preset, threads = job
    tmp = out_path + '.tmp.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'image2pipe', '-framerate', str(fps), '-i', '-',
           '-c:v', 'libx264', '-crf', str(crf), '-preset', preset,
           '-threads', str(threads), '-pix_fmt', 'yuv420p',
           # libx264's yuv420p needs even dimensions; a no-op wherever they already are.
           '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
           '-an', tmp]
    t0 = time.time()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                for name in names:
                    proc.stdin.write(zf.read(name))
                proc.stdin.close()
            except BrokenPipeError:
                pass  # ffmpeg died; its stderr below is the real error
            err = proc.stderr.read().decode(errors='replace')[-500:]
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f'ffmpeg exited {rc}: {err.strip()}')
        # Rename only now, so a killed run leaves a .tmp the next run ignores rather than
        # a short .mp4 it would skip as done.
        os.replace(tmp, out_path)
        return (source, clip, len(names), os.path.getsize(out_path), time.time() - t0, None)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        return (source, clip, len(names), 0, time.time() - t0, str(exc))


def n_frames(path):
    """Frames in an encoded file, by counting packets. None if ffprobe cannot read it."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
             '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=300)
        return int(out.stdout.strip().split(',')[0])
    except Exception:
        return None


def do_container(zip_path, source, strip, dest, verify_only):
    """Extract a container zip as shipped -- one file per video, no transcode needed."""
    os.makedirs(dest, exist_ok=True)
    written = skipped = 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.startswith(strip + '/')]
        for m in members:
            rel = m.filename[len(strip) + 1:]
            if not rel:
                continue
            target = os.path.join(dest, rel)
            if os.path.exists(target) and os.path.getsize(target) == m.file_size:
                skipped += 1
                continue
            if verify_only:
                continue
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            with zf.open(m) as src, open(target + '.tmp', 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(target + '.tmp', target)
            written += 1
    print(f'  {len(members)} videos ({skipped} already present, {written} written) '
          f'in {time.time() - t0:.0f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.path.join(REPO_ROOT, 'data/OVBench'),
                   help='Directory holding the downloaded *.zip files and ovbench.json.')
    p.add_argument('--dest', default=None, help='Default: <src>/videos')
    p.add_argument('--anno', default=None, help='Default: <src>/ovbench.json')
    p.add_argument('--sources', nargs='+', default=ALL_SOURCES, choices=ALL_SOURCES,
                   help='Sources to unpack. Fewer means a smaller run.')
    p.add_argument('--jobs', type=int, default=4, help='Parallel ffmpeg encodes.')
    p.add_argument('--threads', type=int, default=2, help='Threads per ffmpeg encode.')
    p.add_argument('--crf', type=int, default=18,
                   help='x264 quality, 0 = lossless. 18 is ~40 dB PSNR against the JPEGs.')
    p.add_argument('--preset', default='fast', help='x264 speed/size preset.')
    p.add_argument('--verify_only', action='store_true',
                   help='Only re-check an earlier run; encode and extract nothing.')
    p.add_argument('--check_frames', action='store_true',
                   help='Also ffprobe every transcoded clip and compare its frame count '
                        'against the JPEGs it came from. Thorough but slow.')
    args = p.parse_args()

    dest_root = args.dest or os.path.join(args.src, 'videos')
    anno = args.anno or os.path.join(args.src, 'ovbench.json')
    needed = wanted_ids(anno)

    for source in [s for s in args.sources if s in CONTAINER_SOURCES]:
        name, strip = CONTAINER_SOURCES[source]
        zip_path = os.path.join(args.src, f'{source}.zip')
        if not os.path.exists(zip_path):
            print(f'{source}.zip  --  absent, skipping', flush=True)
            continue
        print(f'{source}.zip  ->  videos/{name}/  (containers, extracted as shipped)',
              flush=True)
        do_container(zip_path, name, strip, os.path.join(dest_root, name), args.verify_only)

    # Build the whole encode list first, so one worker pool covers every source and the
    # slow sources cannot leave cores idle at the end of a fast one.
    jobs, present = [], 0
    # (source, clip) -> JPEGs in the zip, so --check_frames can hold a transcoded file
    # against the directory it came from even when this run did not encode it.
    jpeg_counts = {}
    for source in [s for s in args.sources if s in FRAME_SOURCES]:
        zip_path = os.path.join(args.src, f'{source}.zip')
        if not os.path.exists(zip_path):
            print(f'{source}.zip  --  absent, skipping', flush=True)
            continue
        dest = os.path.join(dest_root, source)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = collections.defaultdict(list)
            for m in zf.infolist():
                if m.is_dir():
                    continue
                top, _, base = m.filename.partition('/')
                if base and top in needed[source]:
                    members[top].append(m.filename)
        missing = sorted(set(needed[source]) - set(members))
        if missing:
            print(f'{source}: {len(missing)} referenced clip(s) absent from the zip, '
                  f'e.g. {missing[:3]}', flush=True)
        for clip, names in sorted(members.items()):
            jpeg_counts[(source, clip)] = len(names)
            out_path = os.path.join(dest, clip + '.mp4')
            if os.path.exists(out_path):
                present += 1
                continue
            entry = needed[source][clip]
            jobs.append((zip_path, source, clip, frame_order(names), entry['fps'],
                         out_path, args.crf, args.preset, args.threads))

    if jobs and not args.verify_only:
        n_frames_total = sum(len(j[3]) for j in jobs)
        print(f'\ntranscoding {len(jobs)} frame directories ({n_frames_total:,} JPEGs) '
              f'-> one .mp4 each, {args.jobs} at a time', flush=True)
        done = failed = 0
        t0 = time.time()
        with mp.Pool(args.jobs) as pool:
            for source, clip, n, size, dt, err in pool.imap_unordered(encode_clip, jobs):
                done += 1
                if err:
                    failed += 1
                    print(f'  [{done}/{len(jobs)}] FAILED {source}/{clip}: {err}', flush=True)
                else:
                    print(f'  [{done}/{len(jobs)}] {source}/{clip}  {n} frames  '
                          f'{size / 1e6:.1f} MB  {dt:.0f}s', flush=True)
        print(f'transcoded {done - failed}/{len(jobs)} in {(time.time() - t0) / 60:.1f} min'
              + (f', {failed} FAILED' if failed else ''), flush=True)
    elif present and not args.verify_only:
        print(f'\nall {present} referenced frame directories already transcoded')

    print('\nreferenced by ovbench.json, present on disk:')
    total = have = bad = 0
    n_files = 0
    for source in sorted(needed):
        root = os.path.join(dest_root, source)
        n = 0
        for rest in needed[source]:
            path = os.path.join(root, rest)
            path = path if os.path.exists(path) else path + '.mp4'
            if not os.path.exists(path):
                continue
            n += 1
            if args.check_frames and (source, rest) in jpeg_counts:
                got, want = n_frames(path), jpeg_counts[(source, rest)]
                if got != want:
                    bad += 1
                    print(f'    FRAME COUNT {source}/{rest}: {got} in the .mp4 vs '
                          f'{want} JPEGs in the zip')
        total += len(needed[source])
        have += n
        print(f'  {source:12s} {n:4d}/{len(needed[source]):4d}')
    for _, _, files in os.walk(dest_root):
        n_files += len(files)
    print(f'TOTAL {have}/{total} videos in {n_files} files under {dest_root}')
    if args.check_frames:
        print(f'frame-count check: {bad} mismatch(es)')
    if have < total or bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
