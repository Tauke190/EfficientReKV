"""One-shot setup for ODV-Bench: fetch from the Hub, unpack, build the ReKV annotation.

ODV-Bench is distributed as a Hugging Face *dataset repo*, not a directory of videos.
`huggingface_hub` stores such a repo as a content-addressed cache:

    datasets--MCG-NJU--ODV-Bench/
      blobs/<sha256>       the actual bytes, named by hash
      snapshots/<commit>/  one directory per git commit, filled with symlinks into blobs/
      refs/main            the commit id that `main` currently points at

so `snapshots/<commit>/TR_Analysis.zip -> ../../blobs/7515b4...` is a real 7.7 GB file seen
through a symlink. Nothing is missing and nothing needs converting -- the videos are simply
still inside the three zips, and the annotation is the 4 MB ODVbench.json beside them.

This script resolves the snapshot (downloading it if absent), extracts each zip into
`--dest` under its own name -- which is exactly the prefix ODVbench.json uses for its
`video` paths -- copies the annotation in, and runs video_qa/convert_odvbench.py to emit
`full_mc.json` in ReKV's video-level schema.

Needs ~14 GB for the extracted videos on top of the ~13.8 GB cache. Re-running is cheap:
extraction skips files already present at the right size.

Usage:
    python scripts/dataset_prep/setup_odvbench.py                    # cache -> data/odvbench
    python scripts/dataset_prep/setup_odvbench.py --no_probe         # skip the decord duration probe
    python scripts/dataset_prep/setup_odvbench.py --zips TS_Retrieval  # one collection only
"""

import os
import sys
import json
import time
import shutil
import zipfile
import argparse
import subprocess

REPO_ID = 'MCG-NJU/ODV-Bench'
ZIPS = ['TS_Retrieval', 'TOI_Recognition', 'TR_Analysis']
ANNO = 'ODVbench.json'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_snapshot(local_dir=None):
    """Path to the dataset snapshot, downloading it only if it is not already cached."""
    if local_dir:
        return local_dir
    from huggingface_hub import snapshot_download
    # No allow_patterns: the caller may later want a collection they did not ask for
    # today, and a partial snapshot makes that a second, confusing download.
    return snapshot_download(repo_id=REPO_ID, repo_type='dataset')


def extract(zip_path, dest):
    """Extract `zip_path` into `dest`, skipping members already there at the right size."""
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as f:
        members = [m for m in f.infolist() if not m.is_dir()]
        skipped = 0
        for i, m in enumerate(members):
            target = os.path.join(dest, m.filename)
            if os.path.exists(target) and os.path.getsize(target) == m.file_size:
                skipped += 1
                continue
            f.extract(m, dest)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(members)}', flush=True)
    print(f'  {len(members)} files ({skipped} already present) in {time.time() - t0:.0f}s',
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', type=str, default=os.path.join(REPO_ROOT, 'data/odvbench'),
                        help='Where videos and annotations land.')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='An already-downloaded snapshot directory. Default: ask '
                             'huggingface_hub, which downloads only what is not cached.')
    parser.add_argument('--zips', nargs='+', default=ZIPS, choices=ZIPS,
                        help='Collections to extract. Fewer means a smaller run; pass the '
                             'same subset through to the converter with --skip_missing.')
    parser.add_argument('--no_probe', action='store_true',
                        help="Skip the converter's decord duration probe.")
    parser.add_argument('--skip_extract', action='store_true',
                        help='Only (re)build the annotation from videos already extracted.')
    args = parser.parse_args()

    snap = resolve_snapshot(args.snapshot)
    print(f'snapshot: {snap}')
    os.makedirs(args.dest, exist_ok=True)

    if not args.skip_extract:
        for z in args.zips:
            src = os.path.join(snap, f'{z}.zip')
            if not os.path.exists(src):
                raise SystemExit(f'missing {src} -- is the snapshot complete?')
            print(f'extracting {z}.zip -> {args.dest}/{z}', flush=True)
            extract(src, os.path.join(args.dest, z))

    # Copied rather than symlinked: the annotation is 4 MB and the cache is prunable.
    anno = os.path.join(args.dest, ANNO)
    shutil.copyfile(os.path.join(snap, ANNO), anno)
    print(f'annotation: {anno} ({len(json.load(open(anno)))} questions)')

    cmd = [sys.executable, os.path.join(REPO_ROOT, 'video_qa/convert_odvbench.py'),
           '--src', anno,
           '--video_root', args.dest,
           '--out', os.path.join(args.dest, 'full_mc.json')]
    if args.no_probe:
        cmd.append('--no_probe')
    if set(args.zips) != set(ZIPS):
        cmd.append('--skip_missing')
    print('running: ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
