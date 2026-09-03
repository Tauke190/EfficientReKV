"""Unpack StreamingBench's 21 zips into a per-category video tree, then drop the zips.

StreamingBench ships one zip per task category (a few split into `_1-25`/`_26-50`
ranges), each containing top-level `sample_<N>/video.mp4` directories. The CSVs
address a video only through `question_id = "<category>_sample_<N>_<q>"`, so the
videos have to land under a directory named for that category prefix -- which is
*not* always the zip's name:

    Misleading Context Understanding.zip -> Misleading Context Recognition/
    Sequential Question Answering_*.zip  -> Sequential_Question_Answering/

The zips also carry macOS junk (`__MACOSX/`, `.DS_Store`) that is skipped, and
`Real-Time Visual Understanding_101-150.zip` additionally holds 1186 stray .jpg
frames -- upstream quirks, not a broken download.

Deleting a zip is gated on re-reading its central directory and confirming every
member exists on disk at the recorded size, so an interrupted run never removes
an archive it did not finish writing. Re-running is cheap: members already
present at the right size are skipped.

Usage:
    python scripts/setup_streamingbench.py                  # extract, then rm the zips
    python scripts/setup_streamingbench.py --keep_zips      # extract only
    python scripts/setup_streamingbench.py --verify_only    # check an earlier run
"""

import os
import sys
import time
import zipfile
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# zip basename (no .zip) -> directory name, which must equal the CSV question_id prefix
ZIP_TO_CATEGORY = {
    'Anomaly Context Understanding': 'Anomaly Context Understanding',
    'Misleading Context Understanding': 'Misleading Context Recognition',
    'Emotion Recognition': 'Emotion Recognition',
    'Multimodal Alignment': 'Multimodal Alignment',
    'Source Discrimination': 'Source Discrimination',
    'Scene Understanding_1-25': 'Scene Understanding',
    'Scene Understanding_26-50': 'Scene Understanding',
    'Sequential Question Answering_1-25': 'Sequential_Question_Answering',
    'Sequential Question Answering_26-50': 'Sequential_Question_Answering',
    'Proactive Output_1-25': 'Proactive Output',
    'Proactive Output_26-50': 'Proactive Output',
}
for _lo in range(1, 501, 50):
    ZIP_TO_CATEGORY[f'Real-Time Visual Understanding_{_lo}-{_lo + 49}'] = 'Real-Time Visual Understanding'


def wanted(member):
    """Real dataset content: macOS resource forks and .DS_Store are not."""
    n = member.filename
    return not (member.is_dir() or n.startswith('__MACOSX/')
                or os.path.basename(n).startswith('._')
                or os.path.basename(n) == '.DS_Store')


def members_present(zf, dest):
    """(n_ok, [missing/short paths]) for every real member of `zf` under `dest`."""
    ok, bad = 0, []
    for m in zf.infolist():
        if not wanted(m):
            continue
        t = os.path.join(dest, m.filename)
        if os.path.exists(t) and os.path.getsize(t) == m.file_size:
            ok += 1
        else:
            bad.append(m.filename)
    return ok, bad


def extract(zip_path, dest):
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if wanted(m)]
        skipped = 0
        for i, m in enumerate(members):
            target = os.path.join(dest, m.filename)
            if os.path.exists(target) and os.path.getsize(target) == m.file_size:
                skipped += 1
                continue
            zf.extract(m, dest)
            if (i + 1) % 200 == 0:
                print(f'    {i + 1}/{len(members)}', flush=True)
    print(f'    {len(members)} members ({skipped} already present) in {time.time() - t0:.0f}s',
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.path.join(REPO_ROOT, 'data/StreamingBench'),
                   help='Directory holding the downloaded *.zip files.')
    p.add_argument('--dest', default=None,
                   help='Where the video tree lands. Default: <src>/videos')
    p.add_argument('--keep_zips', action='store_true', help='Do not delete zips after extracting.')
    p.add_argument('--verify_only', action='store_true', help='Only re-check an earlier extraction.')
    args = p.parse_args()
    dest_root = args.dest or os.path.join(args.src, 'videos')

    zips = sorted(f for f in os.listdir(args.src) if f.endswith('.zip'))
    if not zips and not args.verify_only:
        print(f'No zips in {args.src} -- already extracted?')
    unknown = [z for z in zips if z[:-4] not in ZIP_TO_CATEGORY]
    if unknown:
        sys.exit(f'Unrecognised zip(s), refusing to guess a category: {unknown}')

    freed = 0
    for z in zips:
        category = ZIP_TO_CATEGORY[z[:-4]]
        dest = os.path.join(dest_root, category)
        os.makedirs(dest, exist_ok=True)
        print(f'{z}  ->  videos/{category}/', flush=True)

        if not args.verify_only:
            extract(os.path.join(args.src, z), dest)

        with zipfile.ZipFile(os.path.join(args.src, z)) as zf:
            ok, bad = members_present(zf, dest)
        if bad:
            print(f'    INCOMPLETE: {len(bad)} member(s) missing or short, keeping zip'
                  f' (first: {bad[0]})', flush=True)
            continue
        print(f'    verified {ok} members', flush=True)

        if not args.keep_zips and not args.verify_only:
            size = os.path.getsize(os.path.join(args.src, z))
            os.remove(os.path.join(args.src, z))
            freed += size
            print(f'    removed zip ({size / 1e9:.1f} GB)', flush=True)

    samples = 0
    for category in sorted(set(ZIP_TO_CATEGORY.values())):
        d = os.path.join(dest_root, category)
        n = len([x for x in os.listdir(d) if x.startswith('sample_')]) if os.path.isdir(d) else 0
        samples += n
        print(f'{category:35s} {n:4d} samples')
    print(f'TOTAL {samples} samples (expected 900); freed {freed / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
