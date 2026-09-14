"""The LLM judges' per-item verdict cache, kept in ONE file per run.

The cluster home this runs on has an inode (file-count) quota. The judges used to write
one JSON per question into a `tmp_*/` directory, and a sweep of ~40 runs left ~60k of
them in results/ -- enough, with everything else, to hit the quota and stop every write,
git included. So a cache is now a single append-only JSONL beside where that directory
used to be: `--output_dir <run>/tmp_local_streambench` means `<run>/tmp_local_streambench.jsonl`.

Lines are appended as soon as an item is judged, so an interrupted run still resumes. A
line cut short by a crash does not parse, is skipped, and that item is judged again.

    {"key": "<video_id>_<n>", "result": [verdict, qa_set]}   one per judged item
    {"judge_model": "..."}                                    provenance, any position

A leftover `tmp_*/` directory from the old layout is refused rather than silently
ignored (ignoring it would re-judge everything, and for the API judge that costs money).
Pack old directories with:

    python video_qa/eval/verdict_cache.py results/
"""

import json
import os
import shutil
import sys


def path_of(cache):
    """`<run>/tmp_x` or `<run>/tmp_x.jsonl` -> `<run>/tmp_x.jsonl`."""
    cache = cache.rstrip('/')
    return cache if cache.endswith('.jsonl') else cache + '.jsonl'


def _refuse_legacy_dir(cache):
    legacy = path_of(cache)[:-len('.jsonl')]
    if os.path.isdir(legacy):
        raise SystemExit(
            f'{legacy}/ is an old one-file-per-item cache. Pack it into '
            f'{path_of(cache)} first (the inode quota cannot afford it):\n'
            f'    python video_qa/eval/verdict_cache.py {legacy}')


def load(cache):
    """-> (items, meta): key -> [verdict, qa_set], and the merged provenance lines."""
    _refuse_legacy_dir(cache)
    items, meta = {}, {}
    path = path_of(cache)
    if not os.path.exists(path):
        return items, meta
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # torn last line from an interrupted run
            if 'key' in rec:
                items[rec['key']] = rec['result']
            else:
                meta.update(rec)
    return items, meta


def append(cache, results):
    """Append {key: [verdict, qa_set]} to the cache, flushed before returning."""
    path = path_of(cache)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        for key, result in results.items():
            f.write(json.dumps({'key': key, 'result': result}) + '\n')
        f.flush()
        os.fsync(f.fileno())


def write_meta(cache, **meta):
    path = path_of(cache)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(meta) + '\n')


def pack_dir(legacy):
    """Fold an old `tmp_*/` directory into `<dir>.jsonl`, verify it, then delete the dir."""
    legacy = legacy.rstrip('/')
    if not os.listdir(legacy):
        os.rmdir(legacy)
        return 0, '(empty, removed)'
    items = {}
    for name in sorted(os.listdir(legacy)):
        if name.endswith('.json'):
            with open(os.path.join(legacy, name)) as f:
                items[name[:-len('.json')]] = json.load(f)
    meta = {}
    marker = os.path.join(legacy, '.judge_model')
    if os.path.exists(marker):
        with open(marker) as f:
            meta['judge_model'] = f.read().strip()

    path = path_of(legacy)
    # Merge with anything already packed rather than overwrite it.
    existing, existing_meta = {}, {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if 'key' in rec:
                    existing[rec['key']] = rec['result']
                else:
                    existing_meta.update(rec)
    if meta and existing_meta.get('judge_model', meta['judge_model']) != meta['judge_model']:
        raise SystemExit(f'{legacy}: judge {meta} disagrees with {path}: {existing_meta}')
    merged = {**existing, **items}
    merged_meta = {**existing_meta, **meta}

    tmp = path + '.partial'
    with open(tmp, 'w') as f:
        if merged_meta:
            f.write(json.dumps(merged_meta) + '\n')
        for key, result in merged.items():
            f.write(json.dumps({'key': key, 'result': result}) + '\n')
    os.replace(tmp, path)

    # Only delete once the file reads back identical.
    shutil.move(legacy, legacy + '.__packing')
    try:
        got, got_meta = load(legacy)
        if got != merged or got_meta != merged_meta:
            raise SystemExit(f'{path} does not read back what was packed; kept {legacy}')
    except BaseException:
        shutil.move(legacy + '.__packing', legacy)
        raise
    shutil.rmtree(legacy + '.__packing')
    return len(items), path


def main(roots):
    dirs = []
    for root in roots:
        root = root.rstrip('/')
        if os.path.basename(root).startswith('tmp') and os.path.isdir(root):
            dirs.append(root)
            continue
        for dirpath, dirnames, _ in os.walk(root):
            for d in list(dirnames):
                if d.startswith('tmp'):
                    dirs.append(os.path.join(dirpath, d))
                    dirnames.remove(d)
    total = 0
    for d in sorted(dirs):
        n, path = pack_dir(d)
        total += n
        print(f'{n:6d} items -> {path}')
    print(f'packed {total} items from {len(dirs)} directories')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
