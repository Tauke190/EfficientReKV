#!/bin/bash
# StreamingBench: unpack the download, then build the four annotation files.
#
# Assumes data/StreamingBench holds what `huggingface-cli download mjuicem/StreamingBench
# --repo-type dataset --local-dir data` produced: 21 zips, six CSVs, README.md. The zips
# are deleted as they are verified, so this needs ~189 GB free, not ~378 GB.
#
# Proactive Output is unpacked (it is in the same download) but not converted: it is
# scored on *when* the model speaks, not on a letter. See video_qa/run_eval.py.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/StreamingBench}

python scripts/setup_streamingbench.py --src "${DATA}"

for subset in real omni context sqa; do
    python video_qa/convert_streamingbench.py \
        --subset ${subset} \
        --csv_root "${DATA}" \
        --video_root "${DATA}/videos" \
        --out "${DATA}/${subset}.json"
done
