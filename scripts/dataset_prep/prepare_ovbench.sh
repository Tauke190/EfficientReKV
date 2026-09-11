#!/bin/bash
# OVBench: unpack the download, then build the annotation file.
#
# Assumes data/OVBench holds what `huggingface-cli download MCG-NJU/OVBench --repo-type
# dataset --local-dir data/OVBench` produced: ten zips and ovbench.json.
#
# Unlike prepare_streamingbench.sh this does NOT delete the zips. Seven of the ten hold
# directories of JPEGs rather than videos, and setup_ovbench.py transcodes those straight
# out of the zip into one .mp4 per clip -- a lossy re-encode at CRF 18. Deleting the zips
# would make that irreversible, so it is left as a deliberate, separate decision. Budget
# ~190 GB for the zips plus ~77 GB for the output.
#
# The transcode is the expensive step (~2 h at --jobs 4) and is resumable: a clip whose
# .mp4 already exists is skipped, so re-running this script after an interruption costs
# only the remaining clips.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:-data/OVBench}
JOBS=${JOBS:-4}

python scripts/dataset_prep/setup_ovbench.py --src "${DATA}" --jobs "${JOBS}"

# Confirms every transcoded clip decodes with exactly as many frames as the JPEG directory
# it came from. Slow (it re-reads the zip central directories), but this is the only check
# that would catch a truncated encode, and a short clip shifts every timestamp after it.
python scripts/dataset_prep/setup_ovbench.py --src "${DATA}" --verify_only --check_frames

python video_qa/convert_ovbench.py \
    --src "${DATA}/ovbench.json" \
    --video_root "${DATA}/videos" \
    --out "${DATA}/full_mc.json"
