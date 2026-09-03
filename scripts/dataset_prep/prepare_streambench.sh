#!/bin/bash
# StreamBench: build the annotation file from the download.
#
# Assumes data/streambench holds what the HF download produced: streaming_bench_v0.3.json
# plus the Ego/, WebVideo/ and Movie/ directories. No unpacking step -- the videos ship as
# mp4s, already extracted.
#
# Note the archives also unpacked one AppleDouble sidecar per video (._<name>.mp4, 306 of
# them, a few KB each). They are macOS resource forks, not video, and nothing reads them;
# the converter never resolves to one because it builds paths from the annotation. They
# are left in place here because deleting files is not this script's job -- remove them
# with `find data/streambench -name '._*' -delete` if the inode count matters.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/streambench}

python video_qa/convert_streambench.py \
    --src "${DATA}/streaming_bench_v0.3.json" \
    --video_root "${DATA}" \
    --out "${DATA}/full_oe.json"
