#!/bin/bash
#SBATCH --job-name=rekv_ablation
#SBATCH --output=/home/av354855/EfficientReKV/slurmlogs/rekv_ablation-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:4
#SBATCH --qos=group2-3           # up to 36 CPUs / 280G per job (group2-2 caps at 24 CPUs, too few for 4 GPUs x 8)
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=240G              # job 364488 (streambench, 2 workers, thr 0.25-0.9) peaked at 88G -> ~176G for 4; 240G still fits the 257G c2-1/c2-2 nodes

# Ablations of rlt_ref (ours). Each changes exactly one thing, so any accuracy gap is due
# to that one thing (metric, streaming state, first frame kept whole are all shared):
#     rlt_ref   ours       keep token iff dist(feat[t,p], last KEPT feat[p]) > thr   drift bounded by thr
#     rlt_prev  reference  keep token iff dist(feat[t,p], feat[t-1,p])       > thr   drift unbounded
#     rlt_frame granularity  keep ALL of frame t iff >50% of its tokens pass the rlt_ref test
# See model/token_pruning.py.
#
# At equal threshold the three keep different numbers of tokens (rlt_prev's frame-to-frame
# distances are smaller; rlt_frame keeps whole frames). Compare them at matched
# token_keep_rate (results.csv), not at matched threshold.
#
# Runs rlt_prev and rlt_frame; the rlt_ref arms at the same thresholds are already on
# disk from eval.slurm. Arms whose results.csv already exists are skipped, so an
# interrupted job resumes. FORCE=1 re-runs them.
#
# CAC ablation: PRUNE_CACHE=pad pads every pruned frame back to one full KV block, i.e.
# the pruner (TTS) on the unmodified ReKV cache layout, without content-adaptive caching.
#
# Results: results/<model>/<dataset>/64-<fps>-<method><thr>cosine[-pad]
# Usage:
#   sbatch scripts/eval/ablation.slurm
#   DATASETS="ovbench" ARMS="0.1 0.25" sbatch scripts/eval/ablation.slurm
#   METHODS=rlt_frame sbatch scripts/eval/ablation.slurm        # just one of the two
#   METHODS=rlt_ref PRUNE_CACHE=pad sbatch scripts/eval/ablation.slurm   # TTS without CAC

module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"
conda activate Rekv

export MPLBACKEND=Agg        # stop matplotlib dumping PNGs into the log
export TQDM_MININTERVAL=60

echo "Job started at $(date) on $(hostname)"
nvidia-smi

cd /home/av354855/EfficientReKV
# Repo root on sys.path so workers don't import the other ReKV clone
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

num_chunks="${NUM_CHUNKS:-${SLURM_GPUS_ON_NODE:-2}}"   # one process per GPU
models="${MODELS:-llava_ov_7b}"
datasets="${DATASETS:-streambench}"   # open-ended: run_eval scores it with the pinned Llama-3 judge
fps_list="${FPS_LIST:-1}"
arms="${ARMS:-0.25 0.5 0.6 0.7 0.8 0.9}"   # same grid as the rlt_ref arms on disk
methods="${METHODS:-rlt_prev rlt_frame}"
prune_cache="${PRUNE_CACHE:-adaptive}"   # adaptive = CAC (ours); pad = no CAC
cache_tag=""; [ "${prune_cache}" != adaptive ] && cache_tag="-${prune_cache}"
n_local=15000
retrieve_size=64

# Same formatting run_eval uses for the result dir: fps as a float, threshold as %g.
fmt_fps() { python -c "print(float('$1'))"; }
fmt_thr() { python -c "print(f'{float(\"$1\"):g}')"; }

done_already() {  # model dataset fps method thr
  local d
  for d in results/$1/$2/${retrieve_size}-$(fmt_fps $3)*-$4$(fmt_thr $5)cosine${cache_tag}; do
    [ -s "${d}/results.csv" ] && { echo "${d}"; return 0; }
  done
  return 1
}

for model in ${models}; do
  for dataset in ${datasets}; do
    for fps in ${fps_list}; do
      for method in ${methods}; do
        for thr in ${arms}; do
          if [ "${FORCE:-0}" != 1 ] && existing=$(done_already ${model} ${dataset} ${fps} ${method} ${thr}); then
            echo "=== skip ${model} | ${dataset} | ${fps} fps | ${method} ${thr}: ${existing} ==="
            continue
          fi
          echo "=== ${model} | ${dataset} | ${fps} fps | ${method} ${thr} ${prune_cache} ==="
          python -m video_qa.run_eval \
              --num_chunks ${num_chunks} \
              --model ${model} \
              --dataset ${dataset} \
              --sample_fps ${fps} \
              --n_local ${n_local} \
              --retrieve_size ${retrieve_size} \
              --prune_method ${method} \
              --prune_cache ${prune_cache} \
              --prune_threshold ${thr}
        done
      done
    done
  done
done

echo "Job finished at $(date)"
