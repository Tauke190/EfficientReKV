# The number of processes utilized for parallel evaluation.
# Normally, set it to the number of GPUs on your machine.
# Yet, llava_ov_72b needs 4x 80GB GPUs. So set num_chunks to num_gpus//4.
num_chunks=1

# Supported model: llava_ov_0.5b llava_ov_7b llava_ov_72b video_llava_7b longva_7b
model=llava_ov_0.5b

# Supported dataset: qaego4d egoschema cgbench mlvu activitynet_qa rvs_ego rvs_movie
# MLVU has an extremely long video (~9hr). Remove it in the annotation file if your system doesn't have enough RAM.
# Space-separated: each dataset is evaluated in turn.
datasets="rvs_ego rvs_movie"

# rvs_* answers are free-form text, so accuracy needs an LLM judge to decide whether a
# prediction means the same as the reference -- there is no string-match accuracy for
# them. run_eval.py's judge defaults to a local one (own GPU, no API key): Prometheus 2,
# an open evaluator model. Here we skip scoring entirely and just write predictions to
# results.csv, so inference isn't holding a GPU hostage behind the judge. To score later:
#   scripts/score_open_ended.sh results/${model}/rvs_ego/64-0.5/results.csv
# with another judge (presets in video_qa/eval/judges.py; each writes its own file):
#   JUDGE=qwen scripts/score_open_ended.sh results/${model}/rvs_ego/64-0.5/results.csv
# or with the API judge the published numbers used:
#   export OPENAI_API_KEY=sk-...        # OPENAI_BASE_URL / OPENAI_JUDGE_MODEL optional
#   python video_qa/eval/eval_open_ended.py \
#       --pred_path  results/${model}/rvs_ego/64-0.5/results.csv \
#       --output_dir results/${model}/rvs_ego/64-0.5/tmp \
#       --output_json results/${model}/rvs_ego/64-0.5/results.json
skip_scoring=--skip_scoring

for dataset in ${datasets}; do
    echo "=== Evaluating ${model} on ${dataset} ==="
    python -m video_qa.run_eval \
        --num_chunks $num_chunks \
        --model ${model} \
        --dataset ${dataset} \
        --sample_fps 0.5 \
        --n_local 15000 \
        --retrieve_size 64 \
        ${skip_scoring}
done
