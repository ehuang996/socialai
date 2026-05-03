#!/bin/bash
# Stage 7.1 add-on: generate Qwen3 1.7B and 4B locally with vLLM, then merge
# those columns into data/eval_responses.jsonl for the normal Stage 7.2 judge.
#
# The main API/direct sweep remains scripts/run_stage7_1_generate.sh:
#   23 API/direct models = 22,287 generations
# This add-on:
#   2 local HF models x 969 rows = 1,938 generations
# Combined Stage 7.1 after merge:
#   25 models x 969 rows = 24,225 generations

set -euo pipefail

cd /project2/robinjia_875/ehuang97/socialai

SHARDS=${SHARDS:-6}
CONCURRENCY=${CONCURRENCY:-32}
OUTPUT_BASE=${OUTPUT_BASE:-data/eval_responses_local_qwen}
INPUT_PATH=${INPUT_PATH:-data/final_dataset.jsonl}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
ACCOUNT=${ACCOUNT:-robinjia_875}

mkdir -p experiments
LAST_NUM=$(find experiments -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -oE '^[0-9]+' | sort -n | tail -1 || true)
N=$(( 10#${LAST_NUM:-0} + 1 ))
EXPERIMENT_DIR="experiments/$(printf '%02d' "$N")_stage7_1_local_qwen"
mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/figures" "${EXPERIMENT_DIR}/results"

ARRAY_SPEC="0-$((SHARDS - 1))"

echo "======================================================"
echo "  Stage:      7.1 local Qwen add-on"
echo "  Input:      ${INPUT_PATH}"
echo "  Shards:     ${SHARDS}"
echo "  Output:     ${OUTPUT_BASE}_<model>_part_<shard>.jsonl"
echo "  Experiment: ${EXPERIMENT_DIR}"
echo "======================================================"

JID_17=$(sbatch --parsable \
    --job-name=stage7_qwen3_1_7b \
    --array="${ARRAY_SPEC}" \
    --output="${EXPERIMENT_DIR}/logs/qwen3_1_7b_%A_%a.out" \
    --error="${EXPERIMENT_DIR}/logs/qwen3_1_7b_%A_%a.err" \
    --export=ALL,MODEL_COL=qwen3_1_7b,HF_MODEL=Qwen/Qwen3-1.7B,INPUT_PATH="${INPUT_PATH}",OUTPUT_BASE="${OUTPUT_BASE}",CONCURRENCY="${CONCURRENCY}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    slurm/run_eval_local_qwen.sbatch)
echo "Submitted qwen3_1_7b array job: $JID_17"

JID_4=$(sbatch --parsable \
    --job-name=stage7_qwen3_4b \
    --array="${ARRAY_SPEC}" \
    --output="${EXPERIMENT_DIR}/logs/qwen3_4b_%A_%a.out" \
    --error="${EXPERIMENT_DIR}/logs/qwen3_4b_%A_%a.err" \
    --export=ALL,MODEL_COL=qwen3_4b,HF_MODEL=Qwen/Qwen3-4B,INPUT_PATH="${INPUT_PATH}",OUTPUT_BASE="${OUTPUT_BASE}",CONCURRENCY="${CONCURRENCY}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    slurm/run_eval_local_qwen.sbatch)
echo "Submitted qwen3_4b array job: $JID_4"

MERGE=$(sbatch --parsable \
    --dependency=afterok:"${JID_17}":"${JID_4}" \
    --job-name=stage7_qwen_merge \
    --partition=nlp \
    --account="$ACCOUNT" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=0:30:00 \
    --output="${EXPERIMENT_DIR}/logs/merge_%j.out" \
    --error="${EXPERIMENT_DIR}/logs/merge_%j.err" \
    --wrap="cd /project2/robinjia_875/ehuang97/socialai && uv run --no-sync python -m src.evaluation.merge_local_responses --input ${INPUT_PATH} --base-output data/eval_responses.jsonl --local-parts '${OUTPUT_BASE}_*.jsonl' --output data/eval_responses.jsonl")
echo "Submitted merge job: $MERGE (depends on $JID_17 and $JID_4)"

echo ""
echo "Job chain: $JID_17 + $JID_4 -> $MERGE"
echo "Logs: ${EXPERIMENT_DIR}/logs/"
echo "Merged output: data/eval_responses.jsonl"
