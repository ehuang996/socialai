#!/bin/bash
# Run one stage of the WildChat pipeline for a given measure.
#
# Usage (from project root):
#   bash pipeline.sh --measure <measure> --stage <stage> [options]
#
# Stages:
#   coarse_filter       Stage 1 (vLLM): coarse-filter raw conversations
#   low_quality_filter  Stage 2 (vLLM): score filtered conversations
#   high_quality_filter Stage 3 (OpenRouter API): evaluate seed prompts
#
# Creates experiments/<NN>_<stage>/{figures,logs,results}/ automatically.

set -e

MEASURE=""
STAGE=""
SHARDS=6
INPUT=""
KEY=""
EXCLUDE=""
DEPENDS=""
ACCOUNT="robinjia_875"

while [[ $# -gt 0 ]]; do
    case $1 in
        --measure)    MEASURE="$2";  shift 2 ;;
        --stage)      STAGE="$2";    shift 2 ;;
        --shards)     SHARDS="$2";   shift 2 ;;
        --input)      INPUT="$2";    shift 2 ;;
        --key)        KEY="$2";      shift 2 ;;
        --exclude)    EXCLUDE="$2";  shift 2 ;;
        --depends-on) DEPENDS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MEASURE" ] || [ -z "$STAGE" ]; then
    echo "Usage: bash pipeline.sh --measure <name> --stage <stage> [options]"
    echo "Stages: coarse_filter, low_quality_filter, high_quality_filter, final_filter"
    echo "Options:"
    echo "  --shards N    Number of array shards (default: 6, for vLLM stages)"
    echo "  --input  path Input file (default: data/wildchat_raw.jsonl for coarse_filter;"
    echo "                required for low_quality_filter and high_quality_filter)"
    echo "  --key    path OpenRouter API key file (required for high_quality_filter)"
    exit 1
fi

# Default input by stage
if [ -z "$INPUT" ]; then
    case $STAGE in
        coarse_filter)
            INPUT="data/wildchat_raw.jsonl"
            ;;
        *)
            echo "ERROR: --input is required for stage '$STAGE'"
            exit 1
            ;;
    esac
fi

# Auto-number the experiment folder
LAST_NUM=$(ls experiments/ 2>/dev/null | grep -oE '^[0-9]+' | sort -n | tail -1)
N=$(( 10#${LAST_NUM:-0} + 1 ))
EXPERIMENT_DIR="experiments/$(printf '%02d' $N)_${STAGE}"

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/figures" "${EXPERIMENT_DIR}/results"

echo "======================================================"
echo "  Stage:      ${STAGE}"
echo "  Measure:    ${MEASURE}"
echo "  Input:      ${INPUT}"
echo "  Experiment: ${EXPERIMENT_DIR}"
echo "======================================================"

case $STAGE in

    coarse_filter|low_quality_filter)
        # ── vLLM array job ─────────────────────────────────────────────────
        ARRAY_SPEC="0-$((SHARDS-1))"
        export MEASURE STAGE EXPERIMENT_DIR
        export INPUT_PATH="$INPUT"

        if [ "$STAGE" = "coarse_filter" ]; then
            export OUTPUT_BASE="data/${MEASURE}_coarse"
            export EXTRA_ARGS="--prompt-version v1 --concurrency_limit 100"
            OUTPUT_FILE="${EXPERIMENT_DIR}/results/${MEASURE}_coarse.jsonl"
        else
            export OUTPUT_BASE="data/${MEASURE}_scores"
            export EXTRA_ARGS="--concurrency_limit 64"
            OUTPUT_FILE="${EXPERIMENT_DIR}/results/${MEASURE}_scores.jsonl"
        fi

        JID=$(sbatch --parsable \
            --job-name=${MEASURE}_${STAGE} \
            --array=${ARRAY_SPEC} \
            --output=${EXPERIMENT_DIR}/logs/job_%A_%a.out \
            --error=${EXPERIMENT_DIR}/logs/job_%A_%a.err \
            --export=ALL \
            ${EXCLUDE:+--exclude=${EXCLUDE}} \
            ${DEPENDS:+--dependency=afterok:${DEPENDS}} \
            slurm/run_vllm_stage.sbatch)
        echo "Array job submitted: $JID"

        CONCAT=$(sbatch --parsable \
            --dependency=afterok:$JID \
            --job-name=${MEASURE}_concat \
            --partition=nlp \
            --account=$ACCOUNT \
            --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:30:00 \
            --output=${EXPERIMENT_DIR}/logs/concat_%j.out \
            --wrap="cat ${OUTPUT_BASE}_part_*.jsonl > ${OUTPUT_FILE} && rm ${OUTPUT_BASE}_part_*.jsonl && echo \"Concatenated \$(wc -l < ${OUTPUT_FILE}) rows -> ${OUTPUT_FILE}\"")
        echo "Concat job submitted: $CONCAT (depends on $JID)"

        echo ""
        echo "Job chain: $JID → $CONCAT"
        echo "Final output: ${OUTPUT_FILE}"
        ;;

    high_quality_filter|final_filter)
        # ── Sharded OpenRouter API stage ──────────────────────────────────
        if [ -z "$KEY" ]; then
            echo "ERROR: --key is required for $STAGE"
            exit 1
        fi

        if [ "$STAGE" = "high_quality_filter" ]; then
            SUFFIX="high_quality"
        else
            SUFFIX="final"
        fi

        OUTPUT_FILE="${EXPERIMENT_DIR}/results/${MEASURE}_${SUFFIX}.jsonl"
        export OUTPUT_BASE="data/${MEASURE}_${SUFFIX}"
        export MEASURE STAGE EXPERIMENT_DIR KEY
        export INPUT_PATH="$INPUT"
        export EXTRA_ARGS=""

        ARRAY_SPEC="0-$((SHARDS-1))"

        JID=$(sbatch --parsable \
            --job-name=${MEASURE}_${SUFFIX} \
            --array=${ARRAY_SPEC} \
            --output=${EXPERIMENT_DIR}/logs/job_%A_%a.out \
            --error=${EXPERIMENT_DIR}/logs/job_%A_%a.err \
            --export=ALL \
            ${DEPENDS:+--dependency=afterok:${DEPENDS}} \
            slurm/run_api_stage.sbatch)
        echo "Array job submitted: $JID"

        CONCAT=$(sbatch --parsable \
            --dependency=afterok:$JID \
            --job-name=${MEASURE}_${SUFFIX}_concat \
            --partition=nlp \
            --account=$ACCOUNT \
            --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:30:00 \
            --output=${EXPERIMENT_DIR}/logs/concat_%j.out \
            --wrap="cat ${OUTPUT_BASE}_part_*.jsonl > ${OUTPUT_FILE} && rm ${OUTPUT_BASE}_part_*.jsonl && echo \"Concatenated \$(wc -l < ${OUTPUT_FILE}) rows -> ${OUTPUT_FILE}\"")
        echo "Concat job submitted: $CONCAT (depends on $JID)"

        echo ""
        echo "Job chain: $JID → $CONCAT"
        echo "Final output: ${OUTPUT_FILE}"
        ;;

    *)
        echo "ERROR: Unknown stage '$STAGE'. Use: coarse_filter, low_quality_filter, high_quality_filter, final_filter"
        exit 1
        ;;

esac

echo "Logs: ${EXPERIMENT_DIR}/logs/"
