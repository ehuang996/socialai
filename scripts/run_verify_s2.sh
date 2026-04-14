#!/bin/bash
# Run Stage 2 (low_quality_filter) for all 10 measures using existing Stage 1 output.
#
# Input (shared): experiments/verify/verify_1B_human_disfluencies/results/1B_human_disfluencies_coarse.jsonl
# Output: experiments/verify/verify_<measure>/results/<measure>_scores.jsonl
#
# Usage:
#   bash scripts/run_verify_s2.sh [--shards N]

set -e

MEASURES=(
    1B_human_disfluencies
    1B_human_pronoun
    1C_identity_transparency
    2A_fabricated_personal_details
    2B_explicit_emotions
    2B_implicit_emotions
    2B_romantic_bonding
    2C_sycophancy
    2D_human_relationship_encouragement
    3A_engagement_hooks
)

SHARDS=2
ACCOUNT="robinjia_875"

while [[ $# -gt 0 ]]; do
    case $1 in
        --shards) SHARDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ROOT=$(pwd)
COARSE_INPUT="${ROOT}/experiments/verify/verify_1B_human_disfluencies/results/1B_human_disfluencies_coarse.jsonl"

if [[ ! -f "$COARSE_INPUT" ]]; then
    echo "ERROR: Stage 1 output not found: $COARSE_INPUT"
    exit 1
fi

echo "======================================================"
echo "  Verify Stage 2 only: ${#MEASURES[@]} measures"
echo "  Stage 1 input: $COARSE_INPUT ($(wc -l < "$COARSE_INPUT") rows)"
echo "  Shards:  $SHARDS"
echo "======================================================"
echo ""

for MEASURE in "${MEASURES[@]}"; do
    echo "--- $MEASURE ---"

    EXP="${ROOT}/experiments/verify/verify_${MEASURE}"
    mkdir -p "${EXP}/logs" "${EXP}/results"

    OUT2_BASE="${EXP}/results/${MEASURE}_scores"
    SCORES_FINAL="${OUT2_BASE}.jsonl"

    export MEASURE STAGE="low_quality_filter"
    export INPUT_PATH="$COARSE_INPUT"
    export OUTPUT_BASE="$OUT2_BASE"
    export EXTRA_ARGS="--concurrency_limit 64"
    export EXPERIMENT_DIR=""

    JID2=$(sbatch --parsable \
        --job-name="v_${MEASURE:0:12}_s2" \
        --array=0-$((SHARDS-1)) \
        --output="${EXP}/logs/s2_%A_%a.out" \
        --error="${EXP}/logs/s2_%A_%a.err" \
        --export=ALL \
        slurm/run_vllm_stage.sbatch)

    CONCAT=$(sbatch --parsable \
        --dependency=afterok:${JID2} \
        --job-name="v_${MEASURE:0:12}_s2c" \
        --partition=nlp --account=${ACCOUNT} \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:30:00 \
        --output="${EXP}/logs/s2_concat_%j.out" \
        --wrap="cat ${OUT2_BASE}_part_*.jsonl > ${SCORES_FINAL} && rm ${OUT2_BASE}_part_*.jsonl && echo \"Stage 2 done: \$(wc -l < ${SCORES_FINAL}) rows -> ${SCORES_FINAL}\"")

    echo "  Stage 2: array=${JID2}  concat=${CONCAT}"
    echo "  Output: ${SCORES_FINAL}"
    echo ""
done

echo "All Stage 2 jobs submitted."
echo "Monitor with:  squeue -u \$USER"
