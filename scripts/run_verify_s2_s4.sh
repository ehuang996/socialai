#!/bin/bash
# Run stages 2-4 for all 10 measures using the existing Stage 1 coarse output.
#
# Stage 1 output (shared across all measures since the coarse filter prompt is identical):
#   experiments/verify/verify_1B_human_disfluencies/results/1B_human_disfluencies_coarse.jsonl
#
# Creates experiments/verify/verify_<measure>/ dirs for each measure.
# Within each measure, stages 2→3→4 are chained via SLURM job dependencies.
#
# Usage:
#   bash scripts/run_verify_s2_s4.sh --key path/to/openrouter_api_key [--shards N]

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

KEY=""
SHARDS=2
ACCOUNT="robinjia_875"

while [[ $# -gt 0 ]]; do
    case $1 in
        --key)    KEY="$2";    shift 2 ;;
        --shards) SHARDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$KEY" ]]; then
    echo "ERROR: --key <path> is required (path to OpenRouter API key file)"
    exit 1
fi

ROOT=$(pwd)
COARSE_INPUT="${ROOT}/experiments/verify/verify_1B_human_disfluencies/results/1B_human_disfluencies_coarse.jsonl"

if [[ ! -f "$COARSE_INPUT" ]]; then
    echo "ERROR: Stage 1 output not found: $COARSE_INPUT"
    exit 1
fi

echo "======================================================"
echo "  Verify pipeline (stages 2-4): ${#MEASURES[@]} measures"
echo "  Stage 1 input: $COARSE_INPUT ($(wc -l < "$COARSE_INPUT") rows)"
echo "  Shards:  $SHARDS (for Stage 2 vLLM)"
echo "  Key:     $KEY"
echo "======================================================"
echo ""

for MEASURE in "${MEASURES[@]}"; do
    echo "--- $MEASURE ---"

    EXP="${ROOT}/experiments/verify/verify_${MEASURE}"
    mkdir -p "${EXP}/logs" "${EXP}/results"

    # ── Stage 2: low_quality_filter (vLLM array job) ─────────────────────────
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

    CONCAT2=$(sbatch --parsable \
        --dependency=afterok:${JID2} \
        --job-name="v_${MEASURE:0:12}_s2c" \
        --partition=nlp --account=${ACCOUNT} \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:30:00 \
        --output="${EXP}/logs/s2_concat_%j.out" \
        --wrap="cat ${OUT2_BASE}_part_*.jsonl > ${SCORES_FINAL} && rm ${OUT2_BASE}_part_*.jsonl && echo \"Stage 2 done: \$(wc -l < ${SCORES_FINAL}) rows -> ${SCORES_FINAL}\"")

    echo "  Stage 2: array=${JID2}  concat=${CONCAT2}"

    # ── Stage 3: high_quality_filter (OpenRouter GPT-4o-mini) ─────────────────
    HQ_FINAL="${EXP}/results/${MEASURE}_high_quality.jsonl"

    JID3=$(sbatch --parsable \
        --dependency=afterok:${CONCAT2} \
        --job-name="v_${MEASURE:0:12}_s3" \
        --partition=nlp --account=${ACCOUNT} \
        --ntasks=1 --cpus-per-task=4 --mem=16G --time=4:00:00 \
        --output="${EXP}/logs/s3_%j.out" \
        --error="${EXP}/logs/s3_%j.err" \
        --wrap="cd ${ROOT} && module purge && module load gcc/13.3.0 && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run python src/filter/run.py --measure ${MEASURE} --stage high_quality_filter --input_path ${SCORES_FINAL} --output_path ${HQ_FINAL} --key ${KEY}")

    echo "  Stage 3: ${JID3}"

    # ── Stage 4: final_filter (OpenRouter Claude Opus 4.6) ───────────────────
    FINAL="${EXP}/results/${MEASURE}_final.jsonl"

    JID4=$(sbatch --parsable \
        --dependency=afterok:${JID3} \
        --job-name="v_${MEASURE:0:12}_s4" \
        --partition=nlp --account=${ACCOUNT} \
        --ntasks=1 --cpus-per-task=4 --mem=16G --time=4:00:00 \
        --output="${EXP}/logs/s4_%j.out" \
        --error="${EXP}/logs/s4_%j.err" \
        --wrap="cd ${ROOT} && module purge && module load gcc/13.3.0 && export PATH=\"\$HOME/.local/bin:\$PATH\" && uv run python src/filter/run.py --measure ${MEASURE} --stage final_filter --input_path ${HQ_FINAL} --output_path ${FINAL} --key ${KEY}")

    echo "  Stage 4: ${JID4}"
    echo "  Final output: ${FINAL}"
    echo ""
done

echo "All jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Final outputs: experiments/verify/verify_<measure>/results/<measure>_final.jsonl"
