#!/bin/bash
# Run single_turn_104_prepared.jsonl through all 4 filter stages for all 10 measures.
#
# Creates experiments/verify_<measure>/ dirs (bypasses pipeline.sh auto-numbering).
# All 9 measures are submitted in parallel; within each measure, stages 1→2→3→4
# are chained via SLURM job dependencies.
#
# Prerequisites:
#   uv run python scripts/prepare_verify.py   # creates data/single_turn_104_prepared.jsonl
#
# Usage:
#   bash scripts/run_verify.sh --key path/to/openrouter_api_key [--shards N]

set -e

MEASURES=(
    1B_intentional_human_speech
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

INPUT="data/verify/single_turn_104_prepared.jsonl"
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found."
    echo "Run first: uv run python scripts/prepare_verify.py"
    exit 1
fi

ROOT=$(pwd)

echo "======================================================"
echo "  Verify pipeline: single_turn_104 × ${#MEASURES[@]} measures"
echo "  Input:   $INPUT"
echo "  Shards:  $SHARDS (per vLLM stage)"
echo "  Key:     $KEY"
echo "======================================================"
echo ""

for MEASURE in "${MEASURES[@]}"; do
    echo "--- $MEASURE ---"

    EXP="${ROOT}/experiments/verify_${MEASURE}"
    mkdir -p "${EXP}/logs" "${EXP}/results"

    # ── Stage 1: coarse_filter (vLLM array job) ─────────────────────────────
    OUT1_BASE="${EXP}/results/${MEASURE}_coarse"
    COARSE_FINAL="${OUT1_BASE}.jsonl"

    export MEASURE STAGE="coarse_filter" EXPERIMENT_DIR="$EXP"
    export INPUT_PATH="$INPUT"
    export OUTPUT_BASE="$OUT1_BASE"
    export EXTRA_ARGS="--prompt-version v1 --concurrency_limit 100"

    JID1=$(sbatch --parsable \
        --job-name="v_${MEASURE:0:12}_s1" \
        --array=0-$((SHARDS-1)) \
        --output="${EXP}/logs/s1_%A_%a.out" \
        --error="${EXP}/logs/s1_%A_%a.err" \
        --export=ALL \
        slurm/run_vllm_stage.sbatch)

    CONCAT1=$(sbatch --parsable \
        --dependency=afterok:${JID1} \
        --job-name="v_${MEASURE:0:12}_s1c" \
        --partition=nlp --account=${ACCOUNT} \
        --ntasks=1 --cpus-per-task=2 --mem=8G --time=0:30:00 \
        --output="${EXP}/logs/s1_concat_%j.out" \
        --wrap="cat ${OUT1_BASE}_part_*.jsonl > ${COARSE_FINAL} && rm ${OUT1_BASE}_part_*.jsonl && echo \"Stage 1 done: \$(wc -l < ${COARSE_FINAL}) rows -> ${COARSE_FINAL}\"")

    echo "  Stage 1: array=${JID1}  concat=${CONCAT1}"

    # ── Stage 2: low_quality_filter (vLLM array job) ─────────────────────────
    OUT2_BASE="${EXP}/results/${MEASURE}_scores"
    SCORES_FINAL="${OUT2_BASE}.jsonl"

    export STAGE="low_quality_filter"
    export INPUT_PATH="$COARSE_FINAL"
    export OUTPUT_BASE="$OUT2_BASE"
    export EXTRA_ARGS="--concurrency_limit 64"
    # Clear EXPERIMENT_DIR so only shard 0 of Stage 1 snapshotted src/
    export EXPERIMENT_DIR=""

    JID2=$(sbatch --parsable \
        --dependency=afterok:${CONCAT1} \
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
echo "Final outputs: experiments/verify_<measure>/results/<measure>_final.jsonl"
