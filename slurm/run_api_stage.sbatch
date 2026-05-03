#!/bin/bash
# Generic OpenRouter-API pipeline stage template (Stages 3 and 4).
# Do not submit directly — use pipeline.sh which sets all required env vars.
#
# Required env vars (set by pipeline.sh via --export=ALL):
#   MEASURE        — measure name (e.g., 2C_flattery_tone)
#   STAGE          — high_quality_filter | final_filter
#   INPUT_PATH     — input JSONL path (Stage 2 / Stage 3 output)
#   OUTPUT_BASE    — output path prefix (e.g., data/2C_flattery_tone_high_quality)
#                    actual output = ${OUTPUT_BASE}_part_${SLURM_ARRAY_TASK_ID}.jsonl
#   KEY            — path to OpenRouter API key file
#   EXPERIMENT_DIR — if non-empty, shard 0 snapshots src/ into <EXPERIMENT_DIR>/src/
#   EXTRA_ARGS     — additional flags (optional)
#
# --job-name, --array, --output, --error, --dependency are set by pipeline.sh
# on the command line.

#SBATCH --partition=nlp,nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# 4 hours was too short — 2A's Stage 3 hit TIMEOUT at 4:00:24 with 85% done
# (the new "personal information" rubric has much higher keep rate than the
# old "personal details", so Stage 3 input was ~404K rows). 12 h gives headroom.
#SBATCH --time=12:00:00

module purge
module load gcc/13.3.0

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR

uv run --no-sync python src/filter/run.py \
    --measure ${MEASURE} \
    --stage ${STAGE} \
    ${EXPERIMENT_DIR:+--experiment_dir ${EXPERIMENT_DIR}} \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_BASE}_part_${SLURM_ARRAY_TASK_ID}.jsonl \
    --key ${KEY} \
    ${EXTRA_ARGS}
