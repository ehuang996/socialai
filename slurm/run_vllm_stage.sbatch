#!/bin/bash
# Generic vLLM pipeline stage template.
# Do not submit directly — use pipeline.sh which sets all required env vars.
#
# Required env vars (set by pipeline.sh via --export=ALL):
#   MEASURE        — measure name (e.g., anthropomorphism)
#   STAGE          — stage name (e.g., coarse_filter, low_quality_filter)
#   INPUT_PATH     — input JSONL path
#   OUTPUT_BASE    — output path prefix (e.g., data/anthropomorphism_coarse)
#                    actual output = ${OUTPUT_BASE}_part_${SLURM_ARRAY_TASK_ID}.jsonl
#   EXTRA_ARGS     — additional flags passed to src/filter/run.py (space-separated)
#   EXPERIMENT_DIR — if non-empty, shard 0 snapshots src/ into <EXPERIMENT_DIR>/src/
#
# --job-name, --array, --output, --error are set by pipeline.sh on the command line.

#SBATCH --partition=nlp_hiprio,nlp
#SBATCH --account=robinjia_875
#SBATCH --requeue
# nlp_hiprio caps user at 8 GPUs (QOS MaxTRESPU). With 4 measures × 8 shards = 32
# concurrent GPU tasks, SLURM falls back to nlp (60-GPU/user, preemptable) for the
# overflow. --requeue: SLURM auto-resubmits preempted tasks; Judge.load_completed_ids()
# resumes from the per-shard partial file on each restart.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

module purge
module load gcc/13.3.0
module load cuda/12.6.3

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR

# HF cache on /home1 (220T fs, 103T free) — robinjia_875 project is 98% full and
# even tiny vLLM startup writes (refs/main, .no_exist/*) intermittently fail with
# OSError [Errno 122] there. /home1 has no group-quota issues. The Qwen3-VL-8B
# model is already cached at /home1/wangzhu/.cache/huggingface (verified 17 GB).
export HF_HOME=/home1/wangzhu/.cache/huggingface
# TMPDIR on the NEW project (jessetho_1732) — avoid the old robinjia_875 disk that's at 98%.
export TMPDIR=/project2/jessetho_1732/wangzhu/socialai/cache/tmp/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR

export VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using vLLM port: $VLLM_PORT"

uv run --no-sync vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 1 \
    --port $VLLM_PORT \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --disable-log-requests &

SERVER_PID=$!

echo "Waiting for vLLM server on port $VLLM_PORT..."
timeout 300 bash -c "until curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; do sleep 5; done"
if [ $? -ne 0 ]; then
    echo "ERROR: vLLM server failed to start within 5 minutes."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "vLLM server ready."

uv run --no-sync python src/filter/run.py \
    --measure ${MEASURE} \
    --stage ${STAGE} \
    ${EXPERIMENT_DIR:+--experiment_dir ${EXPERIMENT_DIR}} \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_BASE}_part_${SLURM_ARRAY_TASK_ID}.jsonl \
    ${EXTRA_ARGS}

echo "Job done. Killing vLLM server (PID $SERVER_PID)."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "All done."
