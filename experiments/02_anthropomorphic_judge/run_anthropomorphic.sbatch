#!/bin/bash
#SBATCH --job-name=wildchat_anthro
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/02_anthropomorphic_judge/logs/anthro_%A_%a.out
#SBATCH --error=experiments/02_anthropomorphic_judge/logs/anthro_%A_%a.err

# Usage (from project root):
#   sbatch --array=0-5 experiments/02_anthropomorphic_judge/run_anthropomorphic.sbatch
# Prerequisites: data/wildchat_chit_chat.jsonl must exist
#   (output of experiment 03 after concatenating shards)

mkdir -p experiments/02_anthropomorphic_judge/logs

module purge
module load gcc/13.3.0
module load cuda/12.6.3

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR

export HF_HOME=/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface
# Per-task TMPDIR prevents torch inductor cache collisions across array tasks on shared NFS
export TMPDIR=/project2/robinjia_875/wangzhu/eric_huang/.cache/tmp/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR

# Ask the OS for a free port — guaranteed no conflict with other jobs on the same node
export VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using vLLM port: $VLLM_PORT"

# Start vLLM server in background
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 1 \
    --port $VLLM_PORT \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --disable-log-requests &

SERVER_PID=$!

# Wait for server to be ready (up to 5 minutes)
echo "Waiting for vLLM server on port $VLLM_PORT..."
timeout 300 bash -c "until curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; do sleep 5; done"
if [ $? -ne 0 ]; then
    echo "ERROR: vLLM server failed to start within 5 minutes."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "vLLM server ready."

# SLURM sets SLURM_ARRAY_TASK_COUNT=6 automatically when --array=0-5 is used
uv run python experiments/02_anthropomorphic_judge/run.py \
    --input_path data/wildchat_chit_chat.jsonl \
    --output_path data/wildchat_scores_part_${SLURM_ARRAY_TASK_ID}.jsonl \
    --concurrency_limit 64

# Cleanup
echo "Job done. Killing vLLM server (PID $SERVER_PID)."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "All done."
