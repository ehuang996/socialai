#!/bin/bash
#SBATCH --job-name=wildchat_pre
#SBATCH --partition=nlp_hiprio
#SBATCH --array=0-5                 
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=preprocess_logs/pre_%A_%a.out
#SBATCH --error=preprocess_logs/pre_%A_%a.err

mkdir -p preprocess_logs

# Environment
module purge
module load gcc/13.3.0
module load cuda/12.6.3
source activate /project2/robinjia_875/wangzhu/eric_huang/conda_envs/conda_envs/socialai3

export HF_HOME=/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface
export TMPDIR=/project2/robinjia_875/wangzhu/eric_huang/.cache/tmp
mkdir -p $TMPDIR

# Dynamic port assignment (8000-9000)
BASE_PORT=8000
OFFSET=$((SLURM_ARRAY_TASK_ID * 10))
RANDOM_ADD=$(shuf -i 1-9 -n 1)
export VLLM_PORT=$((BASE_PORT + OFFSET + RANDOM_ADD))

# Start vLLM Server in Background
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 1 \
    --port $VLLM_PORT \
    --max-model-len 16384 \
    --trust-remote-code \
    --disable-log-requests &

# Capture PID of vLLM
SERVER_PID=$!

# Wait for Server
timeout 300 bash -c 'until curl -s localhost:${VLLM_PORT}/v1/models > /dev/null; do sleep 5; done'

# Run Preprocessing Script
export SLURM_ARRAY_TASK_COUNT=6
python preprocess_wildchat.py

# Cleanup
kill $SERVER_PID