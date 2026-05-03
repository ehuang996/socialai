#!/bin/bash
# Stage 7.1 local-HF Qwen response generation.
# Do not submit directly unless MODEL_COL and HF_MODEL are exported.

#SBATCH --partition=nlp_hiprio,nlp
#SBATCH --account=robinjia_875
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

: "${MODEL_COL:?Set MODEL_COL to qwen3_1_7b or qwen3_4b}"
: "${HF_MODEL:?Set HF_MODEL to Qwen/Qwen3-1.7B or Qwen/Qwen3-4B}"

INPUT_PATH=${INPUT_PATH:-data/final_dataset.jsonl}
OUTPUT_BASE=${OUTPUT_BASE:-data/eval_responses_local_qwen}
CONCURRENCY=${CONCURRENCY:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-900}

module purge
module load gcc/13.3.0
module load cuda/12.6.3

export PATH="$HOME/.local/bin:$PATH"

cd "$SLURM_SUBMIT_DIR"

# Keep Hugging Face and temporary files off the nearly-full project filesystem.
export HF_HOME=${HF_HOME:-/home1/wangzhu/.cache/huggingface}
export TMPDIR=${TMPDIR:-/project2/jessetho_1732/wangzhu/socialai/cache/tmp/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}}
mkdir -p "$TMPDIR"

export VLLM_PORT
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using vLLM port: $VLLM_PORT"
echo "MODEL_COL: $MODEL_COL"
echo "HF_MODEL: $HF_MODEL"

uv run --no-sync vllm serve "$HF_MODEL" \
    --served-model-name "$MODEL_COL" \
    --tensor-parallel-size 1 \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --disable-log-requests &

SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Killing vLLM server (PID $SERVER_PID)."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM server on port $VLLM_PORT..."
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
until curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "ERROR: vLLM server failed to start within ${VLLM_STARTUP_TIMEOUT}s."
        exit 1
    fi
    sleep 5
done
echo "vLLM server ready."

OUTPUT_PATH="${OUTPUT_BASE}_${MODEL_COL}_part_${SLURM_ARRAY_TASK_ID:-0}.jsonl"

uv run --no-sync python -m src.evaluation.generate_local_responses \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --model_col "$MODEL_COL" \
    --served_model_name "$MODEL_COL" \
    --api_base "http://localhost:${VLLM_PORT}/v1" \
    --concurrency "$CONCURRENCY"

echo "Local generation done: $OUTPUT_PATH"
