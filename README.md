# socialai

Studying anthropomorphic behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat) using LLM-as-a-judge pipelines to identify when AI assistants present themselves as having human-like qualities.

## Setup

```bash
bash install_uv.sh   # if uv not installed
uv sync
```

## Pipeline Overview

The full pipeline runs in two stages on the WildChat-1M dataset:

**Stage 1 — Chit-Chat Filter** (`experiments/01_chit_chat_filter/`)
- Downloads WildChat-1M from HuggingFace, filters English, deduplicates
- LLM judge keeps only casual chit-chat conversations
- Input: HuggingFace → Output: `data/wildchat_chit_chat.jsonl`

**Stage 2 — Anthropomorphic Judge** (`experiments/02_anthropomorphic_judge/`)
- Scores each conversation for anthropomorphic AI behavior (-1 / 0 / 1)
- Input: `data/wildchat_chit_chat.jsonl` → Output: `data/wildchat_scores.jsonl`

## Running the Pipeline

### Step 1: Download & preprocess WildChat (once, no GPU)

```bash
sbatch experiments/01_chit_chat_filter/run_download.sbatch
# Check: wc -l data/wildchat_raw.jsonl
```

### Step 2: Chit-chat filter (6-shard GPU array)

```bash
sbatch experiments/01_chit_chat_filter/run_chit_chat.sbatch
# Monitor: squeue --me
# After all 6 shards complete:
cat data/wildchat_chit_chat_part_*.jsonl > data/wildchat_chit_chat.jsonl
```

### Step 3: Anthropomorphic judge (6-shard GPU array)

```bash
sbatch experiments/02_anthropomorphic_judge/run_anthropomorphic.sbatch
# After all 6 shards complete:
cat data/wildchat_scores_part_*.jsonl > data/wildchat_scores.jsonl
```

### SLURM scripts

Each pipeline stage has its own sbatch script that handles vLLM startup, port allocation, and cleanup:

| Script | Stage | Resources |
|--------|-------|-----------|
| `experiments/01_chit_chat_filter/run_download.sbatch` | Download (once) | CPU only, 40G, 4h |
| `experiments/01_chit_chat_filter/run_chit_chat.sbatch` | Stage 1 (array 0-5) | 1 GPU, 40G, 2d |
| `experiments/02_anthropomorphic_judge/run_anthropomorphic.sbatch` | Stage 2 (array 0-5) | 1 GPU, 40G, 2d |

All scripts must be submitted from the **project root**:
```bash
cd /project2/robinjia_875/ehuang97/socialai
sbatch experiments/01_chit_chat_filter/run_chit_chat.sbatch
```

### Judge CLI (direct use)

The judges can also be run directly without SLURM (requires a running vLLM server):

```bash
# Start vLLM server on a GPU node
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.85

# Run chit-chat filter
uv run python experiments/01_chit_chat_filter/run.py \
    --input_path data/wildchat_raw.jsonl \
    --output_path data/wildchat_chit_chat.jsonl \
    --prompt-version v4

# Run anthropomorphic judge
uv run python experiments/02_anthropomorphic_judge/run.py \
    --input_path data/wildchat_chit_chat.jsonl \
    --output_path data/wildchat_scores.jsonl
```

## Project Structure

```text
src/pipeline/      # LLM-as-a-judge pipeline (Judge base class + implementations)
src/allegro/       # Shared library code
experiments/       # Numbered experiment folders (each with run.py, README, results/)
  01_chit_chat_filter/      # Stage 1: filter WildChat for chit-chat
  02_anthropomorphic_judge/ # Stage 2: score anthropomorphic behavior
  03_openrouter_eval/       # Evaluate seed prompts via OpenRouter API
slurm/             # Generic SLURM job templates
data/              # Datasets (gitignored)
tests/             # pytest tests
depreciated/       # Old monolithic pipeline scripts (reference only)
```

## Run Tests

```bash
uv run pytest
```
