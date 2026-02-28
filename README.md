# socialai

Studying anthropomorphic behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat) using LLM-as-a-judge pipelines to identify when AI assistants present themselves as having human-like qualities.

## Setup

```bash
bash install_uv.sh   # if uv not installed
uv sync
```

## Pipeline Overview

The pipeline runs in three stages on the WildChat-1M dataset. Each stage corresponds to a Python file inside the measure folder (e.g., `src/measure/anthropomorphism/`):

**Stage 1 — Chit-Chat Filter** (`chit_chat.py` + `filter1.json`)
- Downloads WildChat-1M from HuggingFace, filters English, deduplicates
- LLM judge keeps only casual chit-chat conversations
- Input: HuggingFace → Output: `data/wildchat_chit_chat.jsonl`

**Stage 2 — Low-Quality / Domain Judge** (`anthropomorphic.py` or `low_quality_filter.py` + `filter2.json`)
- Scores each conversation on the measure-specific dimension (-1 / 0 / 1)
- For anthropomorphism: detects human-like identity/emotion claims
- Input: `data/wildchat_chit_chat.jsonl` → Output: `data/wildchat_scores.jsonl`

**Stage 3 — High-Quality Filter** (`high_quality_filter.py` + `filter3.json`)
- Evaluates a curated seed set against frontier models via OpenRouter API
- Input: seed CSV → Output: model responses CSV + JSONL files

## Running the Pipeline

All stages are run via the central `src/run.py` dispatcher. Always run from the **project root**.

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

### Step 4: High-quality filter (OpenRouter API)

```bash
uv run python src/run.py \
    --measure anthropomorphism \
    --stage high_quality_filter \
    --input <seed_set.csv> \
    --output experiments/03_openrouter_eval/results/raw.csv \
    --key <path/to/openrouter_api_key>
```

### SLURM scripts

| Script | Stage | Resources |
|--------|-------|-----------|
| `experiments/01_chit_chat_filter/run_download.sbatch` | Download (once) | CPU only, 40G, 4h |
| `experiments/01_chit_chat_filter/run_chit_chat.sbatch` | Stage 1 (array 0-5) | 1 GPU, 40G, 2d |
| `experiments/02_anthropomorphic_judge/run_anthropomorphic.sbatch` | Stage 2 (array 0-5) | 1 GPU, 40G, 2d |

### Direct CLI usage (without SLURM)

All stages can be run directly via `src/run.py` on any node with a running vLLM server:

```bash
# Start vLLM server on a GPU node
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.85

# Stage 1: chit-chat filter
uv run python src/run.py \
    --measure anthropomorphism \
    --stage chit_chat \
    --input_path data/wildchat_raw.jsonl \
    --output_path data/wildchat_chit_chat.jsonl \
    --prompt-version v4

# Stage 2: anthropomorphic judge
uv run python src/run.py \
    --measure anthropomorphism \
    --stage anthropomorphic \
    --input_path data/wildchat_chit_chat.jsonl \
    --output_path data/wildchat_scores.jsonl
```

## Adding a New Measure

To apply the pipeline to a new research question, create a folder under `src/measure/`:

```
src/measure/<your_measure>/
├── filter1.json          # versioned chit-chat filter prompts {"v1": "...", ...}
├── filter2.json          # {"prompt": "<your domain judge prompt>"}
├── filter3.json          # {"system_prompt": "", "models": {...}}
├── chit_chat.py          # Stage 1 judge (copy from an existing measure)
├── low_quality_filter.py # Stage 2 judge (implement for your domain)
└── high_quality_filter.py # Stage 3 OpenRouter judge (copy from an existing measure)
```

Then run:
```bash
uv run python src/run.py --measure <your_measure> --stage chit_chat --input_path ... --output_path ...
```

## Project Structure

```text
src/
  run.py             # unified entry point: --measure, --stage dispatch
  utils.py           # Judge base class, JudgeConfig, async inference
  param.py           # shared argparse definitions
  measure/
    anthropomorphism/  # filter1.json, filter2.json, filter3.json, chit_chat.py, anthropomorphic.py, high_quality_filter.py
    sycophancy/        # same structure with sycophancy-specific prompts
  mode/
    single_turn.py   # format_single_turn() utility for WildChat
    multi_turn.py    # stub for future multi-turn support
  allegro/           # shared library code
experiments/         # numbered experiment folders (sbatch scripts, results, logs)
  01_chit_chat_filter/
  02_anthropomorphic_judge/
  03_openrouter_eval/
slurm/               # generic SLURM job templates
data/                # datasets (gitignored)
tests/               # pytest tests
depreciated/         # old monolithic pipeline scripts (reference only)
```

## Run Tests

```bash
uv run python -m pytest --extra dev
```
