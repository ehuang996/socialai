# socialai

Studying social behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat-1M) using LLM-as-a-judge pipelines to identify when AI assistants exhibit specific behaviors (e.g., anthropomorphism, sycophancy).

## Setup

```bash
bash install_uv.sh   # if uv is not yet installed
uv sync
```

Requires Python ≥ 3.11 (managed by uv). The virtual env is at `.venv/`.

## Pipeline Overview

Each measure runs three sequential stages on the WildChat-1M dataset:

| Stage | Name | Judge | Input → Output |
|-------|------|-------|----------------|
| 1 | `coarse_filter` | vLLM (local) | raw JSONL → chit-chat conversations |
| 2 | `low_quality_filter` | vLLM (local) | chit-chat JSONL → scored conversations |
| 3 | `high_quality_filter` | OpenRouter API | seed CSV → frontier model responses |

**Stage 1 — Coarse Filter** (`filter1.json`)
Keeps only casual chit-chat conversations from WildChat. Uses a versioned prompt from `filter1.json` (default `v4`). Passes only the first user message to the judge; outputs rows where `keep=true`.

**Stage 2 — Low-Quality Filter** (`filter2.json`)
Scores each conversation on the measure-specific dimension. Passes the first user + assistant turn to the judge; outputs a score and reasoning per row.

**Stage 3 — High-Quality Filter** (`filter3.json`)
Evaluates a curated seed CSV against multiple frontier models via the OpenRouter API. Outputs per-model responses as a CSV and individual JSONL files. Supports checkpointing (safe to rerun).

## Running the Pipeline

All stages are submitted via `pipeline.sh` from the **project root**. Each run auto-creates a new `experiments/<NN>_<stage>/` folder with logs, results, and a source snapshot.

### Step 0 — Download WildChat (once, CPU job)

```bash
sbatch slurm/run_download.sbatch
# Output: data/wildchat_raw.jsonl
```

### Step 1 — Coarse filter

```bash
bash pipeline.sh --measure anthropomorphism --stage coarse_filter
# Default input: data/wildchat_raw.jsonl
# Output: experiments/<NN>_coarse_filter/results/anthropomorphism_coarse.jsonl
```

### Step 2 — Low-quality filter

```bash
bash pipeline.sh --measure anthropomorphism --stage low_quality_filter \
    --input experiments/<NN>_coarse_filter/results/anthropomorphism_coarse.jsonl
# Output: experiments/<NN>_low_quality_filter/results/anthropomorphism_scores.jsonl
```

### Step 3 — High-quality filter (OpenRouter API)

```bash
bash pipeline.sh --measure anthropomorphism --stage high_quality_filter \
    --input <seed_set.csv> \
    --key <path/to/openrouter_api_key>
# Output: experiments/<NN>_high_quality_filter/results/anthropomorphism_high_quality.csv
```

### Pipeline options

```
bash pipeline.sh --measure <name> --stage <stage> [options]

Options:
  --shards N    Number of SLURM array shards for vLLM stages (default: 6)
  --input  path Input file (required for stages 2 & 3; defaults to wildchat_raw.jsonl for stage 1)
  --key    path OpenRouter API key file (required for stage 3)
  --exclude N   Comma-separated SLURM node exclusions (optional)
```

### Running a stage directly (without SLURM)

On a GPU node with a running vLLM server:

```bash
# Start the vLLM server
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.85

# Stage 1
uv run python src/run.py \
    --measure anthropomorphism --stage coarse_filter \
    --input_path data/wildchat_raw.jsonl \
    --output_path data/anthropomorphism_coarse.jsonl \
    --prompt-version v4

# Stage 2
uv run python src/run.py \
    --measure anthropomorphism --stage low_quality_filter \
    --input_path data/anthropomorphism_coarse.jsonl \
    --output_path data/anthropomorphism_scores.jsonl

# Stage 3
uv run python src/run.py \
    --measure anthropomorphism --stage high_quality_filter \
    --input <seed_set.csv> \
    --output results/raw.csv \
    --key <path/to/openrouter_api_key>
```

The `VLLM_PORT` env var controls which port the judges connect to (default `8001`).

## Adding a New Measure

Create a folder under `src/measure/` with three JSON config files — no Python files needed:

```
src/measure/<your_measure>/
├── filter1.json   # {"v1": "<system prompt>", "v4": "<system prompt>", ...}
├── filter2.json   # {"prompt": "<system prompt for domain judge>"}
└── filter3.json   # {"system_prompt": "...", "models": {"col_name": "openrouter/model-id", ...}}
```

The base implementations in `src/measure/base/` handle everything automatically. Then run:

```bash
bash pipeline.sh --measure <your_measure> --stage coarse_filter
```

## Project Structure

```
src/
  run.py             # unified entry point: --measure / --stage dispatch
  utils.py           # Judge base class, JudgeConfig, async vLLM inference
  param.py           # shared argparse definitions (add_judge_args)
  measure/
    base/            # generic stage implementations (coarse_filter, low_quality_filter, high_quality_filter)
    anthropomorphism/  # filter1.json, filter2.json, filter3.json
    sycophancy/        # filter1.json, filter2.json, filter3.json
  mode/
    single_turn.py   # format_single_turn() for WildChat rows
    multi_turn.py    # stub for future multi-turn support
  allegro/           # shared library code
scripts/
  download.py        # downloads WildChat-1M from HuggingFace
slurm/
  run_download.sbatch    # CPU job for data download
  run_vllm_stage.sbatch  # GPU array job template (used by pipeline.sh for stages 1 & 2)
experiments/         # auto-created by pipeline.sh; one numbered folder per stage run
data/                # datasets (gitignored)
tests/               # pytest tests
depreciated/         # old monolithic scripts (reference only, do not edit)
```

## Tests

```bash
uv run python -m pytest --extra dev
```

## Cluster Notes (USC CARC)

- **Partitions:** `nlp_hiprio` (no preemption) and `nlp` (preemptable)
- **Account:** `robinjia_875`
- **Default resources per job:** 8 CPUs, 1 GPU, 40G RAM
- **Required modules:** `gcc/13.3.0` and `cuda/12.6.3` (already in sbatch templates)
- **vLLM model:** `Qwen/Qwen3-VL-8B-Instruct` (built with torch+cu126)
- Each SLURM array task gets an OS-assigned free port (`VLLM_PORT`) and a per-task `TMPDIR` to avoid cache collisions on shared NFS
- Judges skip rows already in the output file (`conversation_hash` key) — safe to resubmit failed shards
