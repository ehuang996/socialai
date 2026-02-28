# Allegro Project — Claude Instructions

## Environment
- USC CARC SLURM cluster
- Package manager: uv (NOT conda, NOT pip directly)
- Python: managed by uv, pinned in pyproject.toml (>=3.11)
- Virtual env: `.venv/` at project root (managed by uv)

## Cluster Info
- **Partitions:** `nlp_hiprio` (high priority, no preemption) and `nlp` (preemptable)
- **Account:** `robinjia_875`
- **GPUs:** ~60 A6000s, 4 A100s
- **Default resources per job:** 8 CPUs, 1 GPU, 40G RAM
- **Max walltime:** typically 2 days default in templates

## How to Run Things

### Install dependencies
```
uv sync
```

### Run a pipeline stage directly (on a compute node with GPU)
```bash
uv run python src/run.py \
    --measure <measure> \
    --stage <stage> \
    --input_path <input.jsonl> \
    --output_path <output.jsonl> \
    [additional stage-specific args]
```

`--measure` is the folder name under `src/measure/` (e.g., `anthropomorphism`, `sycophancy`).
`--stage` is the Python file name within that folder (e.g., `coarse_filter`, `low_quality_filter`, `high_quality_filter`).

The optional `--experiment_dir` flag snapshots `src/` into `<experiment_dir>/src/` before running (for version control).

### Submit a SLURM job
Run the full pipeline with a single command:
```bash
bash pipeline.sh --measure <measure> --stage <stage> [--shards 6] [--input <path>]
```
This submits the SLURM job(s) for one stage and creates `experiments/<NN>_<stage>/` for logs and snapshots.

Generic SLURM templates for simple single-script jobs are in `slurm/`:
```bash
sbatch slurm/run_gpu.sbatch <script.py>        # single job, high priority
sbatch slurm/run_preempt.sbatch <script.py>    # single job, preemptable
```

Array jobs set `SLURM_ARRAY_TASK_ID` env var; experiments can read it via `os.environ.get("SLURM_ARRAY_TASK_ID")` (returns `None` for non-array jobs).

### Module loads needed (already in sbatch templates)
```
module purge
module load gcc/13.3.0
module load cuda/12.6.3   # must be 12.6.3 — .venv was built with torch+cu126
```

### Start the vLLM server (on a compute node with GPU)
```
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.85
```
Stages 1 and 2 (`coarse_filter`, `low_quality_filter`) default to `http://localhost:8001/v1` via the `VLLM_PORT` env var (default `8001`).

### Run the WildChat pipeline (anthropomorphism measure)

Always submit from the project root.

**Stage 0 — Download WildChat (once, CPU job):**
```bash
sbatch slurm/run_download.sbatch
```

**Stage 1 — Coarse filter:**
```bash
bash pipeline.sh --measure anthropomorphism --stage coarse_filter
```
Creates `experiments/<NN>_coarse_filter/` with output in its `results/` subfolder.

**Stage 2 — Low-quality filter:**
```bash
bash pipeline.sh --measure anthropomorphism --stage low_quality_filter \
    --input experiments/<NN>_coarse_filter/results/anthropomorphism_coarse.jsonl
```

**Stage 3 — High-Quality Filter (OpenRouter API):**
```bash
bash pipeline.sh --measure anthropomorphism --stage high_quality_filter \
    --input <seed_set.csv> \
    --key <path/to/openrouter_api_key>
```

**Key sbatch design patterns (used in all pipeline scripts):**
- Port: `VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")` — OS-assigned free port, avoids conflicts
- TMPDIR: `job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}` — per-task to avoid torch inductor cache collisions on shared NFS
- `SLURM_ARRAY_TASK_COUNT` is set automatically by SLURM when using `--array=0-5` (no manual export needed)
- Checkpoint/resumption: judges skip rows already in the output file (`conversation_hash` key); safe to resubmit failed shards
- Snapshot: `--experiment_dir` in `src/run.py` copies `src/` into the experiment folder for reproducibility

### Run tests
```
uv run python -m pytest --extra dev
```

## Project Layout
- `src/run.py` — unified pipeline entry point; dispatches to `--measure` / `--stage`
- `src/utils.py` — `Judge` base class + `JudgeConfig`; async inference, sharding, JSONL I/O
- `src/param.py` — shared argparse definitions (`add_judge_args`)
- `src/measure/` — one folder per measurement type; each is self-contained:
  - `<measure>/filter1.json` — coarse filter prompts (Stage 1)
  - `<measure>/filter2.json` — low-quality/domain-specific judge prompt (Stage 2)
  - `<measure>/filter3.json` — OpenRouter model config + system prompt (Stage 3)
  - Stage `.py` files (`coarse_filter.py`, `low_quality_filter.py`, `high_quality_filter.py`) are provided by `src/measure/base/` and run automatically — **no need to add them per measure**
  - Current measures: `anthropomorphism/`, `sycophancy/`
- `src/mode/` — conversation formatting utilities:
  - `single_turn.py` — `format_single_turn(row)` for WildChat single-turn analysis
  - `multi_turn.py` — stub for future multi-turn support
- `src/allegro/` — shared library code (reusable modules)
- `scripts/` — one-off Python scripts (e.g., `download.py` for WildChat)
- `experiments/` — auto-created by `pipeline.sh`; one folder per stage run (`<NN>_<stage>/`)
  - Each folder contains `figures/`, `logs/`, `results/`, and an `src/` snapshot (shard 0 only)
- `data/` — datasets (gitignored)
- `slurm/` — generic SLURM job templates (for simple single-script jobs)
- `tests/` — pytest tests
- `depreciated/` — old monolithic pipeline scripts (reference only, do not edit)

## Conventions
- **Adding a new measure:** create `src/measure/<name>/` with only `filter1.json`, `filter2.json`, and `filter3.json`. Then run `bash pipeline.sh --measure <name> --stage <stage>` — the base implementations handle everything automatically.
- **Experiments:** `experiments/` starts empty and is populated automatically by `pipeline.sh`. Each run creates a new `<NN>_<stage>/` folder. Do not manually add files there.
- Pipeline stages with vLLM use `slurm/run_vllm_stage.sbatch` (parameterized via env vars set by `pipeline.sh`).
- The top-level `experiments/README.md` should only contain brief descriptions of each experiment.
- Reusable code goes in `src/allegro/` with argparse for flexibility.
- Tracking: wandb (disable with `WANDB_MODE=disabled`)
