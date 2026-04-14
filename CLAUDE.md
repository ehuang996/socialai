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

### Run a filter stage directly (on a compute node with GPU)
```bash
uv run python src/filter/run.py \
    --measure <measure> \
    --stage <stage> \
    --input_path <input.jsonl> \
    --output_path <output.jsonl> \
    [additional stage-specific args]
```

`--measure` is the folder name under `src/filter/measure/` (e.g., `1B_human_disfluencies`, `2C_sycophancy`).
`--stage` is the Python file name within that folder (e.g., `coarse_filter`, `low_quality_filter`, `high_quality_filter`, `final_filter`).

The optional `--experiment_dir` flag snapshots `src/filter/` into `<experiment_dir>/src/` before running (for version control).

### Submit a SLURM job
Run filter stages via pipeline.sh:
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
Stages 1 and 2 (`coarse_filter`, `low_quality_filter`) use a local vLLM server via the `VLLM_PORT` env var (default `8001`).

### Run tests
```
uv run python -m pytest --extra dev
```

## Full Pipeline

The pipeline has three phases: **filtering** (Stages 0-4, run per measure), **evaluation** (Stages 5-6, run once across all measures), and **analysis** (Stages 7-8, judge and analyze model responses).

Always submit from the project root.

### Phase 1: Filtering (per measure, via pipeline.sh)

Run Stages 1-4 for each measure (e.g., `1B_human_disfluencies`, `2C_sycophancy`, etc.).

**Stage 0 — Download WildChat (once, CPU job):**
```bash
sbatch slurm/run_download.sbatch
```
Downloads `allenai/WildChat-4.8M`, filters English, deduplicates → `data/wildchat_raw.jsonl`.

**Stage 1 — Coarse Filter (vLLM, Qwen3-VL-8B):**
```bash
bash pipeline.sh --measure 1B_human_disfluencies --stage coarse_filter
```
Keyword/pattern-based filtering using local vLLM. Uses `filter1.json` prompts.
Creates `experiments/<NN>_coarse_filter/` with output in its `results/` subfolder.

**Stage 2 — Low-Quality Filter (vLLM, Qwen3-VL-8B):**
```bash
bash pipeline.sh --measure 1B_human_disfluencies --stage low_quality_filter \
    --input experiments/<NN>_coarse_filter/results/1B_human_disfluencies_coarse.jsonl
```
Domain-specific judge using local vLLM. Uses `filter2.json` prompts.

**Stage 3 — High-Quality Filter (OpenRouter API, GPT-4o-mini):**
```bash
bash pipeline.sh --measure 1B_human_disfluencies --stage high_quality_filter \
    --input experiments/<NN>_low_quality_filter/results/1B_human_disfluencies_scores.jsonl \
    --key <path/to/openrouter_api_key>
```
Dual-check (chitchat + category) with GPT-4o-mini. Uses `filter3.json` config.
Output contains `chitchat_keep` and `category_keep` fields.

**Stage 4 — Final Filter (OpenRouter API, Claude Opus 4.6):**
```bash
bash pipeline.sh --measure 1B_human_disfluencies --stage final_filter \
    --input experiments/<NN>_high_quality_filter/results/1B_human_disfluencies_high_quality.jsonl \
    --key <path/to/openrouter_api_key>
```
Re-evaluates only the intersection rows (both `chitchat_keep=true` AND `category_keep=true` from Stage 3) with Claude Opus 4.6. Uses `filter4.json` config.

### Phase 2: Evaluation (once, across all measures)

After running Stages 1-4 for all measures, collect and evaluate.

**Stage 5 — Data Preprocessing (CPU + OpenRouter API):**
```bash
uv run python src/data_preprocessing/data_preprocessing.py \
    --output data/final.jsonl \
    --key <path/to/openrouter_api_key>
```
Three phases in one script:
1. **Collect & deduplicate** — reads all `experiments/<NN>_final_filter/results/` directories, keeps rows where Opus returned both `chitchat_keep=true` AND `category_keep=true`, and deduplicates by `user_input`. Conversations appearing in multiple measures get a single row with `measure` as a list (e.g., `["1B_human_disfluencies", "2C_sycophancy"]`). **Global chitchat veto:** if a `user_input` has `chitchat_keep=false` from *any* measure's judge, it is dropped from the entire output (even if other measures returned both keeps true), since `chitchat_keep` is a measure-independent property of the conversation. Errors are skipped, not treated as vetoes. See [misc/design.md](misc/design.md) for details.
2. **Split single/multi-turn** — uses Claude Opus 4.6 via OpenRouter to classify each `user_input` and write `single_turn_final.jsonl` (437 rows pre-dedup, 408 post-semantic-dedup), `multi_turn_final.jsonl` (518 rows), and a `split_report_*.json`. Resumable via the report file. Use `--no_split` to skip this phase.
3. **Data cleaning** — semantic dedup of the single-turn file only, using `sentence-transformers/all-MiniLM-L6-v2` + cosine similarity (threshold `0.85`, tunable via `--dedupe_threshold`). Dropped rows are logged to `dedup_report_*.json`. Multi-turn is left untouched. Then every row in all three output files is tagged with `synthetic: false` and `language: "English"` to prepare the schema for later synthetic-data mixing. Adds `sentence-transformers` as a dependency (installed via `uv sync`).

**Stage 6 — Generate Model Responses (OpenRouter API via DSPy):**
```bash
uv run python src/evaluation/stage6_generate_responses/generate_responses.py \
    --input data/final.jsonl \
    --output data/model_responses.jsonl \
    --key <path/to/openrouter_api_key> \
    --model_set 1
```
Sends each `user_input` to multiple models via OpenRouter and records their responses.
Uses DSPy for caching — responses are cached to disk so re-running with the same input/model/temperature returns identical results (reproducibility). Use `--rollout_id N` to force fresh generations while still caching the new results.
Two model sets available: `--model_set 1` (original 4 models) and `--model_set 2` (10 newer models).
Use `--models gpt5_4,claude_opus` to run a subset. Supports resumption — re-running skips already-completed rows.

### Phase 3: Analysis (judge and analyze model responses)

**Stage 7.1 — LLM-as-Judge: Single-Turn (OpenRouter API, Opus 4.6):**
```bash
uv run python src/evaluation/stage7.1_judge_single_turn/stage7_1_evaluate_single_turn.py \
    --key <path/to/openrouter_api_key>
```
Evaluates each of the 14 model responses against the category-specific rubric (`filter2.json`) for every measure in the row's `measure` list. Judge: Opus 4.6 via OpenRouter.
Input: `data/single_turn_model_responses.jsonl` (408 rows, 472 measure-labels × 14 models = 6,608 API calls).
Output: `data/stage7_1_eval_results.jsonl` — one row per (input, measure, model) triple with `judge_output: {reasoning, keep}`.
Sorting: `sort_eval_results.py` sorts by input → measure → model family (OpenAI → Gemini → Claude → Grok, newest first).

**Stage 7.2 — LLM-as-Judge: Multi-Turn (placeholder):**
Directory: `src/evaluation/stage7.2_judge_multi_turn/` — reserved for multi-turn evaluation.

**Stage 8.1 — Analysis of Single-Turn Results:**
Directory: `src/evaluation/stage8.1_analyze_single_turn/` — analysis of Stage 7.1 judge results.

**Key sbatch design patterns (used in all pipeline scripts):**
- Port: `VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")` — OS-assigned free port, avoids conflicts
- TMPDIR: `job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}` — per-task to avoid torch inductor cache collisions on shared NFS
- `SLURM_ARRAY_TASK_COUNT` is set automatically by SLURM when using `--array=0-5` (no manual export needed)
- Checkpoint/resumption: filter stages skip rows already in the output file; safe to resubmit failed shards
- Snapshot: `--experiment_dir` in `src/filter/run.py` copies `src/filter/` into the experiment folder for reproducibility

## Project Layout
- `src/filter/` — filtering pipeline (Stages 1-4):
  - `run.py` — unified pipeline entry point; dispatches to `--measure` / `--stage`
  - `utils.py` — `Judge` base class + `JudgeConfig`; async inference, sharding, JSONL I/O
  - `param.py` — shared argparse definitions (`add_judge_args`)
  - `measure/` — one folder per measurement type; each is self-contained:
    - `<measure>/filter1.json` — coarse filter prompts (Stage 1)
    - `<measure>/filter2.json` — low-quality/domain-specific judge prompt (Stage 2)
    - `<measure>/filter3.json` — GPT-4o-mini dual-check config + system prompt (Stage 3)
    - `<measure>/filter4.json` — Claude Opus 4.6 config + system prompt (Stage 4)
    - Stage `.py` files (`coarse_filter.py`, `low_quality_filter.py`, `high_quality_filter.py`, `final_filter.py`) are provided by `measure/base/` and run automatically — **no need to add them per measure**
    - Current measures: `1B_human_disfluencies/`, `1C_identity_transparency/`, `2A_fabricated_personal_details/`, `2B_explicit_emotions/`, `2B_implicit_emotions/`, `2B_romantic_bonding/`, `2C_sycophancy/`, `2D_human_relationship_encouragement/`, `3A_engagement_hooks/`
  - `mode/` — conversation formatting utilities:
    - `single_turn.py` — `format_single_turn(row)` for WildChat single-turn analysis
    - `multi_turn.py` — stub for future multi-turn support
- `src/data_preprocessing/` — Stage 5: data preprocessing (collect, dedupe, split)
  - `data_preprocessing.py` — Phase 1: collect both-KEEP rows from final_filter experiments, dedupe by `user_input`, merge measures into a list. Phase 2: classify each row as single-turn vs multi-turn via Opus 4.6 and write split files + report. `--no_split` skips Phase 2.
  - `intersect.py` — standalone utility to find intersection of `chitchat_keep` + `category_keep` from Stage 3 output
- `src/evaluation/` — post-filter evaluation and analysis pipeline (Stages 6-8):
  - `stage6_generate_responses/` — Stage 6: generate model responses
    - `generate_responses.py` — send collected conversations to multiple models via OpenRouter; uses DSPy for disk caching (reproducible re-runs)
  - `stage6.1_generate_single_turn/` — Stage 6.1: single-turn variant of generate_responses
  - `stage6.2_generate_multi_turn/` — Stage 6.2: multi-turn variant (placeholder)
  - `stage7.1_judge_single_turn/` — Stage 7.1: LLM-as-judge evaluation (single-turn)
    - `stage7_1_evaluate_single_turn.py` — evaluate 14 model responses per measure with Opus 4.6 judge
    - `sort_eval_results.py` — sort results by input → measure → model family/recency
  - `stage7.2_judge_multi_turn/` — Stage 7.2: LLM-as-judge evaluation (multi-turn, placeholder)
  - `stage8.1_analyze_single_turn/` — Stage 8.1: analysis of single-turn judge results
- `scripts/` — one-off Python scripts (e.g., `download.py` for WildChat-4.8M)
- `experiments/` — auto-created by `pipeline.sh`; one folder per stage run (`<NN>_<stage>/`)
  - Each folder contains `figures/`, `logs/`, `results/`, and an `src/` snapshot (shard 0 only)
- `data/` — datasets (gitignored)
  - `wildchat_raw.jsonl` — downloaded WildChat data (Stage 0)
  - `final.jsonl` — deduplicated final dataset (Stage 5 output, 955 rows)
  - `model_responses.jsonl` — model responses (Stage 6 output, 955 rows × 14 models)
  - `single_turn_final.jsonl` — single-turn subset (408 rows, 472 measure-labels)
  - `multi_turn_final.jsonl` — multi-turn subset (518 rows, 645 measure-labels)
  - `stage7_1_eval_results.jsonl` — LLM-as-judge results for single-turn (6,608 rows: 472 × 14)
- `slurm/` — generic SLURM job templates (for simple single-script jobs)
- `tests/` — pytest tests
- `depreciated/` — old monolithic pipeline scripts (reference only, do not edit)

## Conventions
- **Adding a new measure:** create `src/filter/measure/<name>/` with `filter1.json`, `filter2.json`, `filter3.json`, and `filter4.json`. Then run `bash pipeline.sh --measure <name> --stage <stage>` — the base implementations handle everything automatically.
- **Experiments:** `experiments/` starts empty and is populated automatically by `pipeline.sh`. Each run creates a new `<NN>_<stage>/` folder. Do not manually add files there.
- Pipeline stages with vLLM use `slurm/run_vllm_stage.sbatch` (parameterized via env vars set by `pipeline.sh`).
- The top-level `experiments/README.md` should only contain brief descriptions of each experiment.
- Reusable code goes in `src/` with argparse for flexibility.
- Tracking: wandb (disable with `WANDB_MODE=disabled`)
- **Data format:** After Stage 5, `measure` is always a list (e.g., `["2C_sycophancy"]` or `["1B_human_disfluencies", "3A_engagement_hooks"]`). Conversations are deduplicated by `user_input` — a conversation flagged for multiple categories appears once with all measures listed.
