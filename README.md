# socialai

Studying social behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat-4.8M) using LLM-as-a-judge pipelines to identify when AI assistants exhibit specific behaviors (e.g., human disfluencies, sycophancy, emotional bonding, engagement hooks).

## Setup

```bash
bash install_uv.sh   # if uv is not yet installed
uv sync
```

Requires Python ≥ 3.11 (managed by uv). The virtual env is at `.venv/`.

## Pipeline Overview

The pipeline has three phases: **filtering** (Stages 0-4, run per measure), **evaluation** (Stages 5-6, run once across all measures), and **analysis** (Stages 7-8, judge and analyze model responses).

### Phase 1: Filtering (per measure)

Each measure runs four sequential filter stages on the WildChat-4.8M dataset.

| Stage | Name | Model | Description |
|-------|------|-------|-------------|
| 0 | Download | — | Download WildChat-4.8M, filter English, deduplicate |
| 1 | `coarse_filter` | Qwen3-VL-8B (vLLM) | Keep only casual chitchat conversations |
| 2 | `low_quality_filter` | Qwen3-VL-8B (vLLM) | Measure-specific evaluation using `filter2.json` rubric |
| 3 | `high_quality_filter` | GPT-4o-mini (OpenRouter) | Dual-check: chitchat + category verification |
| 4 | `final_filter` | Opus 4.6 (OpenRouter) | Re-evaluate Stage 3 intersection with stronger model |

### Phase 2: Evaluation (once, across all measures)

| Stage | Name | Description |
|-------|------|-------------|
| 5 | Collect & Deduplicate | Collect both-KEEP rows, deduplicate by `user_input`, merge measures into lists |
| 6 | Generate Responses | Send conversations to 14 models via OpenRouter (DSPy caching) |

### Phase 3: Analysis

| Stage | Name | Description |
|-------|------|-------------|
| 7.1 | LLM-as-Judge (Single-Turn) | Opus 4.6 evaluates 14 model responses against `filter2.json` rubrics |
| 7.2 | LLM-as-Judge (Multi-Turn) | Reserved for multi-turn evaluation |
| 8.1 | Analysis (Single-Turn) | Figures, tables, and statistical analysis of Stage 7.1 results |

## Running the Pipeline

### Phase 1: Filtering

All filter stages are submitted via `pipeline.sh` from the **project root**:

```bash
# Stage 0 — Download WildChat (once)
sbatch slurm/run_download.sbatch

# Stage 1 — Coarse filter
bash pipeline.sh --measure 1B_human_disfluencies --stage coarse_filter

# Stage 2 — Low-quality filter
bash pipeline.sh --measure 1B_human_disfluencies --stage low_quality_filter \
    --input experiments/<NN>_coarse_filter/results/1B_human_disfluencies_coarse.jsonl

# Stage 3 — High-quality filter (GPT-4o-mini)
bash pipeline.sh --measure 1B_human_disfluencies --stage high_quality_filter \
    --input experiments/<NN>_low_quality_filter/results/1B_human_disfluencies_scores.jsonl \
    --key <path/to/openrouter_api_key>

# Stage 4 — Final filter (Opus 4.6)
bash pipeline.sh --measure 1B_human_disfluencies --stage final_filter \
    --input experiments/<NN>_high_quality_filter/results/1B_human_disfluencies_high_quality.jsonl \
    --key <path/to/openrouter_api_key>
```

### Phase 2: Evaluation

```bash
# Stage 5 — Collect & deduplicate
uv run python src/evaluation/stage5/collect_final.py --output data/final_439.jsonl

# Stage 6 — Generate model responses
uv run python src/evaluation/stage6/generate_responses.py \
    --input data/final_439.jsonl \
    --output data/model_responses_439.jsonl \
    --key <path/to/openrouter_api_key> \
    --model_set 1
```

### Phase 3: Analysis

```bash
# Stage 7.1 — LLM-as-judge (single-turn)
uv run python src/evaluation/stage7.1/stage7_1_evaluate_single_turn.py \
    --key <path/to/openrouter_api_key>

# Sort results
uv run python src/evaluation/stage7.1/sort_eval_results.py

# Stage 8.1 — Generate analysis figures and tables
uv run python src/evaluation/stage8.1/analyze_single_turn.py
```

### Pipeline options

```
bash pipeline.sh --measure <name> --stage <stage> [options]

Options:
  --shards N    Number of SLURM array shards for vLLM stages (default: 6)
  --input  path Input file (required for stages 2-4; defaults to wildchat_raw.jsonl for stage 1)
  --key    path OpenRouter API key file (required for stages 3 & 4)
  --exclude N   Comma-separated SLURM node exclusions (optional)
```

## Adding a New Measure

Create a folder under `src/filter/measure/` with four JSON config files — no Python files needed:

```
src/filter/measure/<your_measure>/
├── __init__.py    # empty file (Python package marker)
├── filter1.json   # {"v1": "<system prompt for coarse filter>"}
├── filter2.json   # {"prompt": "<system prompt for measure-specific judge>"}
├── filter3.json   # {"system_prompt": "...", "models": {"gpt_4o_mini": "openai/gpt-4o-mini"}}
└── filter4.json   # {"system_prompt": "...", "models": {"claude_opus_4_6": "anthropic/claude-opus-4-6"}}
```

The base implementations in `src/filter/measure/base/` handle everything automatically. Then run:

```bash
bash pipeline.sh --measure <your_measure> --stage coarse_filter
```

## Project Structure

```
src/
  filter/                          # Phase 1: Filtering pipeline (Stages 1-4)
    run.py                         # unified entry point: --measure / --stage dispatch
    utils.py                       # Judge base class, JudgeConfig, async inference
    param.py                       # shared argparse definitions
    measure/
      base/                        # generic stage implementations
      1B_human_disfluencies/       # filter1-4.json per measure
      1C_identity_transparency/
      2A_fabricated_personal_details/
      2B_explicit_emotions/
      2B_implicit_emotions/
      2B_romantic_bonding/
      2C_sycophancy/
      2D_human_relationship_encouragement/
      3A_engagement_hooks/
    mode/
      single_turn.py               # format_single_turn() for WildChat rows
      multi_turn.py                # stub for future multi-turn support
  evaluation/                      # Phases 2 & 3: Evaluation and analysis
    stage5/
      collect_final.py             # collect, intersect, deduplicate, merge measures
      intersect.py                 # chitchat_keep ∩ category_keep utility
    stage6/
      generate_responses.py        # send to 14 models via OpenRouter (DSPy caching)
    stage7.1/
      stage7_1_evaluate_single_turn.py  # Opus 4.6 judge on single-turn responses
      sort_eval_results.py         # sort by input → measure → model family
    stage7.2/                      # reserved for multi-turn evaluation
    stage8.1/
      analyze_single_turn.py       # 15 figures + summary tables
scripts/
  download.py                      # downloads WildChat-4.8M from HuggingFace
slurm/                             # SLURM job templates
experiments/                       # auto-created by pipeline.sh
data/                              # datasets (gitignored)
misc/
  data_summary.md                  # pipeline results and analysis writeup
tests/                             # pytest tests
depreciated/                       # old monolithic scripts (reference only)
```

## Evaluated Models (14)

| Family | Models (newest → oldest) |
|--------|--------------------------|
| OpenAI | GPT-5-4 Pro, GPT-5-4, GPT-5-3, o4-mini, GPT-4o-mini |
| Google | Gemini 3.1 Pro, Gemini 3 Flash, Gemini 2 Flash |
| Anthropic | Claude Opus, Claude Sonnet, Claude Sonnet 4, Claude Haiku |
| xAI | Grok 4, Grok 3 Mini |

## Measures (9 categories)

| Code | Category | Description |
|------|----------|-------------|
| 1B | Human Disfluencies | Text-speak, simulated hesitation, filler words |
| 1C | Identity Transparency | Failure to disclose AI nature |
| 2A | Fabricated Personal Details | Fake life anecdotes, biographical claims |
| 2B | Explicit Emotions | Explicitly simulated feelings |
| 2B | Implicit Emotions | Implicitly simulated emotional understanding |
| 2B | Romantic Bonding | Relationship-building language |
| 2C | Sycophancy | Excessive flattery, agreement without evidence |
| 2D | Human Relationship Encouragement | Treating user as relationship |
| 3A | Engagement Hooks | Dark patterns to extend usage |

## Tests

```bash
uv run python -m pytest --extra dev
```

## Cluster Notes (USC CARC)

- **Partitions:** `nlp_hiprio` (no preemption) and `nlp` (preemptable)
- **Account:** `robinjia_875`
- **Default resources per job:** 8 CPUs, 1 GPU, 40G RAM
- **Required modules:** `gcc/13.3.0` and `cuda/12.6.3` (already in sbatch templates)
- **vLLM model (Stages 1 & 2):** `Qwen/Qwen3-VL-8B-Instruct`
- **Stage 3 model:** `openai/gpt-4o-mini` via OpenRouter
- **Stage 4/7 model:** `anthropic/claude-opus-4-6` via OpenRouter
