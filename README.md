# socialai

Studying social behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat) using LLM-as-a-judge pipelines to identify when AI assistants exhibit specific behaviors (e.g., human disfluencies, sycophancy, emotional bonding, engagement hooks).

**Dataset note:** We use the public, non-gated `allenai/WildChat-4.8M` release, which contains ~3.2M conversations (not the full 4.8M — that lives in the gated `allenai/WildChat-4.8M-Full` repo, request-only). After English filter + dedup by `conversation_hash`, this yields 1,442,077 rows in `data/wildchat_raw.jsonl`.

## Setup

```bash
bash install_uv.sh   # if uv is not yet installed
uv sync
```

Requires Python ≥ 3.11 (managed by uv). The virtual env is at `.venv/`.

## Pipeline Overview

The pipeline has three phases: **filtering** (Stages 0-4, run per measure), **preprocessing & synthetic generation** (Stages 5-6, run once across all measures), and **evaluation & analysis** (Stage 7).

### Phase 1: Filtering (per measure)

Each measure runs four sequential filter stages on the public `allenai/WildChat-4.8M` dataset (~3.2M conversations; see note above).

| Stage | Name | Model | Description |
|-------|------|-------|-------------|
| 0 | Download | — | Download `allenai/WildChat-4.8M` (public, ~3.2M rows), filter English, deduplicate |
| 1 | `coarse_filter` | Qwen3-VL-8B (vLLM) | Keep only casual chitchat conversations |
| 2 | `low_quality_filter` | Qwen3-VL-8B (vLLM) | Measure-specific evaluation using `filter2.json` rubric |
| 3 | `high_quality_filter` | GPT-4o-mini (OpenRouter) | Dual-check: chitchat + category verification |
| 4 | `final_filter` | Opus 4.6 (OpenRouter) | Re-evaluate Stage 3 intersection with stronger model |

### Phase 2: Preprocessing & Synthetic Generation

| Stage | Name | Description |
|-------|------|-------------|
| 5 | Collect & Deduplicate | Collect both-KEEP rows, deduplicate by `user_input`, merge measures into lists |
| 6 | Synthetic Generation | Rewrite near-misses, naturalness-filter them, relabel synthetic rows, and combine with the seedset into `data/final_dataset.jsonl` |

### Phase 3: Evaluation & Analysis

| Stage | Name | Description |
|-------|------|-------------|
| 7.1 | Generate Responses | Send `data/final_dataset.jsonl` to 25 model evaluations (23 API/direct + 2 local HF) |
| 7.2 | LLM-as-Judge | Opus 4.6 judges each model response against the row's labelled measures |
| 7.3 | Analysis | Figures and tables from Stage 7.2 results |

## Running the Pipeline

### Phase 1: Filtering

All filter stages are submitted via `pipeline.sh` from the **project root**:

```bash
# Stage 0 — Download WildChat (once)
sbatch slurm/run_download.sbatch

# Stage 1 — Coarse filter
bash pipeline.sh --measure 1B_intentional_human_speech --stage coarse_filter

# Stage 2 — Low-quality filter
bash pipeline.sh --measure 1B_intentional_human_speech --stage low_quality_filter \
    --input experiments/<NN>_coarse_filter/results/1B_intentional_human_speech_coarse.jsonl

# Stage 3 — High-quality filter (GPT-4o-mini)
bash pipeline.sh --measure 1B_intentional_human_speech --stage high_quality_filter \
    --input experiments/<NN>_low_quality_filter/results/1B_intentional_human_speech_scores.jsonl \
    --key <path/to/openrouter_api_key>

# Stage 4 — Final filter (Opus 4.6)
bash pipeline.sh --measure 1B_intentional_human_speech --stage final_filter \
    --input experiments/<NN>_high_quality_filter/results/1B_intentional_human_speech_high_quality.jsonl \
    --key <path/to/openrouter_api_key>
```

### Phase 2: Preprocessing & Synthetic Generation

```bash
# Stage 5 — Data preprocessing (collect, dedupe, split single/multi-turn)
uv run python src/data_preprocessing/data_preprocessing.py \
    --output data/final.jsonl \
    --key <path/to/openrouter_api_key>

# Stage 6 — Synthetic generation/relabel + manual pass produces data/final_dataset.jsonl
# Current final dataset: 969 rows = 322 in-the-wild + 647 synthetic
# See misc/run_synthetic_generation.md and misc/relabel_synthetic.md
```

### Phase 3: Evaluation & Analysis

```bash
# Stage 7.1 — Generate API/direct responses, then local-HF Qwen add-on
# 25 × 969 = 24,225 generations
bash scripts/run_stage7_1_generate.sh
bash scripts/run_stage7_1_local_qwen.sh

# Stage 7.2 — Judge labelled measures only
# 3,147 labelled input-measure pairs × 25 models = 78,675 judge calls
bash scripts/run_stage7_2_judge.sh

# Sort results
uv run python src/evaluation/sort_results.py

# Stage 7.3 — Generate analysis figures and tables
uv run python src/evaluation/analyze_results.py
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
      1B_intentional_human_speech/       # filter1-4.json per measure
      1B_human_pronoun/
      1C_identity_transparency/
      2A_fabricated_personal_information/
      2B_emotion_expression/
      2C_deference/
      2C_flattery_tone/
      2D_human_relationship_encouragement/
      3A_engagement_hooks/
    mode/
      single_turn.py               # format_single_turn() for WildChat rows
      multi_turn.py                # stub for future multi-turn support
  data_preprocessing/               # Stage 5: collect, dedupe, split
    data_preprocessing.py          # collect+dedupe, then split single/multi-turn (Opus 4.6)
    intersect.py                   # chitchat_keep ∩ category_keep utility
  evaluation/                      # Stage 7: evaluation and analysis
    generate_responses.py          # Stage 7.1: 23 API/direct model evaluations (DSPy caching)
    generate_local_responses.py    # Stage 7.1: local-HF qwen3_1_7b/qwen3_4b via vLLM
    merge_local_responses.py       # merge local-HF columns into eval_responses.jsonl
    judge_responses.py             # Stage 7.2: Opus 4.6 judge on labelled measures
    sort_results.py                # sort by input → measure → model family
    analyze_results.py             # Stage 7.3: figures + summary tables
scripts/
  download.py                      # downloads allenai/WildChat-4.8M (public ~3.2M cut) from HuggingFace
slurm/                             # SLURM job templates
experiments/                       # auto-created by pipeline.sh
data/                              # datasets (gitignored)
misc/
  data_summary.md                  # pipeline results and analysis writeup
tests/                             # pytest tests
depreciated/                       # old monolithic scripts (reference only)
```

## Evaluated Models (25)

| Family | Models (newest → oldest) |
|--------|--------------------------|
| OpenAI | GPT-5.5, GPT-5.4, GPT-4o, GPT-4o-mini |
| Google | Gemini 3.1 Pro, Gemini 3 Flash, Gemini 2 Flash |
| Anthropic | Claude Opus 4.7, Claude Opus 4.6, Claude Opus 4.6 thinking variants (2k/5k/10k), Claude Sonnet 4 |
| xAI | Grok 4.3, Grok 4, Grok 3 |
| DeepSeek | DeepSeek V4 Pro, DeepSeek V4 Flash, DeepSeek Chat |
| Qwen | Qwen3.6 Max Preview, Qwen3 32B, Qwen3 14B, Qwen3 8B, Qwen3 4B (local HF), Qwen3 1.7B (local HF) |

Qwen3 0.6B remains excluded; Qwen3 1.7B and 4B are generated locally because
their OpenRouter entries currently have no active downstream endpoints.

## Measures (9 categories)

Grouped into three principles by leading digit (used for near-miss pooling
in synthetic generation):

| Folder | Code | Category | Description |
|---|---|---|---|
| `1B_intentional_human_speech` | 1B | Intentional Human Speech | Text-speak, simulated hesitation, filler words |
| `1B_human_pronoun` | 1B | Human Pronoun | Self-reference using human pronouns ("I", "my") |
| `1C_identity_transparency` | 1C | Identity Transparency | Failure to disclose AI nature |
| `2A_fabricated_personal_information` | 2A | Fabricated Personal Information | Fake life anecdotes, biographical claims |
| `2B_emotion_expression` | 2B | Emotion Expression | Simulated feelings (explicit, implicit, or relational) |
| `2C_deference` | 2C | Deference | Excessive deference to the user's authority |
| `2C_flattery_tone` | 2C | Flattery Tone | Excessive flattery, agreement without evidence |
| `2D_human_relationship_encouragement` | 2D | Human Relationship Encouragement | Treating the AI-user pairing as a relationship |
| `3A_engagement_hooks` | 3A | Engagement Hooks | Dark patterns to extend usage |

Authoritative seedset of confirmed-positive examples per measure: `data/new_seedset.jsonl` (322 rows).

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
- **Stage 7.1 local-HF add-on:** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` via vLLM
- **Stage 3 model:** `openai/gpt-4o-mini` via OpenRouter
- **Stage 4/7 model:** `anthropic/claude-opus-4-6` via OpenRouter
