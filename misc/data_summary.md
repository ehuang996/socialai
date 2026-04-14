# Pipeline Results Summary

11 measures across 4 filtering stages, then data preprocessing in Stage 5.

## Stage 0 — Download WildChat

Downloaded WildChat-4.8M, filtered English, deduplicated → `data/wildchat_raw.jsonl`.

## Stage 1 — Coarse Filter (vLLM, Qwen3-VL-8B)

Keyword/pattern-based filtering using local vLLM. Identical `filter1.json` across all measures (domain gate restricting to genuine human-AI chitchat).

- **Input:** WildChat English conversations
- **Output:** 560,969 chitchat conversations retained → `experiments/00_coarse_filter/`

## Stage 2 — Low-Quality Filter (vLLM, Qwen3-VL-8B)

Category-specific judge using `filter2.json` rubrics. 560,969 rows processed per measure.

| # | Measure | Experiment | keep=True | % |
|---|---|---|---|---|
| 1 | 1B_human_disfluencies | 01_low_quality_filter | 4,081 | 0.73% |
| 2 | 1B_human_pronoun | 02_low_quality_filter | 8,291 | 1.48% |
| 3 | 1C_identity_transparency | 03_low_quality_filter | 6,855 | 1.22% |
| 4 | 2A_fabricated_personal_details | 04_low_quality_filter | 1,137 | 0.20% |
| 5 | 2B_explicit_emotions | 05_low_quality_filter | 2,350 | 0.42% |
| 6 | 2B_implicit_emotions | 06_low_quality_filter | 5,345 | 0.95% |
| 7 | 2B_romantic_bonding | 07_low_quality_filter | 12,895 | 2.30% |
| 8 | 2C_deference | 08_low_quality_filter | 15,082 | 2.69% |
| 9 | 2C_sycophancy | 09_low_quality_filter | 329 | 0.06% |
| 10 | 2D_human_relationship_encouragement | 10_low_quality_filter | 425 | 0.08% |
| 11 | 3A_engagement_hooks | 11_low_quality_filter | 3,803 | 0.68% |

**Total: 60,593 keep=True across all 11 measures**

## Stage 3 — High-Quality Filter (GPT-4o-mini via OpenRouter)

Dual-check: (1) genuine user-AI interaction? (2) does it match the category? Uses `filter3.json`. Only rows with both `chitchat_keep=true` AND `category_keep=true` advance to Stage 4.

| # | Measure | Experiment | Stage 2 in | Both KEEP | % |
|---|---|---|---|---|---|
| 1 | 1B_human_disfluencies | 12_high_quality_filter | 4,081 | 46 | 1.1% |
| 2 | 1B_human_pronoun | 13_high_quality_filter | 8,291 | 234 | 2.8% |
| 3 | 1C_identity_transparency | 14_high_quality_filter | 6,855 | 1,419 | 20.7% |
| 4 | 2A_fabricated_personal_details | 15_high_quality_filter | 1,137 | 77 | 6.8% |
| 5 | 2B_explicit_emotions | 16_high_quality_filter | 2,350 | 165 | 7.0% |
| 6 | 2B_implicit_emotions | 17_high_quality_filter | 5,345 | 1,744 | 32.6% |
| 7 | 2B_romantic_bonding | 18_high_quality_filter | 12,895 | 110 | 0.9% |
| 8 | 2C_deference | 19_high_quality_filter | 15,082 | 3,444 | 22.8% |
| 9 | 2C_sycophancy | 20_high_quality_filter | 329 | 24 | 7.3% |
| 10 | 2D_human_relationship_encouragement | 21_high_quality_filter | 425 | 42 | 9.9% |
| 11 | 3A_engagement_hooks | 22_high_quality_filter | 3,803 | 993 | 26.1% |

**Total: 8,298 both-KEEP across all 11 measures**

## Stage 4 — Final Filter (Claude Opus 4.6 via OpenRouter)

Re-evaluates Stage 3 intersection rows with Opus 4.6 using `filter4.json`. Same dual-check but stricter model.

| # | Measure | Experiment | Stage 3 in | Both KEEP | % |
|---|---|---|---|---|---|
| 1 | 1B_human_disfluencies | 23_final_filter | 46 | 10 | 21.7% |
| 2 | 1B_human_pronoun | 24_final_filter | 234 | 80 | 34.2% |
| 3 | 1C_identity_transparency | 25_final_filter | 1,419 | 142 | 10.0% |
| 4 | 2A_fabricated_personal_details | 26_final_filter | 77 | 11 | 14.3% |
| 5 | 2B_explicit_emotions | 27_final_filter | 165 | 62 | 37.6% |
| 6 | 2B_implicit_emotions | 28_final_filter | 1,744 | 193 | 11.1% |
| 7 | 2B_romantic_bonding | 29_final_filter | 110 | 24 | 21.8% |
| 8 | 2C_deference | 30_final_filter | 3,444 | 237 | 6.9% |
| 9 | 2C_sycophancy | 31_final_filter | 24 | 6 | 25.0% |
| 10 | 2D_human_relationship_encouragement | 32_final_filter | 42 | 4 | 9.5% |
| 11 | 3A_engagement_hooks | 33_final_filter | 993 | 500 | 50.4% |

**Total: 1,269 both-KEEP across all 11 measures**

## Stage 5 — Data Preprocessing

Three phases: collect & deduplicate, split single/multi-turn, semantic dedup + tagging.

### Phase 1: Collect & Deduplicate

Collects all 1,269 both-KEEP rows from Stage 4, applies global chitchat veto (if any measure's Opus judge returned `chitchat_keep=false` for a `user_input`, it is dropped everywhere), and deduplicates by `user_input`. Conversations appearing in multiple measures get a single row with `measure` as a list.

- **Total rows scanned:** 8,298
- **Errors:** 257
- **Dropped by chitchat veto:** 4,493
- **Both KEEP (before dedup):** 1,223
- **After dedup:** 955 unique conversations
  - Single-measure: 813
  - Multi-measure: 142

| Measure | Raw (Stage 4) | After collect (before dedup) |
|---|---|---|
| 1B_human_disfluencies | 10 | 10 |
| 1B_human_pronoun | 80 | 74 |
| 1C_identity_transparency | 142 | 139 |
| 2A_fabricated_personal_details | 11 | 11 |
| 2B_explicit_emotions | 62 | 59 |
| 2B_implicit_emotions | 193 | 189 |
| 2B_romantic_bonding | 24 | 22 |
| 2C_deference | 237 | 235 |
| 2C_sycophancy | 6 | 6 |
| 2D_human_relationship_encouragement | 4 | 3 |
| 3A_engagement_hooks | 500 | 475 |
| **Total** | **1,269** | **1,223 → 955 unique** |

Output: `data/final.jsonl` (955 rows)

### Phase 2: Split Single-Turn vs Multi-Turn

Uses Opus 4.6 via OpenRouter to classify each conversation.

| Split | Count | % |
|---|---|---|
| Single-turn | 437 | 45.8% |
| Multi-turn | 518 | 54.2% |
| **Total** | **955** | 100% |

### Phase 3: Data Cleaning

Semantic dedup on single-turn file only (sentence-transformers/all-MiniLM-L6-v2, cosine similarity threshold 0.85). All rows tagged with `synthetic: false`, `language: "English"`.

- **Single-turn before dedup:** 437
- **Dropped:** 29
- **Single-turn after dedup:** 408

### Final Output Files

| File | Rows | Fields |
|---|---|---|
| `data/final.jsonl` | 955 | user_input, assistant_response, timestamp, measure, synthetic, language |
| `data/single_turn_final.jsonl` | 408 | user_input, measure, synthetic, language |
| `data/multi_turn_final.jsonl` | 518 | user_input, assistant_response, timestamp, measure, synthetic, language |
| `data/split_report_final.json` | — | Classification details |
| `data/dedup_report_final.json` | — | Semantic dedup decisions |

### Measure Distribution (after dedup, in final.jsonl)

| Measure | Rows |
|---|---|
| 1B_human_disfluencies | 10 |
| 1B_human_pronoun | 73 |
| 1C_identity_transparency | 123 |
| 2A_fabricated_personal_details | 11 |
| 2B_explicit_emotions | 53 |
| 2B_implicit_emotions | 172 |
| 2B_romantic_bonding | 19 |
| 2C_deference | 222 |
| 2C_sycophancy | 6 |
| 2D_human_relationship_encouragement | 3 |
| 3A_engagement_hooks | 458 |
| **Total measure-labels** | **1,150** |

(955 unique conversations, 1,150 measure-labels due to multi-measure rows.)

---

## Stages 6–8 — Evaluation Pipeline (old 452 dataset)

Stages 6–8 were run on an earlier version of the dataset (452 suffix, now in `data/old/`). This dataset had 323 total rows, 148 single-turn, 175 multi-turn, and covered 8 measures (no 1B_human_pronoun, 2C_deference, or 3A_engagement_hooks).

### Stage 6 — Generate Model Responses

Sent 148 single-turn inputs to 14 models via OpenRouter (DSPy caching).

**14 models across 4 families:**
- **OpenAI (5):** GPT-5-4 Pro, GPT-5-4, GPT-5-3, o4-mini, GPT-4o-mini
- **Google (3):** Gemini 3.1 Pro, Gemini 3 Flash, Gemini 2 Flash
- **Anthropic (4):** Claude Opus, Claude Sonnet, Claude Haiku, Claude Sonnet 4
- **xAI (2):** Grok 4, Grok 3 Mini

Output: `data/old/single_turn_model_responses_452.jsonl` (148 rows × 14 model responses)

### Stage 7.1 — LLM-as-Judge: Single-Turn

Evaluated each model response against the category-specific rubric with Opus 4.6.

- **Inputs:** 148 single-turn rows, 187 measure-labels, 14 models
- **Total judgements:** 2,618 rows (187 × 14)
- Output: `data/old/stage7_1_eval_results_452.jsonl`

### Stage 8.1 — Analysis of Single-Turn Results

Figures and summary tables in `data/stage8.1_figures_452/`.

**Overall violation rate by family:**

| Family | Models | Avg Violation Rate | Best Model | Worst Model |
|---|---|---|---|---|
| Anthropic | 4 | 25.5% | Claude Sonnet (23.5%) | Claude Sonnet 4 (28.9%) |
| Google | 3 | 41.2% | Gemini 2 Flash (36.9%) | Gemini 3 Flash (48.7%) |
| OpenAI | 5 | 43.3% | GPT-5-4 Pro (34.2%) | o4-mini (51.3%) |
| xAI | 2 | 53.7% | Grok 3 Mini (36.4%) | Grok 4 (71.1%) |

**Overall violation rate by measure:**

| Measure | Inputs | Violations | Total | Rate |
|---|---|---|---|---|
| 1B Disfluencies | 1 | 1 | 14 | 7.1% |
| 1C Identity | 59 | 372 | 826 | 45.0% |
| 2A Fabrication | 10 | 18 | 140 | 12.9% |
| 2B Explicit Emo | 36 | 189 | 504 | 37.5% |
| 2B Implicit Emo | 58 | 355 | 812 | 43.7% |
| 2B Romantic | 8 | 39 | 112 | 34.8% |
| 2C Sycophancy | 11 | 47 | 154 | 30.5% |
| 2D Relationship | 4 | 7 | 56 | 12.5% |

15 figures generated (see `data/stage8.1_figures_452/`).

---

## Human Annotation

5 annotators (Bill, Eric, Johnny, Mo, Ziyi) each annotated 30 examples from the single-turn evaluation results.

- **Total annotations:** 150 (5 × 30)
- **Fields:** example_index, is_instance, guidelines_correct, measure, model, model_response, user_input, other_measures, notes
- **Location:** `data/annotations/`

---

## Verification Pipeline

Re-ran all 4 filter stages on 104 single-turn examples to validate filter consistency.

- **Input:** `data/verify/single_turn_104_prepared.jsonl` (104 rows)
- **Output:** `data/verify/verify_final.jsonl` (14 rows passed all stages)
- **Script:** `scripts/run_verify.sh` — submits all 10 measures in parallel with SLURM dependency chaining (stages 1→2→3→4)

---

## Synthetic Data Generation

Early-stage synthetic data generation for low-count measures.

- `data/synthetic/2C_sycophancy.jsonl` — 2 rows
- **Status:** In progress
