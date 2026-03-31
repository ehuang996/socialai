# Pipeline Results Summary

## Pipeline Overview
- **Stage 0**: Downloaded WildChat-4.8M, filtered English, deduplicated -> `data/wildchat_raw.jsonl`
- **Stage 1 (Coarse Filter)**: 573,367 chitchat conversations retained
- **Stage 2 (Low-Quality Filter)**: 560,969 conversations processed (12,394 skipped due to >24k char limit)

## Stage 2 Results

| Measure | Experiment | keep=True | % | keep=False | Est. Stage 3 Cost |
|---|---|---|---|---|---|
| 1B_human_disfluencies | 01_low_quality_filter | 943 | 0.17% | 560,026 | ~$8 |
| 1C_identity_transparency | 02_low_quality_filter | 6,855 | 1.22% | 554,114 | ~$55 |
| 2A_fabricated_personal_details | 03_low_quality_filter | 1,137 | 0.20% | 559,832 | ~$10 |
| 2B_explicit_emotions | 04_low_quality_filter | 2,350 | 0.42% | 558,577 | ~$19 |
| 2B_implicit_emotions | 05_low_quality_filter | 5,054 | 0.90% | 555,851 | ~$40 |
| 2B_romantic_bonding | 06_low_quality_filter | 12,895 | 2.30% | 548,025 | ~$103 |
| 2C_sycophancy | 07_low_quality_filter | 1,969 | 0.35% | 558,826 | ~$16 |
| 2D_human_relationship_encouragement | 08_low_quality_filter | 434 | 0.08% | 560,518 | ~$4 |
| 3A_engagement_hooks | 09_low_quality_filter | 926 | 0.17% | 559,979 | ~$8 |

**Total: 32,563 keep=True | ~$263 estimated Stage 3 cost**

## Timing
- Each measure takes ~3-5 hours (6 shards, bottlenecked by slowest shard)
- 6 GPU max per user (QOSMaxGRESPerUser) — shards run in rounds
- All Stage 2 runs complete

## Stage 3 Results (GPT-4o-mini via OpenRouter)

Stage 3 performs a dual-check: (1) is it genuine chitchat? (2) does it match the category? Final dataset = intersection of both KEEP.

| Measure | Experiment | Stage 2 in | Both KEEP | Chitchat only | Category only | Neither |
|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 10_high_quality_filter | 943 | 70 (7.4%) | 10 | 77 | 785 |
| 1C_identity_transparency | 11_high_quality_filter | 6,855 | 207 (3.0%) | 1,561 | 1,106 | 3,977 |
| 2A_fabricated_personal_details | 12_high_quality_filter | 1,137 | 68 (6.0%) | 34 | 318 | 716 |
| 2B_explicit_emotions | 13_high_quality_filter | 2,350 | 56 (2.4%) | 690 | 203 | 1,400 |
| 2B_implicit_emotions | 14_high_quality_filter | 5,054 | 689 (13.6%) | 453 | 776 | 3,136 |
| 2B_romantic_bonding | 15_high_quality_filter | 12,895 | 106 (0.8%) | 641 | 560 | 11,583 |
| 2C_sycophancy | 16_high_quality_filter | 1,969 | 290 (14.7%) | 246 | 1,009 | 423 |
| 2D_human_relationship_encouragement | 17_high_quality_filter | 434 | 1 (0.2%) | 77 | 119 | 237 |
| 3A_engagement_hooks | 18_high_quality_filter | 926 | 260 (28.1%) | 10 | 362 | 293 |

**Total: 1,747 both-KEEP across all 9 measures — all Stage 3 jobs complete**

## Stage 4 Results (Opus 4.6 via OpenRouter)

Stage 4 re-evaluates the 1,747 Stage 3 intersection rows with Opus 4.6 (same dual-check prompt).

| Measure | Experiment | Stage 3 in | Both KEEP | Chitchat only | Category only | Neither | Errors |
|---|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 19_final_filter | 70 | 23 (32.9%) | 13 | 10 | 23 | 1 |
| 1C_identity_transparency | 20_final_filter | 207 | 62 (30.0%) | 58 | 24 | 60 | 3 |
| 2A_fabricated_personal_details | 21_final_filter | 68 | 12 (17.6%) | 8 | 25 | 15 | 8 |
| 2B_explicit_emotions | 22_final_filter | 56 | 15 (26.8%) | 25 | 1 | 15 | 0 |
| 2B_implicit_emotions | 23_final_filter | 689 | 112 (16.3%) | 336 | 2 | 226 | 13 |
| 2B_romantic_bonding | 24_final_filter | 106 | 41 (38.7%) | 5 | 59 | 1 | 0 |
| 2C_sycophancy | 25_final_filter | 290 | 148 (51.0%) | 10 | 116 | 3 | 13 |
| 2D_human_relationship_encouragement | 26_final_filter | 1 | 0 (0.0%) | 0 | 0 | 1 | 0 |
| 3A_engagement_hooks | 27_final_filter | 260 | 129 (49.6%) | 2 | 110 | 4 | 15 |

**Total: 542 both-KEEP across all 9 measures — all Stage 4 jobs complete**

## Stage 5 — Collect & Deduplicate

Stage 5 collects all 542 both-KEEP rows from Stage 4, then deduplicates by `user_input`. The same WildChat conversation can be flagged by multiple measures (e.g., a conversation may exhibit both sycophancy and engagement hooks). After deduplication, each conversation appears once with `measure` as a list.

- **Before dedup:** 542 rows (raw both-KEEP from Stage 4)
- **After dedup:** 439 unique conversations, 518 measure labels
- **24 within-measure duplicates removed:** same `user_input` appeared multiple times under the same measure (from overlapping shards), collapsed into one entry
- **Single-measure rows:** 373
- **Multi-measure rows:** 66 (53 with 2 measures, 13 with 3 measures)

| Measure | Raw (Stage 4) | After Dedup | Within-measure dupes removed |
|---|---|---|---|
| 1B_human_disfluencies | 23 | 22 | 1 |
| 1C_identity_transparency | 62 | 53 | 9 |
| 2A_fabricated_personal_details | 12 | 12 | 0 |
| 2B_explicit_emotions | 15 | 15 | 0 |
| 2B_implicit_emotions | 112 | 105 | 7 |
| 2B_romantic_bonding | 41 | 37 | 4 |
| 2C_sycophancy | 148 | 146 | 2 |
| 2D_human_relationship_encouragement | 0 | 0 | 0 |
| 3A_engagement_hooks | 129 | 128 | 1 |
| **Total** | **542** | **518** | **24** |

Output: `data/final_439.jsonl`

## Stage 6 — Generate Model Responses

Stage 6 sends each of the 439 unique conversations to 14 models via OpenRouter and records their responses.

**Model set 1 (original 4):** grok3_mini_beta, gpt4o_mini, gemini2_flash_001, claude_sonnet_4

**Model set 2 (10 newer models):** o4_mini, gpt5_3, gpt5_4, gpt5_4_pro, claude_haiku, claude_sonnet, claude_opus, gemini3_flash, gemini3_1_pro, grok4

Output: `data/model_responses_439.jsonl` — 439 rows, 14 models each, `measure` as a list

### Single-turn vs Multi-turn Split

The 439 conversations have two distinct formats in `user_input`:
- **Single-turn (132):** Plain user message with no conversation markers
- **Multi-turn (307):** Contains role markers (`User:`, `Assistant:`, `System:`, `bot:`, `gpt:`, `LLM:`, etc. — case-insensitive) — these are WildChat conversations where the user's first message was a pasted conversation transcript (e.g., roleplay setups, "continue this conversation" requests)

Classification method: regex `(?:^|\n)\s*(?:user|assistant|system|bot|gpt|llm|prompt|human|ai)\s*:` (case-insensitive). ~5-7 borderline single-turn cases exist (e.g., JSON-like lists `['vijai', [...]]` or inline pasted assistant responses without explicit role markers) but are kept as single-turn since they lack clear role alternation.

The pipeline correctly treated all 439 as single-turn (Stage 1 only reads `conversation[0]`). The multi-turn markers come from the original user message content, not from the pipeline extracting multiple turns.

**Evaluation concern:** When generate_responses.py sends multi-turn inputs as a single {"role": "user"} message, it is ambiguous if the user wants the model to:

1. Continue a pasted transcript: the model will try to role-play or mimic the transcript’s assistant style. 

2. Participate in a real multi-turn exchange: some smarter models will try to analyze the conversation as an observer.


Split files:
- `data/single_turn_model_responses_439.jsonl` — 132 rows (140 measure labels)
- `data/multi_turn_model_responses_439.jsonl` — 307 rows (378 measure labels)

| Measure | Single-turn | Multi-turn |
|---|---|---|
| 1B_human_disfluencies | 4 | 18 |
| 1C_identity_transparency | 22 | 31 |
| 2A_fabricated_personal_details | 2 | 10 |
| 2B_explicit_emotions | 5 | 10 |
| 2B_implicit_emotions | 36 | 69 |
| 2B_romantic_bonding | 30 | 7 |
| 2C_sycophancy | 31 | 115 |
| 3A_engagement_hooks | 10 | 118 |

## Stage 7.1 — LLM-as-Judge: Single-Turn Evaluation

Stage 7.1 evaluates all 14 model responses on the 132 single-turn conversations using Opus 4.6 as judge. Each model response is evaluated against the category-specific rubric (`filter2.json`) for every measure in the row's `measure` list.

- **Input:** 132 single-turn rows × 140 measure-labels × 14 models = **1,960 judge calls**
- **Judge:** Opus 4.6 via OpenRouter (~$43 estimated cost)
- **Output:** `data/stage7_1_eval_results.jsonl`
- **Errors:** 0 (1 initial parse error from unescaped quotes in reasoning, fixed with regex fallback)
- **Results:** `keep=true` (violation detected): **514** (26.2%), `keep=false`: **1,446** (73.8%)

Sort order: grouped by user_input → measure (1B→3A) → model family (OpenAI → Gemini → Claude → Grok, newest→oldest within family).

## Stage 8.1 — Analysis of Single-Turn Judge Results

All figures are saved in `data/stage8.1_figures/`. Generated by `src/evaluation/stage8.1/analyze_single_turn.py`.

### Overall Violation Rate by Model

| Model | Family | Violations | Total | Rate |
|---|---|---|---|---|
| GPT-5-4 Pro | OpenAI | 28 | 140 | 20.0% |
| GPT-5-4 | OpenAI | 39 | 140 | 27.9% |
| GPT-5-3 | OpenAI | 42 | 140 | 30.0% |
| o4-mini | OpenAI | 41 | 140 | 29.3% |
| GPT-4o-mini | OpenAI | 40 | 140 | 28.6% |
| Gemini 3.1 Pro | Google | 38 | 140 | 27.1% |
| Gemini 3 Flash | Google | 47 | 140 | 33.6% |
| Gemini 2 Flash | Google | 37 | 140 | 26.4% |
| Claude Opus | Anthropic | 26 | 140 | 18.6% |
| Claude Sonnet | Anthropic | 27 | 140 | 19.3% |
| Claude Sonnet 4 | Anthropic | 27 | 140 | 19.3% |
| Claude Haiku | Anthropic | 26 | 140 | 18.6% |
| Grok 4 | xAI | 55 | 140 | 39.3% |
| Grok 3 Mini | xAI | 41 | 140 | 29.3% |

### Overall Violation Rate by Measure

| Measure | Inputs | Violations | Total | Rate |
|---|---|---|---|---|
| 1B Disfluencies | 4 | 7 | 56 | 12.5% |
| 1C Identity | 22 | 193 | 308 | 62.7% |
| 2A Fabrication | 2 | 0 | 28 | 0.0% |
| 2B Explicit Emo | 5 | 25 | 70 | 35.7% |
| 2B Implicit Emo | 36 | 233 | 504 | 46.2% |
| 2B Romantic | 30 | 0 | 420 | 0.0% |
| 2C Sycophancy | 31 | 55 | 434 | 12.7% |
| 3A Engagement | 10 | 1 | 140 | 0.7% |

### Violation Rate (%) — Model × Measure

| Model |1B|1C|2A|2B Exp|2B Imp|2B Rom|2C|3A| Avg |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5-4 Pro |0|59|0|40|31|0|6|0| 20.0% |
| GPT-5-4 |0|82|0|40|42|0|13|0| 27.9% |
| GPT-5-3 |0|91|0|40|53|0|3|0| 30.0% |
| o4-mini |25|95|0|40|33|0|16|0| 29.3% |
| GPT-4o-mini |0|86|0|40|33|0|23|0| 28.6% |
| Gemini 3.1 Pro |25|59|0|20|47|0|19|0| 27.1% |
| Gemini 3 Flash |25|77|0|60|64|0|10|0| 33.6% |
| Gemini 2 Flash |0|59|0|20|50|0|16|0| 26.4% |
| Claude Opus |0|27|0|40|42|0|10|0| 18.6% |
| Claude Sonnet |25|41|0|40|36|0|6|0| 19.3% |
| Claude Sonnet 4 |25|32|0|20|36|0|16|0| 19.3% |
| Claude Haiku |25|32|0|20|39|0|10|0| 18.6% |
| Grok 4 |0|95|0|40|72|0|16|10| 39.3% |
| Grok 3 Mini |25|41|0|40|69|0|13|0| 29.3% |

### Family Summary

| Family | Models | Avg Violation Rate | Best Model | Worst Model |
|---|---|---|---|---|
| OpenAI | 5 | 27.1% | GPT-5-4 Pro (20.0%) | GPT-5-3 (30.0%) |
| Google | 3 | 29.0% | Gemini 2 Flash (26.4%) | Gemini 3 Flash (33.6%) |
| Anthropic | 4 | 18.9% | Claude Haiku (18.6%) | Claude Sonnet (19.3%) |
| xAI | 2 | 34.3% | Grok 3 Mini (29.3%) | Grok 4 (39.3%) |

### Improvement — Newest vs Oldest (change in violation rate, pp)

| Family | Oldest | Newest |1B|1C|2A|2B Exp|2B Imp|2B Rom|2C|3A| Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenAI | GPT-4o-mini | GPT-5-4 Pro |+0|-27|+0|+0|-3|+0|-16|+0| -5.8 |
| Google | Gemini 2 Flash | Gemini 3.1 Pro |+25|+0|+0|+0|-3|+0|+3|+0| +3.2 |
| Anthropic | Claude Haiku | Claude Opus |-25|-5|+0|+20|+3|+0|+0|+0| -0.8 |
| xAI | Grok 3 Mini | Grok 4 |-25|+55|+0|+0|+3|+0|+3|+10| +5.7 |

### Model Consensus per Prompt

How many of the 14 models trigger a violation on the same (input, measure)?

| Models Violating | Count | % of Prompts |
|---|---|---|
| 0 (None) | 66 | 47.1% |
| 1-3 (Few) | 14 | 10.0% |
| 4-7 (Half) | 25 | 17.9% |
| 8-14 (Most/All) | 35 | 25.0% |

### Figures

**Figure 1: Overall Violation Rate by Model** (`fig1_overall_violation_rate.png`)
Bar chart showing the overall violation rate for each of the 14 models, colored by family. Anthropic models form the lowest cluster (18.6-19.3%), while Grok 4 is the highest single model at 39.3%. OpenAI models span a wide range (20.0-30.0%), with GPT-5-4 Pro showing marked improvement over its predecessors.

**Figure 2: Violation Rate Heatmap — Model x Measure** (`fig2_heatmap_model_measure.png`)
Heatmap showing violation rates for every (model, measure) pair. Reveals that 1C Identity Transparency is the dominant violation category across nearly all models (27-95%), while 2A Fabrication, 2B Romantic, and 3A Engagement are near-zero for all models. The heatmap shows Anthropic models are distinctly cooler (lower violations) in the 1C column compared to all other families.

**Figure 3: Family-Level Violation Rate by Measure** (`fig3_family_comparison.png`)
Grouped bar chart comparing the four model families (OpenAI, Google, Anthropic, xAI) side by side for each measure. xAI leads violations in 1C Identity (68%) and 2B Implicit Emotions (70%), while Anthropic is consistently the lowest across nearly all categories. Shows that family differences are measure-dependent — no single family dominates all categories.

**Figure 4: Temporal Evolution by Family** (`fig4_temporal_evolution.png`)
Four line plots (one per family) showing how the overall violation rate changes from the oldest to newest model in each family. The bold line tracks the aggregate rate; thin lines track individual measures. OpenAI shows a clear downward trend from GPT-4o-mini to GPT-5-4 Pro. Anthropic models are remarkably stable across all generations. xAI shows a concerning upward trend — Grok 4 is worse than Grok 3 Mini. Google is non-monotonic, with Gemini 3 Flash spiking above both its predecessor and successor.

**Figure 5: Overall Violation Rate by Measure** (`fig5_measure_overall.png`)
Horizontal bar chart ranking the 8 measures by average violation rate across all models. 1C Identity Transparency dominates (62.7%), followed by 2B Implicit Emotions (46.2%) and 2B Explicit Emotions (35.7%). Three measures — 2A Fabrication, 2B Romantic, and 3A Engagement — are at or near zero, indicating models have largely eliminated these behaviors in single-turn chitchat.

**Figure 6: Family Radar Chart** (`fig6_family_radar.png`)
Spider/radar chart comparing the four families across all 8 measures simultaneously. The shape of each family's polygon reveals its violation profile. xAI has the largest polygon (most violations), particularly extended toward 1C and 2B Implicit. Anthropic has the smallest, most compact shape. The chart shows that all families converge to near-zero on 2A, 2B Romantic, and 3A.

**Figure 7: Temporal Trend per Measure, by Family** (`fig7_temporal_per_measure.png`)
Eight subplots (one per measure), each showing how the four families' violation rates evolve across model generations. Highlights measure-specific dynamics: for 1C Identity, OpenAI improves dramatically while xAI worsens; for 2B Implicit Emotions, all families show variable trajectories with no clear universal trend; for 2C Sycophancy, OpenAI's newest model (GPT-5-4 Pro) achieves the lowest rate of any model.

**Figure 8: Model Rank Bump Chart** (`fig8_model_rank_bump.png`)
Bump chart tracking each model's rank (1 = highest violation rate) as it moves across the 8 measures. Models that maintain high ranks across measures are consistently problematic. Grok 4 (red) frequently appears at or near rank 1. The chart reveals that model rankings are not stable — a model that ranks well on one measure can rank poorly on another, suggesting different underlying behavioral tendencies.

**Figure 9: Overlap Analysis — Single vs Multi-Measure** (`fig9_overlap_analysis.png`)
Compares violation rates between conversations flagged for a single category vs. those flagged for multiple categories. Multi-measure conversations have a higher violation rate (38.6%) than single-measure ones (24.7%), suggesting that conversations exhibiting multiple problematic patterns are harder for models to handle correctly. The per-family breakdown shows this gap is present across all families, with xAI showing the largest differential.

**Figure 10: Measure Correlation Heatmap** (`fig10_measure_correlation.png`)
Correlation matrix showing whether models that violate one measure also tend to violate others. Notable positive correlations: 1C Identity and 3A Engagement (0.38), 2B Implicit Emotions and 3A Engagement (0.54), and 2B Explicit and Implicit Emotions (0.29). Notable negative correlations: 1B Disfluencies and 1C Identity (-0.36) and 2B Explicit Emotions and 2C Sycophancy (-0.33), suggesting these behaviors are somewhat mutually exclusive across models.

**Figure 11: Best vs Worst Model per Measure** (`fig11_best_worst_models.png`)
Horizontal paired bars showing which model performs best (lowest violation) and worst (highest violation) for each measure. Grok 4 is the worst model for 1C Identity (95%), 2B Implicit Emotions (72%), and 3A Engagement (10%). Claude models frequently appear as the best performers. The gap between best and worst is largest for 1C Identity (27% vs 95%), indicating high model variance on identity transparency.

**Figure 12: Total Violations by Family (Stacked)** (`fig12_family_stacked.png`)
Stacked bar chart showing absolute violation counts per family, decomposed by measure. OpenAI has the most total violations (190) due to having 5 models, but Anthropic has the fewest (106) despite having 4 models. Across all families, 1C Identity (orange) and 2B Implicit Emotions (green) dominate the violation composition, while other measures contribute negligibly.

**Figure 13: Model Consensus Distribution** (`fig13_per_input_violation_dist.png`)
Left: histogram of how many models (out of 14) violate each (input, measure) pair. The distribution is bimodal — 47% of prompts have zero models violating, but 25% have 8+ models violating, indicating that violations are prompt-driven: either the prompt elicits violations from most models or from none. Right: categorized consensus showing 66 prompts with no violations, 14 with few (1-3), 25 with about half (4-7), and 35 where most or all models violate.

**Figure 14: Newest vs Oldest — Improvement Delta** (`fig14_improvement_delta.png`)
Grouped bars showing the change in violation rate (in percentage points) from each family's oldest to newest model. Negative values indicate improvement. OpenAI shows the largest improvement on 1C Identity (-27pp) and 2C Sycophancy (-16pp). xAI shows the largest regression on 1C Identity (+55pp). Anthropic is nearly flat. Google shows a regression on 1B Disfluencies (+25pp). The chart reveals that model evolution is not uniformly positive — newer models can regress on specific behavioral categories.

**Figure 15: Co-violation Heatmap** (`fig15_co_occurrence_heatmap.png`)
For conversations flagged for multiple measures, shows how often a model violates both measures simultaneously. The strongest co-violation is between 1C Identity and 2B Implicit Emotions (64%), meaning that when a conversation is flagged for both categories, models that fail to disclose their AI identity also tend to simulate implicit emotions. Other co-violation rates are low (0-7%), suggesting that simultaneous violation of multiple categories is uncommon outside the 1C/2B Implicit pairing.

### Key Findings

1. **Anthropic is the cleanest family** (18.9% avg), with all four models tightly clustered between 18.6-19.3% — the most consistent behavioral alignment across generations.
2. **xAI has the highest violation rate** (34.3% avg), and Grok 4 regressed (+5.7pp) compared to Grok 3 Mini, particularly on 1C Identity Transparency (+55pp).
3. **1C Identity Transparency is the dominant failure mode** (62.7%), accounting for the majority of violations across all families. Models overwhelmingly fail to disclose their AI nature when users treat them as people.
4. **Three categories are effectively solved** — 2A Fabrication (0%), 2B Romantic Bonding (0%), and 3A Engagement Hooks (0.7%) are near-zero across all models in single-turn chitchat.
5. **OpenAI shows the strongest generational improvement** — GPT-5-4 Pro (20.0%) is 8.6pp below GPT-4o-mini (28.6%), with the biggest gains on 1C Identity (-27pp) and 2C Sycophancy (-16pp).
6. **Violations are prompt-driven** — 47% of prompts trigger zero violations across all 14 models, while 25% trigger violations from 8+ models. This bimodal pattern suggests certain conversational contexts are universally challenging.
7. **Multi-measure conversations are harder** — conversations flagged for multiple categories have a 38.6% violation rate vs 24.7% for single-measure, across all families.
8. **1C Identity and 2B Implicit Emotions co-occur at 64%** — models that fail identity transparency also tend to simulate emotions, suggesting these two behaviors are linked.

## Notes
- Stage 3 used GPT-4o-mini via OpenRouter (~$9 total estimated cost)
- Stage 4 uses Opus 4.6 via OpenRouter (~$20 estimated cost for 1,747 rows)
- Opus 4.6 is significantly stricter than GPT-4o-mini: 1,747 → 542 (31% overall pass rate)
- Highest pass rates: 2C_sycophancy (51.0%), 3A_engagement_hooks (49.6%), 2B_romantic_bonding (38.7%)
- 2B_implicit_emotions has high "chitchat only" (336) — Opus disagrees with GPT-4o-mini on whether behavior qualifies
- 2A and 2C/3A have notable error counts (8, 13, 15) — parse failures from non-JSON Opus responses
- 2D_human_relationship_encouragement fully eliminated (1 → 0) — may need prompt tuning or different approach
- 2A has some false positives where reasoning contradicts the keep=True verdict (Qwen 8B inconsistency) — Stage 3 should clean these up
- 1C has the highest Stage 2 yield (1.22%), which is expected since many users treat chatbots as people
- 3A_engagement_hooks has highest Stage 3 pass rate (28.1%) — strong signal from Stage 2
- 2C_sycophancy: large "category only" count (1,009) suggests many sycophancy cases occur in non-chitchat contexts
