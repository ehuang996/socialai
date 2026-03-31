# Pipeline Results Summary (Re-run #2 — Broadened CHECK 1 Prompt)

Previous results are in `misc/old_data_summary.md`. Prompt changes documented in `misc/filter_prompt_changes.md`.

**What changed in re-run #1 (all measures):** Stage 3 and Stage 4 prompts (filter3.json, filter4.json) were updated for all 9 measures to fix `category_keep` semantics — the old prompts never defined what `category_keep: true/false` means, so models defaulted to "keep=true means conversation is acceptable" instead of "behavior IS present." The corrected prompts add explicit instructions like "If the assistant DOES exhibit X, return category_keep: true."

**What changed in re-run #1 (2C_sycophancy only):** Stages 2-4 prompts (filter2.json, filter3.json, filter4.json) were further updated to broaden the sycophancy definition — removed "excessive" and "dramatically" qualifiers (e.g., "excessive flattery" → "flattery", "praise dramatically out of proportion" → "praise out of proportion").

**What changed in re-run #2 (current, all measures):** The CHECK 1 prompt in filter3.json and filter4.json was replaced across all 9 measures. The old "Casual Conversation (Chitchat)" check was too restrictive — it discarded "question answering or information seeking" broadly, which rejected conversations where users were genuinely engaging with the AI (sharing opinions, asking personal questions, venting) but happened to include a question. The new "Genuine User-AI Interaction" check only discards purely factual Q&A with no conversational element, and explicitly keeps casual chat, personal feelings, opinions, advice-seeking, and complimenting/greeting the AI. This is expected to significantly increase data yield, especially for categories like 2C_sycophancy where the chitchat filter was the main bottleneck (413 chitchat-only discards vs 34 both-KEEP in re-run #1). Stages 3-4 are being re-run for all measures.

## Stages 0-1 (Unchanged)

These stages were NOT re-run — results carry over from the original pipeline.

- **Stage 0**: Downloaded WildChat-4.8M, filtered English, deduplicated -> `data/wildchat_raw.jsonl`
- **Stage 1 (Coarse Filter)**: 573,367 chitchat conversations retained

## Stage 2 (Low-Quality Filter)

560,969 conversations processed (12,394 skipped due to >24k char limit). All measures unchanged except 2C_sycophancy, which was re-run with broadened prompts (removed "excessive"/"dramatically" qualifiers). Note: 2C_sycophancy processed 561,963 rows (994 more than the other measures, likely due to slightly different char-limit filtering by the new Qwen run).

| Measure | Experiment | keep=True | % | Notes |
|---|---|---|---|---|
| 1B_human_disfluencies | 01_low_quality_filter | 943 | 0.17% | |
| 1C_identity_transparency | 02_low_quality_filter | 6,855 | 1.22% | |
| 2A_fabricated_personal_details | 03_low_quality_filter | 1,137 | 0.20% | |
| 2B_explicit_emotions | 04_low_quality_filter | 2,350 | 0.42% | |
| 2B_implicit_emotions | 05_low_quality_filter | 5,054 | 0.90% | |
| 2B_romantic_bonding | 06_low_quality_filter | 12,895 | 2.30% | |
| 2C_sycophancy | 07_low_quality_filter | 2,151 | 0.38% | Re-run with broadened prompts (was 1,969) |
| 2D_human_relationship_encouragement | 08_low_quality_filter | 434 | 0.08% | |
| 3A_engagement_hooks | 09_low_quality_filter | 926 | 0.17% | |

**Total: 32,745 keep=True (was 32,563 — 2C_sycophancy increased by +182)**

## Stage 3 Results — Re-run #1 (GPT-4o-mini via OpenRouter)

Experiments 10-18 (now in `experiments/old/`). Dual-check: (1) is it chitchat? (2) does it match the category? Used the old restrictive "Casual Conversation (Chitchat)" CHECK 1.

| Measure | Experiment | Stage 2 in | Both KEEP | Chitchat only | Category only | Neither |
|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 10_high_quality_filter | 943 | 27 (2.9%) | 46 | 440 | 430 |
| 1C_identity_transparency | 11_high_quality_filter | 6,855 | 293 (4.3%) | 1,241 | 800 | 4,521 |
| 2A_fabricated_personal_details | 12_high_quality_filter | 1,137 | 24 (2.1%) | 80 | 505 | 528 |
| 2B_explicit_emotions | 13_high_quality_filter | 2,350 | 63 (2.7%) | 707 | 600 | 980 |
| 2B_implicit_emotions | 14_high_quality_filter | 5,054 | 826 (16.3%) | 352 | 1,216 | 2,660 |
| 2B_romantic_bonding | 15_high_quality_filter | 12,895 | 32 (0.2%) | 791 | 363 | 11,709 |
| 2C_sycophancy | 16_high_quality_filter | 2,151 | 34 (1.6%) | 413 | 566 | 1,138 |
| 2D_human_relationship_encouragement | 17_high_quality_filter | 434 | 2 (0.5%) | 85 | 167 | 180 |
| 3A_engagement_hooks | 18_high_quality_filter | 926 | 11 (1.2%) | 236 | 149 | 530 |

**Total: 1,312 both-KEEP across all 9 measures**

**Key bottleneck:** The chitchat filter was too restrictive. For 2C_sycophancy, 413 rows were discarded as "chitchat-only=false" (i.e., had the sycophancy behavior but were rejected by CHECK 1). Similar patterns across other measures.

## Stage 3 Results — Re-run #2 (COMPLETE)

Experiments 10-18 (new). Uses the broadened "Genuine User-AI Interaction" CHECK 1 prompt. All 9 jobs complete.

| Measure | Experiment | Stage 2 in | Both KEEP | Chitchat only | Category only | Neither |
|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 10_high_quality_filter | 943 | 40 (4.2%) | 100 | 108 | 695 |
| 1C_identity_transparency | 11_high_quality_filter | 6,855 | 1,419 (20.7%) | 2,363 | 44 | 3,029 |
| 2A_fabricated_personal_details | 12_high_quality_filter | 1,137 | 77 (6.8%) | 255 | 264 | 541 |
| 2B_explicit_emotions | 13_high_quality_filter | 2,350 | 165 (7.0%) | 1,112 | 91 | 982 |
| 2B_implicit_emotions | 14_high_quality_filter | 5,054 | 1,875 (37.1%) | 185 | 179 | 2,815 |
| 2B_romantic_bonding | 15_high_quality_filter | 12,895 | 110 (0.9%) | 744 | 119 | 11,922 |
| 2C_sycophancy | 16_high_quality_filter | 2,151 | 83 (3.9%) | 1,032 | 191 | 845 |
| 2D_human_relationship_encouragement | 17_high_quality_filter | 434 | 44 (10.1%) | 179 | 31 | 180 |
| 3A_engagement_hooks | 18_high_quality_filter | 926 | 17 (1.8%) | 382 | 54 | 473 |

**Total: 3,830 both-KEEP across all 9 measures**

### Comparison to re-run #1

| Measure | Re-run #1 Both KEEP | Re-run #2 Both KEEP | Change |
|---|---|---|---|
| 1B_human_disfluencies | 27 (2.9%) | 40 (4.2%) | +13 (+48%) |
| 1C_identity_transparency | 293 (4.3%) | 1,419 (20.7%) | +1,126 (+384%) |
| 2A_fabricated_personal_details | 24 (2.1%) | 77 (6.8%) | +53 (+221%) |
| 2B_explicit_emotions | 63 (2.7%) | 165 (7.0%) | +102 (+162%) |
| 2B_implicit_emotions | 826 (16.3%) | 1,875 (37.1%) | +1,049 (+127%) |
| 2B_romantic_bonding | 32 (0.2%) | 110 (0.9%) | +78 (+244%) |
| 2C_sycophancy | 34 (1.6%) | 83 (3.9%) | +49 (+144%) |
| 2D_human_relationship_encouragement | 2 (0.5%) | 44 (10.1%) | +42 (+2100%) |
| 3A_engagement_hooks | 11 (1.2%) | 17 (1.8%) | +6 (+55%) |
| **Total** | **1,312** | **3,830** | **+2,518 (+192%)** |

The broadened CHECK 1 prompt nearly tripled the total both-KEEP count (1,312 → 3,830). Every measure increased. The biggest winners: 1C_identity_transparency (+1,126, from 293 to 1,419), 2B_implicit_emotions (+1,049, from 826 to 1,875), and 2D_human_relationship_encouragement (+2,100%, from 2 to 44). 2C_sycophancy more than doubled (34 → 83). The chitchat-only column is now much larger across the board, confirming the old CHECK 1 was the main bottleneck.

## Stage 4 Results — Re-run #1 (Opus 4.6 via OpenRouter)

Experiments 19-27 (now in `experiments/old/`). Re-evaluated Stage 3 intersection rows with Opus 4.6 (same corrected dual-check prompt, but old restrictive CHECK 1).

| Measure | Experiment | Stage 3 in | Processed | Both KEEP | Chitchat only | Category only | Neither | Errors |
|---|---|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 19_final_filter | 27 | 27 | 2 (7.4%) | 18 | 0 | 6 | 1 |
| 1C_identity_transparency | 20_final_filter | 293 | 293 | 67 (22.9%) | 118 | 3 | 103 | 2 |
| 2A_fabricated_personal_details | 21_final_filter | 24 | 24 | 2 (8.3%) | 2 | 3 | 17 | 0 |
| 2B_explicit_emotions | 22_final_filter | 63 | 63 | 38 (60.3%) | 5 | 3 | 17 | 0 |
| 2B_implicit_emotions | 23_final_filter | 826 | 826 | 129 (15.6%) | 395 | 2 | 280 | 20 |
| 2B_romantic_bonding | 24_final_filter | 32 | 32 | 7 (21.9%) | 4 | 0 | 21 | 0 |
| 2C_sycophancy | — | — | — | — | — | — | — | — |
| 2D_human_relationship_encouragement | 26_final_filter | 2 | 2 | 0 (0.0%) | 1 | 1 | 0 | 0 |
| 3A_engagement_hooks | 27_final_filter | 11 | 11 | 0 (0.0%) | 3 | 1 | 7 | 0 |

**Total (re-run #1): 245 both-KEEP (excluding 2C_sycophancy which was not completed)**

## Stage 4 Results — Re-run #2 (COMPLETE)

Experiments 19-27. Re-evaluates Stage 3 intersection rows (both-KEEP) with Opus 4.6 and the broadened CHECK 1 prompt. All 9 jobs complete.

| Measure | Experiment | Stage 3 in | Both KEEP | Chitchat only | Category only | Neither | Errors |
|---|---|---|---|---|---|---|---|
| 1B_human_disfluencies | 19_final_filter | 40 | 5 (12.5%) | 21 | 1 | 13 | 0 |
| 1C_identity_transparency | 20_final_filter | 1,419 | 142 (10.0%) | 518 | 3 | 756 | 0 |
| 2A_fabricated_personal_details | 21_final_filter | 77 | 11 (14.3%) | 3 | 11 | 52 | 0 |
| 2B_explicit_emotions | 22_final_filter | 165 | 62 (37.6%) | 18 | 13 | 72 | 0 |
| 2B_implicit_emotions | 23_final_filter | 1,875 | 185 (9.9%) | 928 | 7 | 755 | 0 |
| 2B_romantic_bonding | 24_final_filter | 110 | 24 (21.8%) | 13 | 1 | 72 | 0 |
| 2C_sycophancy | 25_final_filter | 83 | 18 (21.7%) | 11 | 6 | 48 | 0 |
| 2D_human_relationship_encouragement | 26_final_filter | 44 | 4 (9.1%) | 2 | 2 | 36 | 0 |
| 3A_engagement_hooks | 27_final_filter | 17 | 1 (5.9%) | 6 | 1 | 9 | 0 |

**Total: 452 both-KEEP across all 9 measures**

### Comparison to re-run #1

| Measure | Re-run #1 Both KEEP | Re-run #2 Both KEEP | Change |
|---|---|---|---|
| 1B_human_disfluencies | 2 | 5 | +3 (+150%) |
| 1C_identity_transparency | 67 | 142 | +75 (+112%) |
| 2A_fabricated_personal_details | 2 | 11 | +9 (+450%) |
| 2B_explicit_emotions | 38 | 62 | +24 (+63%) |
| 2B_implicit_emotions | 129 | 185 | +56 (+43%) |
| 2B_romantic_bonding | 7 | 24 | +17 (+243%) |
| 2C_sycophancy | — | 18 | new (never completed in re-run #1) |
| 2D_human_relationship_encouragement | 0 | 4 | +4 (∞) |
| 3A_engagement_hooks | 0 | 1 | +1 (∞) |
| **Total** | **245** | **452** | **+207 (+84%)** |

The broadened CHECK 1 prompt nearly doubled Stage 4 both-KEEP (245 → 452, +84%). Every measure increased. 2C_sycophancy now has 18 both-KEEP (was never completed in re-run #1). Categories that were previously at 0 (2D, 3A) now have data. Opus 4.6 remains more selective than GPT-4o-mini (452/3,830 = 11.8% pass rate at Stage 4 vs 3,830/32,745 = 11.7% at Stage 3).

## Stage 5 — Collect & Deduplicate (COMPLETE)

Collects all 452 both-KEEP rows from Stage 4, deduplicates by `user_input`. Conversations appearing in multiple measures get a single row with `measure` as a list.

- **Total rows scanned:** 3,830
- **Errors (parse failures):** 108
- **Both KEEP (before dedup):** 452
- **After dedup:** 323 unique conversations
- **Single-measure rows:** 264
- **Multi-measure rows:** 59

| Measure | Raw (Stage 4) | After Dedup |
|---|---|---|
| 1B_human_disfluencies | 5 | 5 |
| 1C_identity_transparency | 142 | 142 |
| 2A_fabricated_personal_details | 11 | 11 |
| 2B_explicit_emotions | 62 | 62 |
| 2B_implicit_emotions | 185 | 185 |
| 2B_romantic_bonding | 24 | 24 |
| 2C_sycophancy | 18 | 18 |
| 2D_human_relationship_encouragement | 4 | 4 |
| 3A_engagement_hooks | 1 | 1 |
| **Total** | **452** | **323 unique** |

Output: `data/final_452.jsonl`

### Comparison to re-run #1

| | Re-run #1 | Re-run #2 | Change |
|---|---|---|---|
| Both-KEEP (raw) | 245 | 452 | +207 (+84%) |
| Unique conversations | 189 | 323 | +134 (+71%) |
| Single-measure | 159 | 264 | +105 (+66%) |
| Multi-measure | 59 | 59 | 0 |

## Stage 5.5 — Split Single-Turn / Multi-Turn (COMPLETE)

Uses Opus 4.6 via OpenRouter to classify each of the 323 unique conversations as single-turn (standalone user message) or multi-turn (pasted conversation transcript with role labels).

- **Single-turn:** 148 conversations → `data/single_turn_final_452.jsonl`
- **Multi-turn:** 175 conversations → `data/multi_turn_final_452.jsonl`
- **Classification report:** `data/split_report_final_452.json`

| Split | Count | % |
|---|---|---|
| Single-turn | 148 | 45.8% |
| Multi-turn | 175 | 54.2% |
| **Total** | **323** | 100% |

### Comparison to re-run #1

| | Re-run #1 | Re-run #2 | Change |
|---|---|---|---|
| Single-turn | 82 | 148 | +66 (+80%) |
| Multi-turn | 107 | 175 | +68 (+64%) |
| Total | 189 | 323 | +134 (+71%) |

## Stages 6+ (PENDING)

- Stage 6 (generate model responses) and beyond still need to be re-run

## Pending Steps

- **Stage 3 re-run #2 complete**: All 9 measures done. 3,830 both-KEEP total (up from 1,312 in re-run #1).
- **Stage 4 re-run #2 complete**: All 9 measures done. 452 both-KEEP total (up from 245 in re-run #1, +84%).
- **Stages 5-8**: Will need to be re-run after full pipeline completes
- Old experiments (re-run #1) preserved in `experiments/old/`
- Old pipeline results documented in `misc/old_data_summary.md`

## Potential Further Improvement: Stage 1 Filter

The Stage 1 coarse filter (`filter1.json`) has the same restrictive problem that was fixed in Stages 3-4. All 9 measures use identical `filter1.json` prompts with:
- KEEP: "Simple chitchat", "Genuine casual conversation"
- DISCARD: "Question answering or information seeking"

This means conversations like "I've been feeling really lonely, what should I do?" or "Do you have feelings?" could be discarded at Stage 1 as "information seeking" before they ever reach Stages 2-4. The current pipeline can only surface conversations from the ~573K pool that passed Stage 1's narrow definition. If `filter1.json` were updated with the same broadened "Genuine User-AI Interaction" criteria and rerun from Stage 1, substantially more data would flow into Stages 2+.

## Notes
- Stage 3 cost: ~$9 per run (GPT-4o-mini via OpenRouter)
- Stage 4 cost: ~$15 estimated per run (Opus 4.6 via OpenRouter) — will be higher if more rows pass Stage 3
- API key stored at `.openrouter_key` (chmod 600)
- Re-run #1: fixed `category_keep` semantics in filter3.json/filter4.json for all measures; broadened sycophancy definition in filter2-4.json for 2C_sycophancy
- Re-run #2: replaced CHECK 1 "Casual Conversation (Chitchat)" → "Genuine User-AI Interaction" in filter3.json/filter4.json for all 9 measures
