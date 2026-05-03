# Pipeline Results Summary

9 measures across 4 filtering stages, then preprocessing (Stage 5),
synthetic generation + relabel (Stage 6), and a refactored evaluation
pipeline (Stage 7.1 generate → 7.2 judge → 7.3 analyze).

The schema collapsed from an earlier 11-measure version (3 separate 2B
sub-measures, plus `2A_fabricated_personal_details`, `2C_sycophancy`,
`1B_human_disfluencies`). Renames and consolidations:

| Old name | New name | What happened |
|---|---|---|
| `1B_human_disfluencies` | `1B_intentional_human_speech` | renamed only — Stage 1–4 outputs reused as-is |
| `2A_fabricated_personal_details` | `2A_fabricated_personal_information` | renamed + Stage 2–4 rerun with new prompts (jessetho mirror) |
| `2B_explicit_emotions` + `2B_implicit_emotions` + `2B_romantic_bonding` | `2B_emotion_expression` | three sub-measures merged into one + Stage 2–4 rerun (jessetho mirror) |
| `2C_sycophancy` | `2C_flattery_tone` | renamed + Stage 2–4 rerun (jessetho mirror) |
| `2C_deference` | (same name) | Stage 3–4 rerun with new prompts (main repo); Stage 2 unchanged |
| `2D_human_relationship_encouragement` | (same name) | Stage 2–4 rerun (jessetho mirror) |
| `1B_human_pronoun`, `1C_identity_transparency`, `3A_engagement_hooks` | (same name) | Stage 1–4 outputs reused as-is |

**Two-repo layout.** Re-run measures live in the jessetho mirror; reused
measures and the consolidated downstream artifacts live in the main repo.

- Main: `/project2/robinjia_875/ehuang97/socialai/experiments/`
- Mirror: `/project2/jessetho_1732/wangzhu/socialai/experiments/` (different folder numbering)

## Stage 0 — Download WildChat

Downloaded `allenai/WildChat-4.8M` (the **public, non-gated** release —
~3.2M conversations; the full 4.8M lives in the request-only
`allenai/WildChat-4.8M-Full` repo), filtered English, deduplicated by
`conversation_hash` → `data/wildchat_raw.jsonl`.

- **Total rows from HF:** 3,199,860
- **After English filter:** 1,679,371
- **After dedup:** 1,442,077

## Stage 1 — Coarse Filter (vLLM, Qwen3-VL-8B)

Identical `filter1.json` across all 9 measures (domain gate restricting
to genuine human-AI chitchat).

- **Input:** WildChat English conversations
- **Output (main):** 560,969 chitchat conversations →
  `experiments/00_coarse_filter/`

## Stage 2 — Low-Quality Filter (vLLM, Qwen3-VL-8B)

Category-specific judge using `filter2.json` rubrics. All 9 listed
Stage 2 result files contain 560,969 scored rows. The 4 rerun measures
were executed in the jessetho mirror, but their concatenated result
files have the same row count as the main Stage 1 coarse output.

| # | Measure | Source | keep=True | % of input |
|---|---|---|---:|---:|
| 1 | 1B_intentional_human_speech | main `01_low_quality_filter` | 4,081 | 0.73% |
| 2 | 1B_human_pronoun | main `02_low_quality_filter` | 8,291 | 1.48% |
| 3 | 1C_identity_transparency | main `03_low_quality_filter` | 6,855 | 1.22% |
| 4 | 2A_fabricated_personal_information | mirror `25_low_quality_filter` | 404,591 | 72.1% |
| 5 | 2B_emotion_expression | mirror `16_low_quality_filter` | 3,233 | 0.58% |
| 6 | 2C_deference | main `08_low_quality_filter` | 15,082 | 2.69% |
| 7 | 2C_flattery_tone | mirror `19_low_quality_filter` | 3,179 | 0.57% |
| 8 | 2D_human_relationship_encouragement | mirror `22_low_quality_filter` | 214 | 0.04% |
| 9 | 3A_engagement_hooks | main `11_low_quality_filter` | 3,803 | 0.68% |

**Note on 2A.** 404,591 / 560,969 ≈ 72% pass-through reflects the broad
Stage 2 `2A_fabricated_personal_information` rubric; the GPT-4o-mini
Stage 3 judge then compresses it down to 78. None of the other rerun
measures behaves this way.

## Stage 3 — High-Quality Filter (GPT-4o-mini via OpenRouter)

Dual-check: (1) genuine user-AI interaction? (2) does it match the
category? Only rows with both `chitchat_keep=true` AND `category_keep=true`
advance to Stage 4. Output schema in this stage and Stage 4 nests the
judge output under `model_responses[<judge_key>].raw_response` (a JSON
string with the four fields), not flat top-level keys.

| # | Measure | Source | Stage 2 in | Both KEEP | % |
|---|---|---|---:|---:|---:|
| 1 | 1B_intentional_human_speech | main `12_high_quality_filter` | 4,081 | 46 | 1.1% |
| 2 | 1B_human_pronoun | main `13_high_quality_filter` | 8,291 | 234 | 2.8% |
| 3 | 1C_identity_transparency | main `14_high_quality_filter` | 6,855 | 1,419 | 20.7% |
| 4 | 2A_fabricated_personal_information | mirror `26_high_quality_filter` | 404,591 | 78 | 0.02% |
| 5 | 2B_emotion_expression | mirror `28_high_quality_filter` | 3,233 | 392 | 12.1% |
| 6 | 2C_deference | main `19_high_quality_filter` (rerun) | 15,082 | 546 | 3.6% |
| 7 | 2C_flattery_tone | mirror `20_high_quality_filter` | 3,179 | 293 | 9.2% |
| 8 | 2D_human_relationship_encouragement | mirror `23_high_quality_filter` | 214 | 70 | 32.7% |
| 9 | 3A_engagement_hooks | main `22_high_quality_filter` | 3,803 | 993 | 26.1% |

For 2B_emotion_expression there are two runs in the mirror — `17/18`
(initial) and `28/29` (rerun, Apr 26 ~11:00). The `28/29` numbers are
the current values used downstream and shown in the table; `17/18`
(289 → 74) is superseded.

## Stage 4 — Final Filter (Claude Opus 4.6 via OpenRouter)

Re-evaluates Stage 3 intersection rows with Opus 4.6 using `filter4.json`.

| # | Measure | Source | Stage 3 in | Both KEEP | % |
|---|---|---|---:|---:|---:|
| 1 | 1B_intentional_human_speech | main `23_final_filter` | 46 | 10 | 21.7% |
| 2 | 1B_human_pronoun | main `24_final_filter` | 234 | 80 | 34.2% |
| 3 | 1C_identity_transparency | main `25_final_filter` | 1,419 | 142 | 10.0% |
| 4 | 2A_fabricated_personal_information | mirror `27_final_filter` | 78 | 13 | 16.7% |
| 5 | 2B_emotion_expression | mirror `29_final_filter` | 392 | 91 | 23.2% |
| 6 | 2C_deference | main `30_final_filter` (rerun) | 546 | 43 | 7.9% |
| 7 | 2C_flattery_tone | mirror `21_final_filter` | 293 | 144 | 49.1% |
| 8 | 2D_human_relationship_encouragement | mirror `24_final_filter` | 70 | 24 | 34.3% |
| 9 | 3A_engagement_hooks | main `33_final_filter` | 993 | 500 | 50.4% |

**Total: 1,047 both-KEEP across all 9 measures** (pre-dedup, before
preprocessing).

## Stage 5 — Data Preprocessing + Seedset Consolidation

The current seedset files are the authoritative record for this stage.
`data_preprocessing_v2.py` records the v2 preprocessing procedure and
produced the jessetho intermediate files `final_v2.jsonl` and
`single_turn_final_v2.jsonl`; the final raw seedset was then manually
consolidated into the jessetho mirror's `data/new_seedset.jsonl`, copied
locally as `data/seedset_raw.jsonl`.

Differences from v1:

1. Reads from a hand-curated list of `(source_path, measure)` pairs
   spanning **both** project locations (main + mirror), so re-run
   new-prompt measures mix with measures that didn't get rerun. The
   checked-in script's source list covers 8 measures; the final seedset
   consolidation also brings in the `2C_deference` rerun.
2. **No chitchat veto.** v1 dropped any `user_input` whose
   `chitchat_keep=false` was returned by *any* measure's judge; v2
   keeps every both-keep row regardless.
3. Batched single/multi classification: one Opus 4.6 call labels 10
   messages at a time (~10× fewer API calls), DSPy disk cache.

The direct v2 script output `single_turn_final_v2.jsonl` had 324 rows
and 370 original tags. The manually consolidated `new_seedset.jsonl` /
`seedset_raw.jsonl` has 322 rows and 368 original tags: 285 exact
`user_input`s overlap with `single_turn_final_v2`, 39 v2 rows were
omitted, 37 rows were newly included, and 3 shared rows gained an
original `2C_deference` tag. After the union relabel pass described
below, the current canonical in-the-wild portion of `final_dataset.jsonl`
is `data/seedset_data.jsonl` (the same 322 rows, 960 relabelled tags).

### Raw seedset measure distribution (322 rows, schema `{user_input, measure, synthetic, language}`)

| Measure | Tags |
|---|---:|
| 1B_intentional_human_speech | 2 |
| 1B_human_pronoun | 72 |
| 1C_identity_transparency | 40 |
| 2A_fabricated_personal_information | 11 |
| 2B_emotion_expression | 29 |
| 2C_deference | 32 |
| 2C_flattery_tone | 93 |
| 2D_human_relationship_encouragement | 3 |
| 3A_engagement_hooks | 86 |
| **sum (tags, not rows)** | **368** |

Avg measures/row ≈ 1.14. The 1B_intentional_human_speech count (2) is
particularly low; it expands sharply after the Stage 6 relabel pass
(see below).

## Stage 6 — Synthetic Generation + Union Relabel

Rewrites near-miss user inputs (from the Stage 3 XOR pool) into
on-target candidates anchored to the seedset few-shot pool.

- **Input pool:** `data/near_misses.jsonl` (11,520 rows, pooled by
  leading principle digit).
- **Raw rewrites:** `data/synthetic_data_full.jsonl` (1,722 rows).
- **Cosine-similarity filter** on `cos(user_input, source_input) ≥ 0.75`
  → `data/synthetic_data.jsonl` (684 rows).
- **Re-judge / union relabel** (see [`relabel_synthetic.md`](relabel_synthetic.md)):
  each row scored across all 9 measures by 3 responder models (gpt-4o,
  claude-sonnet-4, gemini-2.0-flash-001), Opus 4.6 judge. The new
  `measure` list = union(original tags, any measure where ANY of the
  three responders is judged `keep=true`). Output:
  `data/synthetic_data_relabelled.jsonl` (684 rows).
- **Manual pass:** selects 647 of the 684 relabelled synthetic rows for
  `data/final_dataset.jsonl` (37 rows removed). The source
  `data/synthetic_data.jsonl` and `data/synthetic_data_relabelled.jsonl`
  files remain the 684-row pre-manual-pass artifacts. The manual pass also
  edits 14 accepted synthetic prompts, so the 647 synthetic rows in
  `final_dataset.jsonl` are not an exact `user_input` subset of
  `synthetic_data_relabelled.jsonl`.
- The judge output for each `(input, model, measure)` triple lives in
  `data/synthetic_scored.jsonl` (3 × 684 × 9 = 18,468 rows). The same
  scoring pass on the seedset lives in `data/seedset_eval.jsonl`
  (3 × 322 × 9 = 8,694 rows).

### Per-measure relabel deltas

Tag totals count tag occurrences across rows, not distinct rows. See
[`data_results.md`](data_results.md) for the full per-measure
breakdown.

| Category | Seedset orig → relab (Δ) | Synthetic orig → relab (Δ) |
|---|---:|---:|
| 1B_intentional_human_speech | 2 → 143 (+141) | 97 → 450 (+353) |
| 1B_human_pronoun | 72 → 106 (+34) | 43 → 210 (+167) |
| 1C_identity_transparency | 40 → 48 (+8) | 61 → 167 (+106) |
| 2A_fabricated_personal_information | 11 → 37 (+26) | 112 → 153 (+41) |
| 2B_emotion_expression | 29 → 112 (+83) | 83 → 253 (+170) |
| 2C_deference | 32 → 59 (+27) | 105 → 218 (+113) |
| 2C_flattery_tone | 93 → 206 (+113) | 98 → 395 (+297) |
| 2D_human_relationship_encouragement | 3 → 21 (+18) | 48 → 139 (+91) |
| 3A_engagement_hooks | 86 → 228 (+142) | 37 → 291 (+254) |
| **sum (tags)** | **368 → 960 (+592)** | **684 → 2,276 (+1,592)** |

`2A_fabricated_personal_information` is the smallest synthetic gainer
and one of the smaller seedset gainers, so fabricated-personal-info
behavior is added less often by the cross-measure union relabel pass
than the high-expansion labels such as `1B_intentional_human_speech`,
`2C_flattery_tone`, and `3A_engagement_hooks`.

### Final dataset (Stage 6 output → Stage 7 input)

`data/final_dataset.jsonl` = relabelled in-the-wild seedset (322) +
manually accepted relabelled synthetic (647) = **969 rows**. Schema:
`{user_input, measure, synthetic, language}`. The final file is the
authoritative Stage 7 input; `wc -l` may show 968 if the file lacks a
trailing newline, but JSON parsing yields 969 records.

| Measure | In-the-wild tags | Synthetic tags | Total tags |
|---|---:|---:|---:|
| 1B_intentional_human_speech | 143 | 432 | 575 |
| 1B_human_pronoun | 106 | 200 | 306 |
| 1C_identity_transparency | 48 | 161 | 209 |
| 2A_fabricated_personal_information | 37 | 148 | 185 |
| 2B_emotion_expression | 112 | 246 | 358 |
| 2C_deference | 59 | 205 | 264 |
| 2C_flattery_tone | 206 | 381 | 587 |
| 2D_human_relationship_encouragement | 21 | 132 | 153 |
| 3A_engagement_hooks | 228 | 282 | 510 |
| **sum (tags)** | **960** | **2,187** | **3,147** |

Avg measures/row ≈ 3.25 (969 rows, 3,147 tags).

## Stage 7 — Refactored Evaluation Pipeline

The previous numbering (Stage 6.1 generate, Stage 7.1 single-turn judge,
Stage 7.2 multi-turn, Stage 8.1 analyze) has been collapsed into three
top-level scripts under `src/evaluation/`. Multi-turn is no longer in
the active pipeline; everything operates on `data/final_dataset.jsonl`.

| Stage | Script | Input | Output |
|---|---|---|---|
| 7.1 generate | `src/evaluation/generate_responses.py` + `src/evaluation/generate_local_responses.py` | `data/final_dataset.jsonl` | `data/eval_responses.jsonl` |
| 7.2 judge | `src/evaluation/judge_responses.py` | `data/eval_responses.jsonl` | `data/eval_judge_results.jsonl` |
| 7.3 analyze | `src/evaluation/analyze_results.py` | `data/eval_judge_results.jsonl` | `data/eval_figures/` |

Wrapper shell scripts: `scripts/run_stage7_1_generate.sh`,
`scripts/run_stage7_1_local_qwen.sh`, and `scripts/run_stage7_2_judge.sh`.
They wrap the Python entry points with the canonical input/output paths.
Common helpers live in
`src/evaluation/_eval_common.py` (`ALL_MEASURES`, judge prompt loader,
parse helpers).

### Stage 7.1 — Generate Model Responses

25 model evaluations spanning 6 providers, partitioned by role. The API/direct
launcher covers 23 models; the local-HF add-on serves Qwen3 1.7B and 4B with
vLLM on CARC GPUs and merges them into the same `eval_responses.jsonl` schema.

- **Target (3):** `gemini_2_flash_001`, `gpt_4o`, `claude_sonnet_4`
- **Rewriting (3):** `gemini_3_1_pro`, `gpt_5_4`, `claude_opus_4_6`
- **Frontier (19):** GPT 5.5, Claude Opus 4.7, three Opus 4.6
  thinking-budget variants (`t2k`, `t5k`, `t10k`), Gemini 3 Flash, three
  Grok variants, GPT-4o-mini, three DeepSeek variants, Qwen 3.6 Max
  preview + five Qwen3 sizes (1.7B, 4B, 8B, 14B, 32B). Qwen3 1.7B and 4B
  are local-HF add-ons because they were not usable through OpenRouter
  during setup; Qwen3 0.6B remains excluded.

Caching/resumption: the code requests DSPy disk caching at `cache/dspy/`.
When disk caching is active, the three Opus 4.6 thinking-budget variants
get distinct cache entries because `extra_body` is part of the request
payload. File-based resumption is the authoritative resume layer:
already-written rows in the canonical output or shard/part files are
skipped before forming new model calls. `--rollout_id` forces fresh
generations while still writing to cache.

Total scale: 25 × 969 = **24,225 generations** at full coverage
(22,287 API/direct + 1,938 local HF).

The active OpenRouter slugs were checked during setup against OpenRouter
endpoint availability; missing required key sections still hard-fail at
startup.

### Stage 7.2 — LLM-as-Judge

For every row produced by 7.1, every model response is judged only
against the row's labelled `measure` list. Multi-label rows therefore
produce one judge call per (input, labelled measure, model) triple.
Judge: Anthropic Claude Opus 4.6 via OpenRouter (no thinking,
temperature=0, 3-attempt retry on parse failures with rollout-id bump).

Total scale after the local-Qwen merge: 3,147 labelled input-measure pairs
× 25 models = **78,675 judge calls** at full coverage.

Output schema: one row per (user_input, measure, model_name) triple
with `judge_output: {reasoning, keep}`. Errored rows preserve a
`raw_output` so partial failures don't corrupt downstream aggregates.

### Stage 7.3 — Analysis

`src/evaluation/analyze_results.py` produces 15 figures and a
`summary_tables.md` in `data/eval_figures/`. Model families and
display order are defined in `MODEL_FAMILIES` / `MODEL_ORDER` at the
top of the file; the three Opus 4.6 thinking-budget variants share the
Anthropic family but are plotted as separate models, except in the
"per-family generation" line plot (fig4) where only the no-thinking
variant counts as the Opus 4.6 generation point.

`MEASURE_SHORT` provides the figure-friendly labels: `1B Speech`,
`1B Pronoun`, `1C Identity`, `2A Fabrication`, `2B Emotion`,
`2C Deference`, `2C Flattery`, `2D Relationship`, `3A Engagement`.

**Status.** Stage 7.1 generation has started but is not yet finalized in
the canonical merged output. The worktree currently contains partial API
shard outputs and completed local-Qwen part files; the final destination
after merge remains `data/eval_responses.jsonl`. Stage 7.2 and 7.3 have
not yet produced final artifacts: `data/eval_judge_results.jsonl` is not
present, and `data/eval_figures/` is not populated with current
evaluation results.

## Auxiliary: Human Annotation

8 annotators (bill, claude, eric, johnny, mo, nate, ravi, ziyi)
each annotated 30 examples from a single-turn evaluation result file.

- **Total annotations:** 240 (8 × 30)
- **Fields:** `example_index, user_input, measure, model, model_response,
  guidelines_correct, is_instance, other_measures, notes`
- **Location:** `data/annotations/`

## Auxiliary: Verification Pipeline

Re-ran all 4 filter stages on 104 single-turn examples to validate
filter consistency.

- **Input:** `data/verify/single_turn_104_prepared.jsonl` (104 rows)
- **Output:** `data/verify/verify_final.jsonl`
- **Driver:** `scripts/run_verify.sh`, `scripts/run_verify_s2.sh`,
  `scripts/run_verify_s2_s4.sh` — submit measures in parallel with
  SLURM dependency chaining (stages 1→2→3→4)
