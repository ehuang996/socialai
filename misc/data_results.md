# Seedset, Synthetic, and Final Dataset Results (Current Artifacts)

Per-measure tag totals before and after applying the union relabel rule
defined in [`relabel_synthetic.md`](relabel_synthetic.md). For each row,
the new `measure` list is the union of the original tags and any measure
where ANY of the three responder models (gpt-4o, claude-sonnet-4,
gemini-2.0-flash-001) was judged `keep=true` by Opus 4.6 under that
measure's `filter2.json` rubric.

Tag totals (the "sum" row) count tag occurrences across rows, not
distinct rows — a single row can contribute to multiple measures.

## Seedset (322 rows)

Source: `data/seedset_raw.jsonl` → `data/seedset_data.jsonl`. These are
byte-identical to the original jessetho mirror files
`data/new_seedset.jsonl` → `data/new_seedset_relabelled.jsonl`.
The relabelled file was verified against the union rule with 0
mismatches across all 322 rows.

Lineage note: `data/new_seedset.jsonl` is a manual consolidation after
the v2 preprocessing intermediate `single_turn_final_v2.jsonl`, not a
byte-identical copy of that intermediate. The checked-in v2 script covers
8 source measures; the consolidated seedset also includes `2C_deference`.
Compared with `single_turn_final_v2.jsonl`, the current raw seedset has
285 exact `user_input` overlaps, 39 omitted v2 rows, 37 newly included
rows, and 3 shared rows with an added original `2C_deference` tag.
The seedset scoring file has 5 judge parse failures out of 8,694 triples;
those failures are treated as not flagged by the union relabel rule.

| Category | Orig | Relab | Δ |
|---|---:|---:|---:|
| 1B_intentional_human_speech | 2 | 143 | +141 |
| 1B_human_pronoun | 72 | 106 | +34 |
| 1C_identity_transparency | 40 | 48 | +8 |
| 2A_fabricated_personal_information | 11 | 37 | +26 |
| 2B_emotion_expression | 29 | 112 | +83 |
| 2C_deference | 32 | 59 | +27 |
| 2C_flattery_tone | 93 | 206 | +113 |
| 2D_human_relationship_encouragement | 3 | 21 | +18 |
| 3A_engagement_hooks | 86 | 228 | +142 |
| **sum (tags, not rows)** | **368** | **960** | **+592** |

Notes:
- Avg measures/row: orig ≈ 1.14 → relab ≈ 2.98.
- 1B_intentional_human_speech gains the most relative to its starting
  size (2 → 143): the raw seedset carried very few original tags for this
  measure, and the judge frequently flags it on responses originally
  labelled for other measures.

## Synthetic (684 rows)

Source: `data/synthetic_data.jsonl` → `data/synthetic_data_relabelled.jsonl`
(main repo). Filtered from the 1,722-row `synthetic_data_full.jsonl` by
cos(`user_input`, `source_input`) ≥ 0.75 before scoring.

| Category | Orig | Relab | Δ |
|---|---:|---:|---:|
| 1B_intentional_human_speech | 97 | 450 | +353 |
| 1B_human_pronoun | 43 | 210 | +167 |
| 1C_identity_transparency | 61 | 167 | +106 |
| 2A_fabricated_personal_information | 112 | 153 | +41 |
| 2B_emotion_expression | 83 | 253 | +170 |
| 2C_deference | 105 | 218 | +113 |
| 2C_flattery_tone | 98 | 395 | +297 |
| 2D_human_relationship_encouragement | 48 | 139 | +91 |
| 3A_engagement_hooks | 37 | 291 | +254 |
| **sum (tags, not rows)** | **684** | **2,276** | **+1,592** |

Row-level summary:
- Avg measures/row: orig = 1.00 → relab = 3.33.
- Rows that gained measures: **553 (80.8%)**.
- Rows unchanged: 131 (19.2%).
- Judge error rate (parse failures): 213 / 18,468 = 1.15%. Errored
  triples are treated as "not flagged" by the union rule (conservative
  under-tag, no corruption).

## Shape comparison

Both runs show the same qualitative pattern: `1B_intentional_human_speech`,
`2C_flattery_tone`, and `3A_engagement_hooks` expand heavily, while
`2A_fabricated_personal_information` stays comparatively small,
especially in the synthetic set.

| Category | Seedset Δ | Synthetic Δ |
|---|---:|---:|
| 1B_intentional_human_speech | +141 | +353 |
| 2C_flattery_tone | +113 | +297 |
| 3A_engagement_hooks | +142 | +254 |
| 2B_emotion_expression | +83 | +170 |
| 1B_human_pronoun | +34 | +167 |
| 2C_deference | +27 | +113 |
| 1C_identity_transparency | +8 | +106 |
| 2D_human_relationship_encouragement | +18 | +91 |
| 2A_fabricated_personal_information | +26 | +41 |

`2A_fabricated_personal_information` is the smallest synthetic gainer
and one of the smaller seedset gainers, so fabricated-personal-info
behaviour is added less often by the cross-measure union relabel pass
than the high-expansion labels above.

## Final dataset after manual pass (969 rows)

Source: current `data/final_dataset.jsonl`. The in-the-wild seedset is
unchanged at 322 rows. The manual pass selects 647 of the 684 relabelled
synthetic candidates for Stage 7 evaluation, removing 37 rows. It also
edits 14 accepted synthetic prompts before they enter `final_dataset.jsonl`,
so 14 final synthetic `user_input`s no longer exactly match any
`user_input` in `data/synthetic_data_relabelled.jsonl`. The final
synthetic portion is therefore derived from the relabelled file, but is
not an exact `user_input` subset of it.

| Source | Rows | Tags | Avg tags/row |
|---|---:|---:|---:|
| In-the-wild / seedset | 322 | 960 | 2.98 |
| Synthetic after manual pass | 647 | 2,187 | 3.38 |
| **Final dataset** | **969** | **3,147** | **3.25** |

| Category | In-the-wild tags | Synthetic tags | Final tags |
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
| **sum (tags, not rows)** | **960** | **2,187** | **3,147** |

Synthetic manual-pass effect relative to the 684-row relabelled file:

| Category | Relabelled synthetic | Final synthetic | Δ |
|---|---:|---:|---:|
| 1B_intentional_human_speech | 450 | 432 | -18 |
| 1B_human_pronoun | 210 | 200 | -10 |
| 1C_identity_transparency | 167 | 161 | -6 |
| 2A_fabricated_personal_information | 153 | 148 | -5 |
| 2B_emotion_expression | 253 | 246 | -7 |
| 2C_deference | 218 | 205 | -13 |
| 2C_flattery_tone | 395 | 381 | -14 |
| 2D_human_relationship_encouragement | 139 | 132 | -7 |
| 3A_engagement_hooks | 291 | 282 | -9 |
| **sum (tags, not rows)** | **2,276** | **2,187** | **-89** |
