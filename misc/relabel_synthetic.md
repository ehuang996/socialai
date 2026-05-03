# How to relabel synthetic data (handoff)

This is a runbook for a fresh agent to relabel the synthetic set
`data/synthetic_data.jsonl` (684 rows, filtered from the 1,722-row
`data/synthetic_data_full.jsonl` by cos(`user_input`, `source_input`)
>= 0.75) using the same methodology that produced
`data/new_seedset_relabelled.jsonl` from `data/new_seedset.jsonl`.

The relabel rule is one line:

> For each row, the new `measure` list = union(original `measure`, any
> measure where ANY of three responder models was judged `keep=true`),
> sorted in canonical 1B → 3A order.

The work was previously done on the 322-row seedset and verified
byte-identical against this rule (see verification at the bottom of this
doc). No standalone relabel script exists — it was a one-off transformation
on top of the scored output. The agent recreating it for the synthetic set
needs to (a) score the synthetic data with the same pipeline, (b) apply
the union rule.

The synthetic data lives in **two repos**. Both have the same code; the
seedset outputs and DSPy cache happen to live in the jessetho mirror,
but new synth outputs go to the main repo:

- **Main** (run from here, write outputs here):
  `/project2/robinjia_875/ehuang97/socialai`
- **Mirror** (jessetho copy; holds the historical seedset outputs and
  DSPy cache; `.openrouter_key` is a symlink back to main):
  `/project2/jessetho_1732/wangzhu/socialai`

Run from `/project2/robinjia_875/ehuang97/socialai`.

---

## Files to read

### 1. Reference: how it was done on the seedset

| Role | Path | Purpose |
|---|---|---|
| Input (pre-relabel) | [`/project2/jessetho_1732/wangzhu/socialai/data/new_seedset.jsonl`](/project2/jessetho_1732/wangzhu/socialai/data/new_seedset.jsonl) | 322 rows, `{user_input, measure, synthetic, language}`. `measure` is the *original* tag list. |
| Judge results | [`/project2/jessetho_1732/wangzhu/socialai/data/seedset_scored.jsonl`](/project2/jessetho_1732/wangzhu/socialai/data/seedset_scored.jsonl) (jessetho only, 22 MB) | 322 × 3 models × 9 measures = **8,694 rows**. Each row carries `judge_output: {reasoning, keep}`. This is the ground truth that the relabel rule reads. |
| Output (post-relabel) | [`/project2/jessetho_1732/wangzhu/socialai/data/new_seedset_relabelled.jsonl`](/project2/jessetho_1732/wangzhu/socialai/data/new_seedset_relabelled.jsonl) | Same 322 rows, `measure` field rewritten by the union rule. Verified byte-equivalent. |
| Scoring script | [`/project2/jessetho_1732/wangzhu/socialai/src/evaluation/score_seedset.py`](/project2/jessetho_1732/wangzhu/socialai/src/evaluation/score_seedset.py) | Phase A (responses) + Phase B (judge). The exact script that produced `seedset_scored.jsonl`. |

### 2. Inputs for the synthetic relabel

| Role | Path |
|---|---|
| Synthetic data (to be relabelled) | [`/project2/robinjia_875/ehuang97/socialai/data/synthetic_data.jsonl`](/project2/robinjia_875/ehuang97/socialai/data/synthetic_data.jsonl) (684 rows, post cos-sim filter) |
| Synthetic data (pre-filter, for reference) | [`/project2/robinjia_875/ehuang97/socialai/data/synthetic_data_full.jsonl`](/project2/robinjia_875/ehuang97/socialai/data/synthetic_data_full.jsonl) (1,722 rows) |
| Filter rubrics (one per measure) | [`/project2/jessetho_1732/wangzhu/socialai/src/filter/measure/<measure>/filter2.json`](/project2/jessetho_1732/wangzhu/socialai/src/filter/measure/) — keys: `1B_intentional_human_speech`, `1B_human_pronoun`, `1C_identity_transparency`, `2A_fabricated_personal_information`, `2B_emotion_expression`, `2C_deference`, `2C_flattery_tone`, `2D_human_relationship_encouragement`, `3A_engagement_hooks` |
| OpenRouter API key | `/project2/robinjia_875/ehuang97/socialai/.openrouter_key` (the jessetho copy is a symlink back to here) |
| DSPy disk cache | `/project2/robinjia_875/ehuang97/socialai/cache/dspy/` — keyed on `(model, sampling kwargs, messages)`; re-runs are free given the same script. The seedset's cache lives at `/project2/jessetho_1732/wangzhu/socialai/cache/dspy/` but synth user_inputs differ, so cache hits are not expected anyway. |

### 3. Schemas (for dropping straight into a notebook)

`data/synthetic_data.jsonl` row (relevant fields):
```jsonc
{
  "user_input": "<rewritten prompt>",
  "measure": ["3A_engagement_hooks"],          // originally just the target measure
  "synthetic": true,
  "language": "English",
  "source_input": "...",                       // pre-rewrite near-miss
  "rewrite_model": "google/gemini-3.1-pro-preview",
  "response_model": "anthropic/claude-sonnet-4",
  "assistant_response": "..."                  // ONE response from response_model
}
```

`data/seedset_scored.jsonl` row (the shape the synth relabel must produce):
```jsonc
{
  "user_input": "<text>",
  "labeled_measure": ["..."],                  // original tag list
  "model_name": "gpt_4o" | "claude_sonnet_4" | "gemini_2_flash",
  "model_response": "<assistant text>",
  "measure": "3A_engagement_hooks",            // the rubric this row was judged under
  "raw_judge_response": "{ ... }",
  "judge_output": {"reasoning": "...", "keep": true|false}
}
```

---

## The relabel rule (verified)

```python
ORDER = [
    "1B_intentional_human_speech",
    "1B_human_pronoun",
    "1C_identity_transparency",
    "2A_fabricated_personal_information",
    "2B_emotion_expression",
    "2C_deference",
    "2C_flattery_tone",
    "2D_human_relationship_encouragement",
    "3A_engagement_hooks",
]

def relabel(orig_measure: list[str], scored_rows_for_input: list[dict]) -> list[str]:
    flagged = {r["measure"] for r in scored_rows_for_input
               if (r.get("judge_output") or {}).get("keep") is True}
    union = set(orig_measure) | flagged
    return sorted(union, key=ORDER.index)
```

This was applied per `user_input`. Verified on all 322 seedset rows: 322
match, 0 mismatch. The seedset tag totals before/after are:

| measure | orig | relab |
|---|---:|---:|
| 1B_intentional_human_speech | 2 | 143 |
| 1B_human_pronoun | 72 | 106 |
| 1C_identity_transparency | 40 | 48 |
| 2A_fabricated_personal_information | 11 | 37 |
| 2B_emotion_expression | 29 | 112 |
| 2C_deference | 32 | 59 |
| 2C_flattery_tone | 93 | 206 |
| 2D_human_relationship_encouragement | 3 | 21 |
| 3A_engagement_hooks | 86 | 228 |

Use these numbers as a sanity reference — the synth relabel will produce
its own table.

---

## Procedure for the synthetic set

### Step 1 — Score the synthetic file

Reuse `src/evaluation/score_seedset.py` (already in the main repo). It
already takes `--input` / `--output` / `--key` flags, so no copy is
needed. The script reads `.openrouter_key`, runs Phase A across the
three response models (`gpt-4o`, `claude-sonnet-4`,
`gemini-2.0-flash-001`) and Phase B (Opus 4.6 judge under all 9
measures), writing one row per `(user_input, model, measure)` triple —
`684 × 3 × 9 = 18,468` triples.

**Methodological note**: do **not** reuse the `assistant_response`
already in `synthetic_data.jsonl`. That response was generated by *one*
rotating model (per `response_model`) and would bias the judge — only
that model's behaviour would be measured. The seedset relabel scored
all 3 models independently. To stay consistent, regenerate responses on
the same 3 models, ignoring the recorded `response_model`.

Invoke directly with the synth paths:

```bash
cd /project2/robinjia_875/ehuang97/socialai
uv run python src/evaluation/score_seedset.py \
    --input data/synthetic_data.jsonl \
    --output data/synthetic_scored.jsonl \
    --key .openrouter_key
```

Everything else (models, judge, DSPy cache, `N_WORKERS=4`,
`ACTIVE_MEASURES`) stays identical.

Expected wall-time: scaled from the seedset, the synth scoring is
~2.1× more triples (18,468 vs 8,694). Budget a couple of hours and watch
for OpenRouter rate limits.

### Step 2 — Apply the relabel rule

After Step 1 finishes, write
`/project2/robinjia_875/ehuang97/socialai/data/synthetic_data_relabelled.jsonl`
with the rule above, joined on `user_input`. Pseudocode:

```python
import json
from collections import defaultdict

ORDER = [...]  # see above

SCORED_PATH = "/project2/robinjia_875/ehuang97/socialai/data/synthetic_scored.jsonl"
INPUT_PATH  = "/project2/robinjia_875/ehuang97/socialai/data/synthetic_data.jsonl"
OUTPUT_PATH = "/project2/robinjia_875/ehuang97/socialai/data/synthetic_data_relabelled.jsonl"

scored = defaultdict(list)
for line in open(SCORED_PATH):
    r = json.loads(line)
    scored[r["user_input"]].append(r)

with open(OUTPUT_PATH, "w") as fout:
    for line in open(INPUT_PATH):
        row = json.loads(line)
        flagged = {r["measure"] for r in scored.get(row["user_input"], [])
                   if (r.get("judge_output") or {}).get("keep") is True}
        union = set(row["measure"]) | flagged
        row["measure"] = sorted(union, key=ORDER.index)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
```

### Step 3 — Sanity checks

- Row count: `wc -l /project2/robinjia_875/ehuang97/socialai/data/synthetic_data_relabelled.jsonl` should be `684`.
- All `user_input` from the original file are present (no drops).
- Per-measure tag totals printed (compare to the all-rewrites-target ≈ 76/measure
  baseline; expect spread similar in shape to the seedset table above).
- Spot-check a 3A_engagement_hooks row: many synth rows that were
  *only* tagged 3A may pick up 2C_flattery_tone or 2B_emotion_expression
  after relabel, since rewrites often graft on multiple bait patterns.
- For each row, `len(row["measure"]) >= len(orig["measure"])` (the union
  can only grow).

---

## Caveats

1. **Existing `assistant_response` in synth rows is for QC only** —
   it is the response the synth pipeline saw at Step 4 of generation.
   Step 1 above regenerates fresh responses across all 3 models for
   relabelling so the judge sees a balanced sample.
2. **DSPy cache reuse** — synth runs use a fresh cache at
   `/project2/robinjia_875/ehuang97/socialai/cache/dspy/`. The seedset's
   cache (in jessetho) won't help anyway since synth `user_input`
   strings differ. All 18,468 judge calls in Step 1 will be cache
   misses on first run.
3. **Prompt-leakage rows in the pre-filter file** — the 1,722-row
   `synthetic_data_full.jsonl` contains rewriter meta-reasoning
   bleed-through ("Wait, I need to rewrite..."). The exact leakage strings
   do not appear in the current 684-row `synthetic_data.jsonl`; rerun this
   check if the synthetic file is regenerated.
4. **Two near-clones to filter first** — two synth rows in the
   pre-filter `synthetic_data_full.jsonl` have cos(revised, nearest
   few-shot trigger) ≥ 0.85 (one is a verbatim copy of the trigger
   `"i missed you"`). They may or may not survive the cos(`user_input`,
   `source_input`) ≥ 0.75 filter that produced the 684-row file —
   double-check before relabel if it matters.
