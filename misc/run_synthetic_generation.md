# Running synthetic generation (Stage 6)

How to generate the synthetic dataset from `data/seedset_data.jsonl` (target
triggers) and `data/near_misses.jsonl` (rewrite candidates), and how to
combine + clean the per-measure outputs into a single `synthetic_data.jsonl`.

## Prerequisites

1. **OpenRouter API key** at `.openrouter_key` (single line, no whitespace).
2. **Python deps** installed:
   ```bash
   uv sync
   ```
3. **Inputs in place** at the project root:
   - `data/seedset_data.jsonl` — confirmed-trigger anchors, schema
     `{user_input, measure, synthetic, language}`. Must have ≥3 rows for
     every measure you want to generate.
   - `data/near_misses.jsonl` — Stage 3 XOR pool, same schema. The pipeline
     pulls per-principle (1x, 2x, 3x) from this file as candidates.

## The 9 measures

| Folder name (= `--measure` value) | Principle |
|---|---|
| `1B_intentional_human_speech` | 1x |
| `1B_human_pronoun` | 1x |
| `1C_identity_transparency` | 1x |
| `2A_fabricated_personal_information` | 2x |
| `2B_emotion_expression` | 2x |
| `2C_deference` | 2x |
| `2C_flattery_tone` | 2x |
| `2D_human_relationship_encouragement` | 2x |
| `3A_engagement_hooks` | 3x |

## Single-measure run (smoke test)

Use this first on one measure to confirm everything wires up before
launching the full set:

```bash
uv run python src/synthetic_generation/generate.py \
    --measure 2C_flattery_tone \
    --target 5 \
    --batch_size 3 \
    --target_triggers_path data/seedset_data.jsonl \
    --near_misses_path data/near_misses.jsonl \
    --key .openrouter_key
```

Expected at the end of the run:
- `data/synthetic/2C_flattery_tone.jsonl` has ≥5 rows, each with
  `synthetic: true`.
- `rewrite_model` and `response_model` rotate across the three
  vendor-pair combos.
- A `data/synthetic/2C_flattery_tone.rejected.jsonl` audit file records
  every candidate that was attempted but didn't pass Steps 4-5.

## All-9 production run (parallel)

Launches all 9 measures concurrently with `--concurrency 6` per process
(54 in-flight API calls system-wide — comfortably under OpenRouter rate
limits but enough to keep things moving):

```bash
mkdir -p logs
for m in 1B_intentional_human_speech 1B_human_pronoun 1C_identity_transparency \
         2A_fabricated_personal_information 2B_emotion_expression \
         2C_deference 2C_flattery_tone 2D_human_relationship_encouragement \
         3A_engagement_hooks; do
    uv run python src/synthetic_generation/generate.py \
        --measure "$m" \
        --target 200 \
        --concurrency 6 \
        --target_triggers_path data/seedset_data.jsonl \
        --near_misses_path data/near_misses.jsonl \
        --key .openrouter_key \
        > "logs/synth_$m.log" 2>&1 &
done
wait
echo "all 9 measures complete"
```

Outputs:
- `data/synthetic/<measure>.jsonl` — accepted rows, target = 200 per measure.
- `data/synthetic/<measure>.rejected.jsonl` — full audit of dropped rows
  with `reject_stage` indicating where they failed.
- `logs/synth_<measure>.log` — per-measure stdout, including the final
  `=== Run summary ===` block.

Resumption is automatic: if a process is killed mid-run, just re-run the
same command — it counts existing rows in the output JSONL and continues
toward target. Use `>>` instead of `>` on the redirect if you want to
preserve prior log content.

## Combine + clean (Stage 5 Phase 1 + Phase 3)

After all 9 measure files are at target, merge them into a single
`synthetic_data.jsonl` and run dedup:

```bash
uv run python - << 'PYEOF'
import json
from collections import OrderedDict, Counter
from pathlib import Path

MEASURES = [
    "1B_intentional_human_speech", "1B_human_pronoun", "1C_identity_transparency",
    "2A_fabricated_personal_information", "2B_emotion_expression",
    "2C_deference", "2C_flattery_tone", "2D_human_relationship_encouragement",
    "3A_engagement_hooks",
]

# Combine
all_rows = []
for m in MEASURES:
    with open(f"data/synthetic/{m}.jsonl") as f:
        all_rows.extend(json.loads(line) for line in f if line.strip())

# Phase 1: user_input dedup (collapses any cross-measure exact-match duplicates).
seen = OrderedDict()
for r in all_rows:
    seen.setdefault(r["user_input"], r)
phase1 = list(seen.values())

# Phase 3: cosine similarity dedup at threshold 0.85.
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode([r["user_input"] for r in phase1], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
sim = embs @ embs.T
keep = []
for i in range(len(phase1)):
    if not any(sim[i, j] >= 0.85 for j in keep):
        keep.append(i)
deduped = [phase1[i] for i in keep]

# Force single-measure label per row (drop any merged-list artifacts).
for r in deduped:
    if isinstance(r.get("measure"), list) and len(r["measure"]) > 1:
        r["measure"] = [r["measure"][0]]

with open("data/synthetic_data.jsonl", "w") as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(deduped)} rows to data/synthetic_data.jsonl "
      f"(combined {len(all_rows)} → phase1 {len(phase1)} → phase3 {len(deduped)})")
PYEOF
```

## Tunable knobs

| Flag | Default | When to change |
|---|---|---|
| `--target` | 200 | Smoke-test with 5; ramp to 200+ for production |
| `--concurrency` | 10 | Drop to 6 when running all 9 in parallel to avoid OpenRouter rate limits |
| `--batch_size` | 20 | Larger = fewer rounds but higher latency between writes |
| `--seed` | 42 | Change to get a different sample of few-shots (re-runs with same seed are cache hits) |
| `--max_rows` | None | Debug cap that overrides `--target` |
| `--judge_model` | `anthropic/claude-opus-4-6` | Step 4 verifier — rarely change |
| `--naturalness_model` | `anthropic/claude-opus-4-6` | Step 5 verifier — rarely change |

The vendor-pair model lists (`REWRITE_MODELS`, `RESPONSE_MODELS`) are
hard-coded in `src/synthetic_generation/generate.py` and validated at
startup (`validate_model_vendor_split`). Edit the source if you need to
swap models.

## Result of the most recent production run

Run dates: **2026-04-23** (4 measures) and **2026-04-27 → 2026-04-28**
(5 measures, with one wrapper kill + resume).

Per-measure outputs at `--target 200` each:

| Measure | Accepted | Rejected | Survival |
|---|---|---|---|
| 2D_human_relationship_encouragement | 200 | 400 | 33.3% |
| 1B_human_pronoun | 200 | 643 | 23.7% |
| 1C_identity_transparency | 200 | 619 | 24.4% |
| 3A_engagement_hooks | 200 | 613 | 24.6% |
| 1B_intentional_human_speech | 200 | 705 | 22.1% |
| 2C_flattery_tone | 200 | 812 | 19.8% |
| 2B_emotion_expression | 200 | 949 | 17.4% |
| 2C_deference | 200 | 1,910 | 9.5% |
| 2A_fabricated_personal_information | 200 | 2,332 | 7.9% |

**Subtotal**: 1,800 accepted across 9 measures.

After combining and running Phase 1 + Phase 3 dedup
(cosine ≥ 0.85, all-MiniLM-L6-v2):

- Combined raw: 1,800 rows.
- Phase 1 (user_input dedup): 1,800 → 1,799 (1 cross-measure duplicate).
- Phase 3 (semantic dedup): 1,799 → **1,722 kept**, 77 dropped.

**`data/synthetic_data.jsonl`: 1,722 rows**, every row tagged with exactly
one measure. Per-measure: 1B_intentional_human_speech 200,
2A 198, 3A 196, 2B 194, 1B_pronoun 193, 2C_deference 192, 1C 190,
2D 182, 2C_flattery 177.

### Wall-clock time

- **Per-measure runtime** (parallel, `--concurrency 6`): roughly 1-4 hours
  depending on survival rate. Easy measures (>20% survival, e.g. 1B_pronoun,
  3A) finish in ~1.5-2 h. Hard measures (<10% survival, e.g. 2A, 2C_deference)
  take 4-6 h.
- **Bottleneck**: 2A_fabricated_personal_information (~7.9% survival) was
  the longest at ~6 hours of API time. Total wall-clock to complete all 9
  measures in parallel = max(per-measure runtimes) ≈ **~6 hours** under
  good rate-limit conditions.
- **OpenRouter rate-limit incidents**: during the 5-measure run, Google
  AI Studio temporarily 504/429'd Gemini-2.0-Flash for ~10 minutes, which
  dragged 2A's effective survival rate down further and added an hour or
  two. The retry queue absorbed it but throughput dropped.
- **Combine + dedup pass**: <1 minute total (1,800-row embedding +
  similarity matrix is trivial for all-MiniLM-L6-v2).

### Cost (rough)

- ~12,000-15,000 OpenRouter API calls across the 9 runs (3 calls per
  attempted candidate × ~4,000 candidates total, averaged).
- Cache hit rate ≈ 11-15% (low because the rewrite model rotation
  across 3 frontier models meant lots of cold cache entries on first run).
- Estimated total spend: **~$200-280** at current OpenRouter pricing,
  dominated by the Step 2 Opus rewrites and Step 4/5 Opus verifications.
