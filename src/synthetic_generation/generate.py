"""Stage 6 — Synthetic data generation (single-pass, flat).

Per measure, the pipeline makes a single pass through the near-miss pool
and stops when either (a) `--target` surviving rows are written, or
(b) the near-miss pool is drained. No depth levels, no recursive
self-feeding — every synthetic row is anchored by real target_triggers.
This is the intentional simplification: the near-miss pool (2,397 for
principle 1x, 5,403 for 2x, 762 for 3x) is large enough that target row
counts in the low hundreds never drain it, so a multi-level scheme was
unnecessary complexity.

Steps (per candidate):
  1. Sample — 3 real target triggers (per measure), 3 near-misses (per
     principle, excluding the current candidate), 1 candidate (the
     current near-miss). Per-candidate deterministic RNG seeded on
     (args.seed, round_idx, candidate_idx) so reruns cache.
  2. Rewrite (frontier model, temp 0.9) — few-shot-driven rewrite of the
     candidate. Rotated by `candidate_idx % 3` across GPT-5.4 /
     Gemini-3.1-Pro / Claude-Opus-4.6 (inner cycle).
  3. Response (equal-performing older model, temp 0.7) — rotated
     INDEPENDENTLY from Step 2 via `(candidate_idx // 3) % 3` across
     GPT-4o / Gemini-2.0-Flash / Claude-Sonnet-4 (outer cycle). Over
     every 9 consecutive candidates all 3×3=9 (rewriter, responder)
     combinations are exercised exactly once, giving balanced coverage
     of cross-vendor prompt-style interactions.
  4. Judge (Opus 4.6, temp 0) — filter2.json rubric; discard if keep=false.
  5. Naturalness (Opus 4.6, temp 0) — rank {the 3 triggers from Step 1,
     the rewritten candidate} from most to least natural; discard if the
     candidate ranks last.
  6. Emit — write with `synthetic: true`.

Inputs:
  - Target triggers: `data/target_triggers.jsonl` (real, per-measure match).
    Must have at least `REQUIRED_TARGET_TRIGGERS` (= 3) rows for the measure.
  - Near misses: `data/near_misses.jsonl`, per-principle match (1x/2x/3x,
    from the leading char of the measure name). Doubles as the candidate
    list and the few-shot near-miss anchors.

Principle groups:
    1 — 1B_intentional_human_speech, 1B_human_pronoun, 1C_identity_transparency
    2 — 2A_*, 2B_*, 2C_*, 2D_*
    3 — 3A_engagement_hooks

All API calls are cached on disk via dspy.LM under `cache/dspy/` keyed on
(model, messages, sampling kwargs); few-shot sampling is deterministic given
`--seed`, so reruns are free and reproducible.

Usage:
    uv run python src/synthetic_generation/generate.py \
        --measure 2C_sycophancy \
        --key .openrouter_key \
        --target 200
"""

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
import random
import re
from pathlib import Path

# Point DSPy disk cache at the project-local cache/dspy/ directory.
# Must be set BEFORE `import dspy` since dspy reads the env var at import time.
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")

import dspy

try:
    from .prompts import step1_rewrite, step2_response, step3_judge, step4_naturalness
except ImportError:
    from prompts import step1_rewrite, step2_response, step3_judge, step4_naturalness

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TARGET_TRIGGERS_PATH = Path("data/target_triggers.jsonl")
DEFAULT_NEAR_MISSES_PATH = Path("data/near_misses.jsonl")
DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
REQUIRED_TARGET_TRIGGERS = 3
REQUIRED_NEAR_MISSES = 3
REQUIRED_MODEL_VENDORS = ("openai", "google", "anthropic")

# Step 2 (rewrite) and Step 3 (response) models are chosen independently
# per candidate so every (rewriter, responder) pair gets exercised, not
# just same-vendor diagonals. See `process_row` for the index math.
#
# REWRITE_MODELS are frontier-tier (craft a subtle elicitation prompt);
# RESPONSE_MODELS are older/weaker "subject under test" models. The two
# lists happen to be same-length (3) today but that is not required —
# the indexing `REWRITE_MODELS[i % R]` and `RESPONSE_MODELS[(i // R) % S]`
# works for any sizes; keeping them equal gives uniform coverage of all
# R×S combinations every R×S candidates.
REWRITE_MODELS = [
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4-6",
]
RESPONSE_MODELS = [
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-sonnet-4",
]


def model_vendor(model: str) -> str:
    """Return the OpenRouter provider prefix for a model id."""
    return model.split("/", 1)[0]


def select_model_pair(candidate_idx: int) -> tuple[str, str]:
    """Select the deterministic (rewrite, response) model pair for a candidate.

    The rewrite model runs on the inner cycle and the response model runs on
    the outer cycle, so the first 9 candidates cover every 3x3 pair exactly
    once, including cross-vendor pairs.
    """
    rewrite_model = REWRITE_MODELS[candidate_idx % len(REWRITE_MODELS)]
    response_model = RESPONSE_MODELS[
        (candidate_idx // len(REWRITE_MODELS)) % len(RESPONSE_MODELS)
    ]
    return rewrite_model, response_model


def validate_model_vendor_split() -> None:
    """Fail fast if rewrite/response models are not balanced across vendors."""
    required = Counter(REQUIRED_MODEL_VENDORS)
    rewrite_vendors = Counter(model_vendor(model) for model in REWRITE_MODELS)
    response_vendors = Counter(model_vendor(model) for model in RESPONSE_MODELS)
    if rewrite_vendors != required:
        raise ValueError(
            f"REWRITE_MODELS must contain exactly one model from each of "
            f"{REQUIRED_MODEL_VENDORS}; got {dict(rewrite_vendors)}"
        )
    if response_vendors != required:
        raise ValueError(
            f"RESPONSE_MODELS must contain exactly one model from each of "
            f"{REQUIRED_MODEL_VENDORS}; got {dict(response_vendors)}"
        )


def principle_of(measure: str) -> str:
    """Return the principle prefix of a measure name.

    Measures are grouped into three principles based on the leading digit of
    the folder name: '1' (identity / human-like presentation), '2'
    (fabrication / emotional attachment / sycophancy / relationship
    encouragement), '3' (engagement). Negative examples are pooled at this
    level because some measures have too few raw negatives on their own.
    """
    return measure[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_filter2_prompt(measure: str, measure_dir: Path) -> str:
    """Load the category definition prompt from filter2.json."""
    path = measure_dir / measure / "filter2.json"
    with open(path) as f:
        return json.load(f)["prompt"]


def _load_pool(path: Path, predicate) -> list[dict]:
    """Load rows from a slim single-turn JSONL, keeping rows where
    `predicate(row_measures)` is true.

    Expected schema per row: {user_input, measure, synthetic, language}.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            measures = r.get("measure") or []
            if predicate(measures):
                rows.append(r)
    return rows


def load_target_triggers_pool(path: Path, measure: str) -> list[dict]:
    """Load target-trigger rows whose `measure` list contains the target measure."""
    return _load_pool(path, lambda ms: measure in ms)


def load_near_misses_pool(path: Path, measure: str) -> list[dict]:
    """Load near-miss rows whose `measure` list contains any measure sharing
    a principle with the target (1x / 2x / 3x grouping)."""
    principle = principle_of(measure)
    return _load_pool(path, lambda ms: any(principle_of(m) == principle for m in ms))


def with_user_input_first(row: dict) -> dict:
    """Return a new dict with `user_input` as the first key (if present).
    Rejected rows whose Step 2 failed don't have `user_input`; return as-is."""
    if "user_input" not in row:
        return row
    rest = {k: v for k, v in row.items() if k != "user_input"}
    return {"user_input": row["user_input"], **rest}


def existing_output_hashes(output_path: Path) -> set[str]:
    """Hashes of `user_input` values already in the output (for write-time dedup)."""
    hashes: set[str] = set()
    if not output_path.exists():
        return hashes
    with open(output_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "user_input" in r:
                    hashes.add(source_hash(r["user_input"]))
            except json.JSONDecodeError:
                continue
    return hashes


def source_hash(user_input: str) -> str:
    return hashlib.md5(user_input.encode()).hexdigest()


def validate_sampling_contract(
    target_trigger_pool: list[dict],
    near_miss_pool: list[dict],
) -> None:
    """Require the production sampling shape: 3 triggers + 3 near misses + 1 candidate."""
    near_miss_inputs = Counter(r["user_input"] for r in near_miss_pool)
    target_trigger_inputs = Counter(r["user_input"] for r in target_trigger_pool)

    # Each candidate is excluded from its own anchor pools. These minima catch
    # tiny pools and accidental overlap before any API calls are made.
    max_near_miss_overlap = max(near_miss_inputs.values(), default=0)
    max_target_overlap = max(
        (target_trigger_inputs.get(user_input, 0) for user_input in near_miss_inputs),
        default=0,
    )
    min_near_misses_available = len(near_miss_pool) - max_near_miss_overlap
    min_target_triggers_available = len(target_trigger_pool) - max_target_overlap

    if min_target_triggers_available < REQUIRED_TARGET_TRIGGERS:
        raise SystemExit(
            f"ERROR: Need {REQUIRED_TARGET_TRIGGERS} target-trigger anchors after "
            f"excluding the current candidate; only {min_target_triggers_available} "
            f"are guaranteed. Add more target triggers or remove overlap with "
            f"near-miss candidates."
        )
    if min_near_misses_available < REQUIRED_NEAR_MISSES:
        raise SystemExit(
            f"ERROR: Need {REQUIRED_NEAR_MISSES} near-miss anchors plus one current "
            f"candidate; only {min_near_misses_available} near-miss anchors are "
            f"guaranteed after excluding a candidate. Add more near misses."
        )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict | None:
    """Extract a JSON object from a model response, handling markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    # Find the first { ... } block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# API call helpers (DSPy-backed disk cache at cache/dspy/)
# ---------------------------------------------------------------------------

# LM instances are cached by (model, temperature, max_tokens) so the DSPy
# disk cache key is stable across calls. dspy.LM keeps its own cache layer
# under DSPY_CACHEDIR (set above to cache/dspy/).
_LM_CACHE: dict[tuple[str, float, int], dspy.LM] = {}
_API_KEY: str | None = None


def _set_api_key(key: str) -> None:
    """Stash the API key for later _get_lm() calls. Set once in main_async."""
    global _API_KEY
    _API_KEY = key


def _get_lm(model: str, temperature: float, max_tokens: int) -> dspy.LM:
    """Return a cached dspy.LM for (model, temperature, max_tokens)."""
    key = (model, round(float(temperature), 4), int(max_tokens))
    lm = _LM_CACHE.get(key)
    if lm is None:
        lm = dspy.LM(
            model=f"openrouter/{model}",
            api_key=_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=True,
            timeout=180,
        )
        _LM_CACHE[key] = lm
    return lm


async def call_api(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 5,
) -> str | None:
    """Make a cached API call via DSPy. Returns raw content or None on failure.

    DSPy's disk cache is keyed on (model, sampling kwargs, messages) and lives
    under cache/dspy/. Cache hits skip the network entirely.
    """
    lm = _get_lm(model, temperature, max_tokens)
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(lm, messages=messages)
            if not response or not response[0]:
                raise ValueError("empty response")
            content = response[0]
            if isinstance(content, dict):
                content = content.get("content", "") or content.get("text", "") or str(content)
            return content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{max_retries} after {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

async def process_row(
    candidate_idx: int,
    round_idx: int,
    row: dict,
    sem: asyncio.Semaphore,
    filter2_prompt: str,
    target_trigger_pool: list[dict],
    near_miss_pool: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Run Steps 2-4 for a single candidate. Always returns a dict carrying
    a `_status` field that is one of:
      - "pending_step5" — passed the Opus judge, needs naturalness ranking
      - "rejected_step2_rewrite_null" — rewriter returned empty
      - "rejected_step3_response_null" — response model returned empty
      - "rejected_step4_judge_null" — judge call returned empty
      - "rejected_step4_judge_parse" — judge response wasn't parseable
      - "rejected_step4_judge_keep_false" — judge said keep=false

    The dict includes every field populated up to the point of rejection,
    so `.rejected.jsonl` entries are self-describing.
    """
    async with sem:
        user_input = row["user_input"]
        sh = source_hash(user_input)

        # Per-candidate deterministic RNG — reproducible given --seed and
        # shared across reruns so Step 2 prompts hit the disk cache.
        rng = random.Random(f"{args.seed}:{round_idx}:{candidate_idx}")

        trigger_pool = [
            t for t in target_trigger_pool if t["user_input"] != user_input
        ]
        trigger_sample = rng.sample(
            trigger_pool, min(REQUIRED_TARGET_TRIGGERS, len(trigger_pool))
        )
        near_miss_candidates = [
            r for r in near_miss_pool if r["user_input"] != user_input
        ]
        near_miss_sample = rng.sample(
            near_miss_candidates, min(REQUIRED_NEAR_MISSES, len(near_miss_candidates))
        )

        trigger_inputs = [r["user_input"][:500] for r in trigger_sample]
        near_miss_inputs = [r["user_input"][:500] for r in near_miss_sample]

        record: dict = {
            "source_input": user_input,
            "source_hash": sh,
            "round_idx": round_idx,
            "candidate_idx": candidate_idx,
            "few_shot_trigger_hashes": [
                source_hash(t["user_input"]) for t in trigger_sample
            ],
        }

        # Step 2 (rewrite) and Step 3 (response) models are chosen
        # INDEPENDENTLY so every (rewrite, response) pair gets exercised,
        # not just the three same-vendor diagonals. Over 9 consecutive
        # candidates all 3×3=9 combinations appear exactly once before
        # the cycle repeats:
        #   rewrite = REWRITE_MODELS[i % 3]        (fast inner cycle)
        #   response = RESPONSE_MODELS[(i // 3) % 3]  (outer cycle)
        rewrite_model = REWRITE_MODELS[candidate_idx % len(REWRITE_MODELS)]
        response_model = RESPONSE_MODELS[(candidate_idx // len(REWRITE_MODELS)) % len(RESPONSE_MODELS)]
        record["rewrite_model"] = rewrite_model
        record["response_model"] = response_model

        # Step 2: Rewrite (frontier model, matches vendor of response model)
        messages = step1_rewrite(
            filter2_prompt, user_input[:2000], trigger_inputs, near_miss_inputs
        )
        rewritten = await call_api(rewrite_model, messages, temperature=0.9)
        if not rewritten:
            record["_status"] = "rejected_step2_rewrite_null"
            return record
        rewritten = rewritten.strip('"').strip("'")
        record["user_input"] = rewritten

        # Step 3: Response (equal-performing older model, same vendor as rewrite)
        messages = step2_response(rewritten)
        response = await call_api(response_model, messages, temperature=0.7)
        if not response:
            record["_status"] = "rejected_step3_response_null"
            return record
        record["assistant_response"] = response

        # Step 4: Opus judge
        messages = step3_judge(filter2_prompt, rewritten, response)
        judge_raw = await call_api(args.judge_model, messages, temperature=0.0)
        if not judge_raw:
            record["_status"] = "rejected_step4_judge_null"
            return record
        parsed = parse_json_response(judge_raw)
        if not parsed:
            print(f"    Failed to parse judge response for {sh[:8]}", flush=True)
            record["_status"] = "rejected_step4_judge_parse"
            record["raw_judge_response"] = judge_raw
            return record
        record["judge_output"] = parsed
        keep = parsed.get("keep")
        if isinstance(keep, str):
            keep = keep.lower() == "true"
        if not keep:
            record["_status"] = "rejected_step4_judge_keep_false"
            return record

        # Passed Step 4 — prepare for Step 5.
        record.update({
            "_status": "pending_step5",
            "measure": [args.measure],
            "synthetic": True,
            "language": "English",
            "naturalness_ranking": None,
            "naturalness_passed": None,
            "_few_shot_trigger_texts": trigger_inputs,
        })
        return record


async def run_naturalness_for_row(
    row: dict,
    args: argparse.Namespace,
) -> dict:
    """Step 5: per-candidate naturalness ranking.

    Rank {the 3 few-shot positives used in Step 2, the rewritten candidate}
    from most to least natural. Discard iff the candidate is ranked last.
    On API or parse failure, keep conservatively.

    Always returns a dict. `_status` is one of:
      - "accepted" — passed naturalness ranking
      - "rejected_step5_ranked_last" — candidate ranked last → discard
      - "accepted_step5_unjudged" — API/parse failure; kept conservatively
    """
    trigger_texts = row.pop("_few_shot_trigger_texts", None) or []

    items = [{"text": t, "is_candidate": False} for t in trigger_texts]
    items.append({"text": row["user_input"][:500], "is_candidate": True})

    rng = random.Random(f"step5:{row['source_hash']}")
    rng.shuffle(items)

    examples = []
    candidate_id: int | None = None
    for i, item in enumerate(items, start=1):
        examples.append({
            "id": i,
            "user_input": item["text"],
            "is_synthetic": item["is_candidate"],
        })
        if item["is_candidate"]:
            candidate_id = i

    messages = step4_naturalness(examples)
    raw = await call_api(args.naturalness_model, messages, temperature=0.0)

    if not raw:
        row["_status"] = "accepted_step5_unjudged"
        row["naturalness_passed"] = True
        row["naturalness_ranking"] = None
        return row

    parsed = parse_json_response(raw)
    if not parsed or "ranking" not in parsed:
        row["_status"] = "accepted_step5_unjudged"
        row["naturalness_passed"] = True
        row["naturalness_ranking"] = None
        return row

    try:
        ranking = [int(x) for x in parsed["ranking"]]
    except (TypeError, ValueError):
        row["_status"] = "accepted_step5_unjudged"
        row["naturalness_passed"] = True
        row["naturalness_ranking"] = None
        return row

    row["naturalness_ranking"] = ranking
    if ranking and ranking[-1] == candidate_id:
        row["_status"] = "rejected_step5_ranked_last"
        row["naturalness_passed"] = False
        print(f"    Step 5: dropped {row['source_hash'][:8]} (ranked last)", flush=True)
        return row
    row["_status"] = "accepted"
    row["naturalness_passed"] = True
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    api_key = Path(args.key).read_text().strip()
    _set_api_key(api_key)

    measure_dir = Path(args.measure_dir)
    filter2_prompt = load_filter2_prompt(args.measure, measure_dir)
    print(f"Loaded filter2 prompt for {args.measure} ({len(filter2_prompt)} chars)")

    target_triggers_path = Path(args.target_triggers_path)
    near_misses_path = Path(args.near_misses_path)
    principle = principle_of(args.measure)

    # Triggers: per-measure real rows only (no recursive self-feed).
    target_trigger_pool = load_target_triggers_pool(target_triggers_path, args.measure)
    print(
        f"Target triggers ({target_triggers_path}): {len(target_trigger_pool)} rows "
        f"for measure {args.measure!r}"
    )
    if len(target_trigger_pool) < REQUIRED_TARGET_TRIGGERS:
        raise SystemExit(
            f"ERROR: Only {len(target_trigger_pool)} target-trigger rows for "
            f"measure {args.measure!r}; need at least {REQUIRED_TARGET_TRIGGERS} to "
            f"form a distinct few-shot anchor set. Add hand-curated rows to "
            f"{target_triggers_path}."
        )

    near_miss_pool = load_near_misses_pool(near_misses_path, args.measure)
    print(
        f"Near misses ({near_misses_path}): {len(near_miss_pool)} rows "
        f"for principle {principle}x"
    )
    if not near_miss_pool:
        raise SystemExit(
            f"ERROR: No near-miss rows for principle {principle}x (measure "
            f"{args.measure!r}) in {near_misses_path}."
        )
    validate_sampling_contract(target_trigger_pool, near_miss_pool)

    output_path = Path(args.output)
    rejected_path = output_path.with_suffix(".rejected.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_hashes = existing_output_hashes(output_path)
    current_count = len(written_hashes)

    # --max_rows overrides --target (for debugging).
    total_target = args.max_rows if args.max_rows else args.target

    print(f"Current output: {current_count} rows")
    print(
        f"Target: {total_target} total rows. Single pass through "
        f"{len(near_miss_pool)} near-miss candidates; stop when target hit "
        f"or pool drained."
    )
    if current_count >= total_target:
        print("Target already reached. Nothing to do.")
        return

    # Per-run counters covering every user_input we generate/attempt.
    stats: dict = {
        "candidates_attempted": 0,
        "rejected_step2_rewrite_null": 0,
        "rejected_step3_response_null": 0,
        "rejected_step4_judge_null": 0,
        "rejected_step4_judge_parse": 0,
        "rejected_step4_judge_keep_false": 0,
        "rejected_step5_ranked_last": 0,
        "accepted_step5_unjudged": 0,
        "accepted": 0,
        "written": 0,
        "duplicates_skipped": 0,
    }

    sem = asyncio.Semaphore(args.concurrency)
    cursor = 0
    round_idx = 0
    written_this_run = 0

    # API-failure retry queue: (cand_idx, round_idx, row) tuples. Filled
    # during the main pool pass, drained in a retry pass after main drain.
    # Preserving the original (cand_idx, round_idx) keeps the prompt
    # identical so any previously-cached calls stay cache hits.
    api_failed_queue: list[tuple[int, int, dict]] = []
    retry_cursor = 0
    retry_announced = False
    API_NULL_STATUSES = {
        "rejected_step2_rewrite_null",
        "rejected_step3_response_null",
        "rejected_step4_judge_null",
    }

    while current_count < total_target:
        in_main_pass = cursor < len(near_miss_pool)
        in_retry_pass = not in_main_pass and retry_cursor < len(api_failed_queue)

        if in_main_pass:
            batch = []
            for _ in range(args.batch_size):
                if cursor >= len(near_miss_pool):
                    break
                row = near_miss_pool[cursor]
                batch.append((cursor, round_idx, row))
                cursor += 1
            print(
                f"\n=== Round {round_idx} | {current_count}/{total_target} "
                f"written | main pass {cursor}/{len(near_miss_pool)} ==="
            )
        elif in_retry_pass:
            if not retry_announced:
                print(
                    f"\n>>> RETRY PASS: re-attempting "
                    f"{len(api_failed_queue)} API-failed candidates "
                    f"(preserving original RNG seeds for cache reuse)."
                )
                retry_announced = True
            batch_end = min(retry_cursor + args.batch_size, len(api_failed_queue))
            batch = api_failed_queue[retry_cursor:batch_end]
            retry_cursor = batch_end
            print(
                f"\n=== Round {round_idx} | {current_count}/{total_target} "
                f"written | retry pass {retry_cursor}/{len(api_failed_queue)} ==="
            )
        else:
            break

        tasks = [
            process_row(
                cand_idx, r_idx, row, sem, filter2_prompt,
                target_trigger_pool, near_miss_pool, args,
            )
            for cand_idx, r_idx, row in batch
        ]
        stats["candidates_attempted"] += len(batch)
        # Map candidate_idx → (cand_idx, round_idx, row) for retry queueing.
        batch_index = {ci: (ci, ri, r) for ci, ri, r in batch}

        step4_pending: list[dict] = []
        step4_rejected: list[dict] = []
        for coro in asyncio.as_completed(tasks):
            record = await coro
            status = record["_status"]
            if status == "pending_step5":
                step4_pending.append(record)
            else:
                stats[status] = stats.get(status, 0) + 1
                step4_rejected.append(record)
                # Queue API failures (main pass only — one retry attempt).
                if in_main_pass and status in API_NULL_STATUSES:
                    ci = record.get("candidate_idx")
                    if ci in batch_index:
                        api_failed_queue.append(batch_index[ci])
        print(
            f"  Steps 2-4: {len(step4_pending)}/{len(batch)} passed Opus judge "
            f"({len(step4_rejected)} rejected)"
        )

        if step4_pending:
            nat_tasks = [
                run_naturalness_for_row(r, args) for r in step4_pending
            ]
            nat_results = await asyncio.gather(*nat_tasks)
        else:
            nat_results = []

        accepted: list[dict] = []
        step5_rejected: list[dict] = []
        for r in nat_results:
            status = r["_status"]
            stats[status] = stats.get(status, 0) + 1
            if status.startswith("accepted"):
                accepted.append(r)
            else:
                step5_rejected.append(r)
        if nat_results:
            print(
                f"  Step 5:    {len(accepted)}/{len(nat_results)} passed naturalness "
                f"({len(step5_rejected)} rejected)"
            )

        # Persist rejected rows (complete audit trail).
        with open(rejected_path, "a") as f:
            for row in step4_rejected + step5_rejected:
                out_row = {k: v for k, v in row.items() if not k.startswith("_")}
                out_row["reject_stage"] = row["_status"]
                f.write(json.dumps(with_user_input_first(out_row), ensure_ascii=False) + "\n")

        # Write accepted rows with write-time dedup; trim to total target.
        new_rows = 0
        with open(output_path, "a") as f:
            for row in accepted:
                ui_hash = source_hash(row["user_input"])
                if ui_hash in written_hashes:
                    stats["duplicates_skipped"] += 1
                    continue
                if current_count + new_rows >= total_target:
                    break
                out_row = {k: v for k, v in row.items() if not k.startswith("_")}
                f.write(json.dumps(with_user_input_first(out_row), ensure_ascii=False) + "\n")
                written_hashes.add(ui_hash)
                new_rows += 1
        current_count += new_rows
        written_this_run += new_rows
        stats["written"] += new_rows
        print(
            f"  Wrote:     {new_rows} new rows "
            f"(total {current_count}/{total_target})"
        )

        round_idx += 1

    # Exit reason:
    if current_count >= total_target:
        print(f"\n>>> Target {total_target} reached. Stopping.")
    elif cursor >= len(near_miss_pool):
        retry_summary = (
            f" (including {len(api_failed_queue)} retry re-attempts)"
            if api_failed_queue
            else ""
        )
        print(
            f"\n>>> Near-miss pool drained ({len(near_miss_pool)} "
            f"candidates){retry_summary} without reaching target. Stopping "
            f"at {current_count}/{total_target}."
        )

    # End-of-run summary — total user_inputs generated + stage-by-stage funnel
    n = stats["candidates_attempted"]
    survival = (stats["written"] / n * 100.0) if n else 0.0
    print(f"\n=== Run summary ({args.measure}) ===")
    print(f"  Candidates attempted (user_inputs generated): {n}")
    print(f"    rejected at Step 2 (rewrite null):       {stats['rejected_step2_rewrite_null']}")
    print(f"    rejected at Step 3 (response null):      {stats['rejected_step3_response_null']}")
    print(f"    rejected at Step 4 (judge null):         {stats['rejected_step4_judge_null']}")
    print(f"    rejected at Step 4 (judge parse fail):   {stats['rejected_step4_judge_parse']}")
    print(f"    rejected at Step 4 (keep=false):         {stats['rejected_step4_judge_keep_false']}")
    print(f"    rejected at Step 5 (ranked last):        {stats['rejected_step5_ranked_last']}")
    print(f"    accepted (Step 5 unjudged, kept conservatively): {stats['accepted_step5_unjudged']}")
    print(f"    accepted (Step 5 passed):                {stats['accepted']}")
    print(f"  Duplicates skipped at write:               {stats['duplicates_skipped']}")
    print(f"  Written this run:                          {written_this_run}")
    print(f"  Total in output file:                      {current_count}")
    print(f"  Survival rate (written / attempted):       {survival:.1f}%")
    target_hit = "✓" if current_count >= total_target else "✗"
    print(f"  {target_hit} Total target:   {current_count}/{total_target}")
    print(f"  Accepted rows:    {output_path}")
    print(f"  Rejected rows:    {rejected_path}")
    print(f"  API cache:        cache/dspy/ (DSPy disk cache)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6 — Synthetic data generation for a single measure.",
    )
    parser.add_argument(
        "--measure", type=str, required=True,
        help="Measure name (folder under src/filter/measure/), e.g. 2C_sycophancy",
    )
    parser.add_argument(
        "--key", type=str, default=".openrouter_key",
        help="Path to OpenRouter API key file",
    )
    parser.add_argument(
        "--measure_dir", type=str, default="src/filter/measure",
        help="Path to measure definitions directory",
    )
    parser.add_argument(
        "--target_triggers_path", type=str, default=str(DEFAULT_TARGET_TRIGGERS_PATH),
        help=f"JSONL of target-trigger rows (confirmed to trigger the category), "
             f"matched per measure (default: {DEFAULT_TARGET_TRIGGERS_PATH})",
    )
    parser.add_argument(
        "--near_misses_path", type=str, default=str(DEFAULT_NEAR_MISSES_PATH),
        help=f"JSONL of near-miss rows (partial matches), pooled per principle "
             f"(1x/2x/3x) for both rewrite candidates and few-shot near-miss "
             f"anchors (default: {DEFAULT_NEAR_MISSES_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for per-candidate few-shot sampling (default: 42).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/synthetic/{measure}.jsonl)",
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="Total target number of surviving synthetic rows for this "
             "measure (default: 200). Single pass through the near-miss "
             "pool; stops when target is hit or pool is drained.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=20,
        help="Candidates processed per round (default: 20)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--judge_model", type=str, default="anthropic/claude-opus-4-6",
        help="Model for Step 4 Opus judge (default: Opus 4.6). Verifier is "
             "Opus 4.6 always — override at your own risk.",
    )
    parser.add_argument(
        "--naturalness_model", type=str, default="anthropic/claude-opus-4-6",
        help="Model for Step 5 naturalness ranking (default: Opus 4.6).",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Debug cap on total surviving rows; overrides --target when set.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(DEFAULT_SYNTHETIC_DIR / f"{args.measure}.jsonl")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
