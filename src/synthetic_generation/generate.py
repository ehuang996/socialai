"""Stage 6 — Synthetic data generation.

For each behavioral category (measure), rewrites Stage 4 candidate inputs
to trigger the category violation, generates a response from a weaker model,
validates with an Opus judge, and filters for naturalness.

Usage:
    uv run python src/synthetic_generation/generate.py \
        --measure 2C_sycophancy \
        --key .openrouter_key \
        --max_rows 5
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

from prompts import step1_rewrite, step2_response, step3_judge, step4_naturalness

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_filter2_prompt(measure: str, measure_dir: Path) -> str:
    """Load the category definition prompt from filter2.json."""
    path = measure_dir / measure / "filter2.json"
    with open(path) as f:
        return json.load(f)["prompt"]


def find_stage4_file(measure: str, experiments_dir: Path) -> Path:
    """Find the Stage 4 result JSONL for the given measure."""
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir() or not re.match(r"^\d+_final_filter$", d.name):
            continue
        candidate = d / "results" / f"{measure}_final.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Stage 4 result found for {measure} in {experiments_dir}"
    )


def load_stage4_pools(
    stage4_path: Path, model_key: str = "claude_opus_4_6"
) -> tuple[list[dict], list[dict]]:
    """Partition Stage 4 rows into positive_pool and candidate_pool.

    positive_pool: chitchat_keep=true AND category_keep=true (both-KEEP)
    candidate_pool: exactly one of chitchat_keep/category_keep is true (XOR)
    """
    positive_pool: list[dict] = []
    candidate_pool: list[dict] = []

    with open(stage4_path) as f:
        for line in f:
            row = json.loads(line)
            resp = row.get("model_responses", {}).get(model_key, {})
            if "error" in resp:
                continue
            ck = resp.get("chitchat_keep", False)
            catk = resp.get("category_keep", False)

            if ck and catk:
                positive_pool.append(row)
            elif ck or catk:  # exactly one true
                candidate_pool.append(row)

    return positive_pool, candidate_pool


def load_completed(output_path: Path) -> set[str]:
    """Load source_hash values from existing output for resumption."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "source_hash" in row:
                        completed.add(row["source_hash"])
                except json.JSONDecodeError:
                    continue
    return completed


def source_hash(user_input: str) -> str:
    return hashlib.md5(user_input.encode()).hexdigest()


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
# API call helpers
# ---------------------------------------------------------------------------

async def call_api(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str | None:
    """Make an API call with retries. Returns raw content or None on failure."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{max_retries} after {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

async def process_row(
    row: dict,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    filter2_prompt: str,
    positive_pool: list[dict],
    candidate_pool: list[dict],
    args: argparse.Namespace,
) -> dict | None:
    """Run Steps 1-3 for a single candidate row. Returns result or None."""
    async with sem:
        user_input = row["user_input"]
        sh = source_hash(user_input)

        # Sample few-shot examples
        pos_sample = random.sample(
            positive_pool, min(args.num_positive, len(positive_pool))
        )
        neg_candidates = [r for r in candidate_pool if r["user_input"] != user_input]
        neg_sample = random.sample(
            neg_candidates, min(args.num_negative, len(neg_candidates))
        )

        pos_inputs = [r["user_input"][:500] for r in pos_sample]
        neg_inputs = [r["user_input"][:500] for r in neg_sample]

        # Step 1: Rewrite
        messages = step1_rewrite(filter2_prompt, user_input[:2000], pos_inputs, neg_inputs)
        rewritten = await call_api(
            client, args.rewrite_model, messages, temperature=0.9
        )
        if not rewritten:
            return None

        # Strip quotes if the model wrapped the output
        rewritten = rewritten.strip('"').strip("'")

        # Step 2: Generate response
        messages = step2_response(rewritten)
        response = await call_api(
            client, args.response_model, messages, temperature=0.7
        )
        if not response:
            return None

        # Step 3: Opus judge
        messages = step3_judge(filter2_prompt, rewritten, response)
        judge_raw = await call_api(
            client, args.judge_model, messages, temperature=0.0
        )
        if not judge_raw:
            return None

        parsed = parse_json_response(judge_raw)
        if not parsed:
            print(f"    Failed to parse judge response for {sh[:8]}", flush=True)
            return None

        keep = parsed.get("keep")
        if isinstance(keep, str):
            keep = keep.lower() == "true"
        if not keep:
            return None

        return {
            "user_input": rewritten,
            "assistant_response": response,
            "measure": [args.measure],
            "synthetic": True,
            "language": "English",
            "source_input": user_input,
            "source_hash": sh,
            "judge_output": parsed,
            "naturalness_passed": None,  # set in Step 4
        }


async def run_naturalness_filter(
    passed_rows: list[dict],
    positive_pool: list[dict],
    client: AsyncOpenAI,
    args: argparse.Namespace,
) -> list[dict]:
    """Step 4: filter out synthetic-looking examples in batches."""
    if not passed_rows:
        return []

    kept = []
    k = args.naturalness_k

    for batch_start in range(0, len(passed_rows), k):
        batch = passed_rows[batch_start : batch_start + k]

        # Build comparison set: real positives + generated
        real_sample = random.sample(
            positive_pool, min(args.num_positive, len(positive_pool))
        )

        examples = []
        id_map = {}  # id -> (is_synthetic, index_in_batch)
        current_id = 1

        # Interleave real and synthetic, shuffled
        all_items = []
        for r in real_sample:
            all_items.append(("real", r["user_input"][:500], None))
        for i, row in enumerate(batch):
            all_items.append(("synthetic", row["user_input"][:500], i))

        random.shuffle(all_items)

        for kind, text, batch_idx in all_items:
            examples.append({
                "id": current_id,
                "user_input": text,
                "is_synthetic": kind == "synthetic",
            })
            id_map[current_id] = (kind == "synthetic", batch_idx)
            current_id += 1

        messages = step4_naturalness(examples)
        raw = await call_api(
            client, args.naturalness_model, messages, temperature=0.0
        )

        if not raw:
            # On failure, keep all (conservative)
            for row in batch:
                row["naturalness_passed"] = True
            kept.extend(batch)
            continue

        parsed = parse_json_response(raw)
        if not parsed or "most_unlikely_id" not in parsed:
            for row in batch:
                row["naturalness_passed"] = True
            kept.extend(batch)
            continue

        flagged_id = parsed["most_unlikely_id"]
        is_synthetic, batch_idx = id_map.get(flagged_id, (False, None))

        for i, row in enumerate(batch):
            if is_synthetic and batch_idx == i:
                row["naturalness_passed"] = False
                print(f"    Step 4: flagged synthetic row {row['source_hash'][:8]}", flush=True)
            else:
                row["naturalness_passed"] = True
                kept.append(row)

    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    api_key = Path(args.key).read_text().strip()
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # Load category definition
    measure_dir = Path(args.measure_dir)
    filter2_prompt = load_filter2_prompt(args.measure, measure_dir)
    print(f"Loaded filter2 prompt for {args.measure} ({len(filter2_prompt)} chars)")

    # Find and load Stage 4 data
    stage4_path = find_stage4_file(args.measure, Path(args.stage4_dir))
    print(f"Stage 4 file: {stage4_path}")

    positive_pool, candidate_pool = load_stage4_pools(stage4_path)
    print(f"Positive pool: {len(positive_pool)} rows")
    print(f"Candidate pool: {len(candidate_pool)} rows")

    if not positive_pool:
        print("ERROR: No positive examples found. Cannot proceed.")
        return
    if not candidate_pool:
        print("ERROR: No rewrite candidates found. Cannot proceed.")
        return

    # Resumption
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    print(f"Already completed: {len(completed)} rows")

    # Filter to unprocessed candidates
    candidates = [
        r for r in candidate_pool
        if source_hash(r["user_input"]) not in completed
    ]
    if args.max_rows:
        candidates = candidates[: args.max_rows]
    print(f"Processing {len(candidates)} candidates\n")

    # Steps 1-3: async
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_row(
            row, client, sem, filter2_prompt,
            positive_pool, candidate_pool, args,
        )
        for row in candidates
    ]

    step3_passed = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        if result:
            step3_passed.append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(tasks):
            print(
                f"  Steps 1-3: {i+1}/{len(tasks)} done, "
                f"{len(step3_passed)} passed Step 3",
                flush=True,
            )

    print(f"\nStep 3 passed: {len(step3_passed)}/{len(candidates)}")

    # Step 4: naturalness filter
    print("Running Step 4 (naturalness filter)...")
    final_rows = await run_naturalness_filter(
        step3_passed, positive_pool, client, args
    )
    print(f"Step 4 passed: {len(final_rows)}/{len(step3_passed)}")

    # Write results
    with open(output_path, "a") as f:
        for row in final_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(final_rows)} rows to {output_path}")
    print(f"Total in output file: {len(completed) + len(final_rows)}")


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
        "--stage4_dir", type=str, default="experiments",
        help="Root experiments directory containing *_final_filter results",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/synthetic/{measure}.jsonl)",
    )
    parser.add_argument(
        "--num_positive", type=int, default=3,
        help="Number of positive few-shot examples in Step 1",
    )
    parser.add_argument(
        "--num_negative", type=int, default=3,
        help="Number of negative few-shot examples in Step 1",
    )
    parser.add_argument(
        "--naturalness_k", type=int, default=3,
        help="Batch size for Step 4 naturalness comparison",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--rewrite_model", type=str, default="openai/gpt-5.3-chat",
        help="Model for Step 1 rewrite",
    )
    parser.add_argument(
        "--response_model", type=str, default="openai/gpt-4o",
        help="Model for Step 2 response generation",
    )
    parser.add_argument(
        "--judge_model", type=str, default="anthropic/claude-opus-4-6",
        help="Model for Step 3 Opus judge",
    )
    parser.add_argument(
        "--naturalness_model", type=str, default="openai/gpt-5.3-chat",
        help="Model for Step 4 naturalness filter",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Limit number of candidates to process (for debugging)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/synthetic/{args.measure}.jsonl"

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
