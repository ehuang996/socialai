"""Stage 7.1 — LLM-as-a-judge evaluation of 14 models on single-turn dataset.

For each row in single_turn_model_responses_439.jsonl, evaluates each of the
14 model responses against the category-specific rubric for every measure in
the row's `measure` list.

Judge: Claude Opus 4.6 via OpenRouter.
Output: one JSONL row per (row, measure, model) triple with the judge's verdict.
Expected: 140 measure-labels × 14 models = 1,960 API calls.
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ── Measure rubrics (loaded from filter2.json) ────────────────────────────

MEASURES = [
    "1B_human_disfluencies",
    "1C_identity_transparency",
    "2A_fabricated_personal_details",
    "2B_explicit_emotions",
    "2B_implicit_emotions",
    "2B_romantic_bonding",
    "2C_sycophancy",
    "2D_human_relationship_encouragement",
    "3A_engagement_hooks",
]

MODEL_COLUMNS = [
    "grok3_mini_beta",
    "gpt4o_mini",
    "gemini2_flash_001",
    "claude_sonnet_4",
    "o4_mini",
    "gpt5_3",
    "gpt5_4",
    "gpt5_4_pro",
    "claude_haiku",
    "claude_sonnet",
    "claude_opus",
    "gemini3_flash",
    "gemini3_1_pro",
    "grok4",
]

# Judge model
JUDGE_MODEL = "anthropic/claude-opus-4-6"


def load_measure_prompts(measure_dir: str) -> dict[str, str]:
    """Load category-specific judge prompts from filter2.json for each measure.

    filter2.json contains the standalone category evaluation prompt (no chitchat
    check), which is exactly what we need for Stage 7.1.
    """
    prompts = {}
    base = Path(measure_dir)
    for measure in MEASURES:
        path = base / measure / "filter2.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {measure}")
            continue
        with open(path) as f:
            config = json.load(f)
        prompts[measure] = config["prompt"]
    return prompts


# ── Parsing & API calls ────────────────────────────────────────────────────

import re as _re


class ParseError(Exception):
    """Raised when the judge response is not valid JSON."""
    pass


def parse_judge_response(raw: str) -> dict:
    """Parse JSON from judge output. Tries multiple strategies:

    1. Direct JSON parse
    2. Strip markdown code fences (```json ... ```)
    3. Extract first JSON object via brace-matching regex
    4. Give up → raise ParseError so tenacity retries the API call
    """
    text = raw.strip()

    # Strategy 1: direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: markdown code fences
    if "```" in text:
        # Handle ```json ... ``` or ``` ... ```
        fence_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1).strip())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

    # Strategy 3: find first { ... } block in the text
    brace_match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: find nested { ... { ... } ... } (one level of nesting)
    nested_match = _re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", text, _re.DOTALL)
    if nested_match:
        try:
            data = json.loads(nested_match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 5: extract "keep" field directly via regex when JSON has
    # unescaped quotes in the reasoning string (common with Opus)
    keep_match = _re.search(r'"keep"\s*:\s*(true|false)', text, _re.IGNORECASE)
    reasoning_match = _re.search(
        r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"keep"', text, _re.DOTALL
    )
    if keep_match:
        keep_val = keep_match.group(1).lower() == "true"
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        return {"reasoning": reasoning, "keep": keep_val}

    raise ParseError(f"Could not parse JSON from response: {raw[:200]}")


def validate_judge_output(parsed: dict) -> dict:
    """Validate that the parsed output has the expected fields.

    Expected: {"reasoning": "...", "keep": true/false}
    Normalizes common variations (e.g., boolean strings).
    """
    if "error" in parsed:
        return parsed

    # Normalize 'keep' field
    if "keep" in parsed:
        val = parsed["keep"]
        if isinstance(val, str):
            parsed["keep"] = val.lower().strip() in ("true", "yes", "1")
        elif not isinstance(val, bool):
            parsed["keep"] = bool(val)

    return parsed


REQUEST_TIMEOUT = 120  # seconds per API call


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type((ParseError, asyncio.TimeoutError, Exception)),
)
async def call_and_parse_judge(
    client: AsyncOpenAI,
    system_prompt: str,
    conversation: str,
    sem: asyncio.Semaphore,
) -> tuple[str, dict]:
    """Call the judge model and parse the response. Retries on both API errors
    AND parse failures (e.g., model responds conversationally instead of JSON).

    Returns (raw_response, parsed_dict).
    """
    async with sem:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
                temperature=0,
                max_tokens=1024,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        raw = response.choices[0].message.content.strip()
        # This raises ParseError on failure, triggering tenacity retry
        parsed = parse_judge_response(raw)
        parsed = validate_judge_output(parsed)
        return raw, parsed


# ── Main pipeline ──────────────────────────────────────────────────────────


async def evaluate_single(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    user_input: str,
    measure: str,
    model_name: str,
    model_response: str,
    judge_prompt: str,
) -> dict:
    """Evaluate one (row, measure, model) triple."""
    conversation = f"USER: {user_input}\nASSISTANT: {model_response}"
    raw = ""
    try:
        raw, parsed = await call_and_parse_judge(
            client, judge_prompt, conversation, sem
        )
    except ParseError as e:
        # All 3 retries failed to produce valid JSON
        parsed = {"error": "parse_failed", "raw_output": raw or str(e)}
    except Exception as e:
        parsed = {"error": str(e)}

    return {
        "user_input": user_input,
        "measure": measure,
        "model_name": model_name,
        "model_response": model_response,
        "raw_judge_response": raw,
        "judge_output": parsed,
    }


async def run(args):
    # Load API key
    if args.key:
        with open(args.key) as f:
            api_key = f.read().strip()
    else:
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("No API key. Use --key or set OPENROUTER_API_KEY.")

    # Load category prompts from filter2.json
    judge_prompts = load_measure_prompts(args.measure_dir)

    # Load input data
    rows = []
    with open(args.input) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows")

    # Load already-completed results for resumption
    completed = set()  # (user_input, measure, model_name) tuples
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                completed.add((r["user_input"], r["measure"], r["model_name"]))
    print(f"Resuming: {len(completed)} results already completed")

    # Build task list
    tasks_to_run = []
    for row in rows:
        user_input = row["user_input"]
        measures = row.get("measure", [])
        model_responses = row.get("model_responses", {})
        for measure in measures:
            if measure not in judge_prompts:
                print(f"WARNING: No rubric for {measure}, skipping")
                continue
            for model_name in MODEL_COLUMNS:
                if (user_input, measure, model_name) in completed:
                    continue
                resp = model_responses.get(model_name, {})
                model_response = resp.get("assistant_response", "")
                if not model_response:
                    continue
                tasks_to_run.append(
                    (user_input, measure, model_name, model_response)
                )

    print(f"Tasks to run: {len(tasks_to_run)}")
    if not tasks_to_run:
        print("Nothing to do.")
        return

    # Set up client with explicit timeout
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=REQUEST_TIMEOUT,
    )
    sem = asyncio.Semaphore(args.concurrency)

    # Fire all tasks, write results as they complete (no batch blocking)
    total = len(tasks_to_run)
    done_count = 0

    async def run_and_write(f_out, ui, m, mn, mr):
        nonlocal done_count
        result = await evaluate_single(
            client, sem, ui, m, mn, mr, judge_prompts[m]
        )
        f_out.write(json.dumps(result) + "\n")
        f_out.flush()
        done_count += 1
        if done_count % 30 == 0 or done_count == total:
            print(f"  Progress: {done_count}/{total}")

    with open(output_path, "a") as f_out:
        tasks = [
            run_and_write(f_out, ui, m, mn, mr)
            for ui, m, mn, mr in tasks_to_run
        ]
        await asyncio.gather(*tasks)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7.1: LLM-as-a-judge evaluation of single-turn model responses."
    )
    parser.add_argument(
        "--input",
        default="data/single_turn_model_responses_439.jsonl",
        help="Input JSONL (default: data/single_turn_model_responses_439.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/stage7_1_eval_results.jsonl",
        help="Output JSONL (default: data/stage7_1_eval_results.jsonl)",
    )
    parser.add_argument(
        "--key", default="", help="Path to OpenRouter API key file"
    )
    parser.add_argument(
        "--measure_dir",
        default="src/filter/measure",
        help="Path to measure directory with filter2.json files",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Max concurrent API calls (default: 15)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
