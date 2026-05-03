"""Stage 7.2 — LLM-as-judge evaluation of model responses.

For every row produced by Stage 7.1 (one row per `user_input`, with a
`model_responses` dict containing the generated model variants), this script
judges every (input, model) pair against the row's labelled measures.

Expected scale after merging the local-HF Qwen add-on:
3,147 labelled input-measure pairs × 25 models = 78,675 calls.

Judge: Claude Opus 4.6 via OpenRouter (no thinking, temperature=0).

Caching:
  - DSPy disk cache (cache/dspy/) — re-runs are free for any cached
    (system_prompt, conversation) pair.
  - File-based resumption: already-written (user_input, measure, model_name)
    triples are skipped before any LM call is made.
  - --rollout_id forces fresh generations while still writing them to cache.

Keys:
  Reads `.keys.json` (default path) and uses the `openrouter` section.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# DSPy disk cache must be configured before import.
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")

import dspy

from src.evaluation._eval_common import (
    ALL_MEASURES,
    DEFAULT_MEASURE_DIR,
    ParseError,
    extract_text,
    load_judge_prompts,
    load_keys,
    normalize_keep,
    parse_judge_response,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-opus-4.6"


def call_judge(lm: dspy.LM, system_prompt: str, conversation: str,
               rollout_id: Optional[int]) -> tuple[str, dict]:
    """Single judge call. Retries parse failures up to 3x by bumping rollout_id
    so each retry skips the previous (cached) bad response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation},
    ]

    last_raw = ""
    last_err: Optional[Exception] = None
    for attempt in range(3):
        config: dict = {}
        if rollout_id is not None:
            config["rollout_id"] = rollout_id + attempt
        elif attempt > 0:
            config["rollout_id"] = attempt
        try:
            response = lm(messages=messages, **config)
            raw = extract_text(response)
            last_raw = raw
            parsed = normalize_keep(parse_judge_response(raw))
            return raw, parsed
        except ParseError as e:
            last_err = e
        except Exception as e:
            last_err = e
    if isinstance(last_err, ParseError):
        return last_raw, {"error": "parse_failed", "raw_output": last_raw}
    return last_raw, {"error": str(last_err) if last_err else "unknown"}


def evaluate_triple(
    lm: dspy.LM, user_input: str, measure: str, model_name: str,
    model_response: str, judge_prompt: str, rollout_id: Optional[int],
) -> dict:
    """Judge one (user_input, measure, model) triple."""
    conversation = f"USER: {user_input}\nASSISTANT: {model_response}"
    raw, parsed = call_judge(lm, judge_prompt, conversation, rollout_id)
    return {
        "user_input": user_input,
        "measure": measure,
        "model_name": model_name,
        "model_response": model_response,
        "raw_judge_response": raw,
        "judge_output": parsed,
    }


def discover_model_columns(rows: list[dict]) -> list[str]:
    """Union of model_responses keys across all input rows.

    Using the union (not the first row's keys) handles paused/resumed Stage 7.1
    output where an early row may have fewer columns than later rows.
    """
    cols: set[str] = set()
    for r in rows:
        cols.update(r.get("model_responses", {}).keys())
    return sorted(cols)


def labelled_measures(row: dict) -> list[str]:
    """Return the row's labelled measure list, normalized and de-duplicated."""
    raw = row.get("measure", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out = []
    seen = set()
    for measure in raw:
        if measure not in ALL_MEASURES:
            raise SystemExit(
                f"Unknown labelled measure {measure!r} for input: "
                f"{row.get('user_input', '')[:120]}"
            )
        if measure not in seen:
            out.append(measure)
            seen.add(measure)
    return out


def run(args):
    keys = load_keys(args.keys)
    if not keys.get("openrouter"):
        raise SystemExit(f"{args.keys} is missing the 'openrouter' section")

    judge_prompts = load_judge_prompts(ALL_MEASURES, Path(args.measure_dir))
    if len(judge_prompts) != len(ALL_MEASURES):
        missing = set(ALL_MEASURES) - set(judge_prompts)
        raise SystemExit(f"Missing filter2.json prompts for: {missing}")

    lm = dspy.LM(
        model=f"openrouter/{JUDGE_MODEL}",
        api_key=keys["openrouter"],
        api_base=OPENROUTER_BASE_URL,
        max_tokens=1024,
        temperature=0,
        cache=True,
        timeout=120,
    )

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} rows from {args.input}")

    model_columns = discover_model_columns(rows)
    print(f"Discovered {len(model_columns)} model columns: {model_columns}")

    completed: set[tuple[str, str, str]] = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                completed.add((r["user_input"], r["measure"], r["model_name"]))
    print(f"Resuming: {len(completed)} triples already in {output_path}")

    # Build the task list. Judges only the measures listed on row["measure"].
    # Rows can carry multiple labels, so each (input, labelled measure, model)
    # triple is evaluated independently.
    tasks = []
    skipped_empty = 0
    labelled_pairs = 0
    for row in rows:
        user_input = row["user_input"]
        model_responses = row.get("model_responses", {})
        measures = labelled_measures(row)
        labelled_pairs += len(measures)
        for measure in measures:
            for model_name in model_columns:
                if (user_input, measure, model_name) in completed:
                    continue
                resp = model_responses.get(model_name, {})
                model_response = resp.get("assistant_response", "")
                if not model_response:
                    skipped_empty += 1
                    continue
                tasks.append((user_input, measure, model_name, model_response))

    print(f"Labelled input-measure pairs: {labelled_pairs}")
    print(f"Tasks to run: {len(tasks)} "
          f"(skipped {skipped_empty} (input, measure, model) triples with empty response)")
    print(f"DSPy caching: enabled (rollout_id={args.rollout_id})")
    if not tasks:
        print("Nothing to do.")
        return

    total = len(tasks)
    done = 0
    with open(output_path, "a") as f_out, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {
            pool.submit(
                evaluate_triple,
                lm, ui, m, mn, mr, judge_prompts[m], args.rollout_id,
            ): (ui, m, mn)
            for ui, m, mn, mr in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  Progress: {done}/{total}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7.2 — LLM-as-judge evaluation on labelled measures."
    )
    parser.add_argument(
        "--input",
        default="data/eval_responses.jsonl",
        help="Input JSONL (default: data/eval_responses.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/eval_judge_results.jsonl",
        help="Output JSONL (default: data/eval_judge_results.jsonl)",
    )
    parser.add_argument(
        "--keys", default=".keys.json",
        help="Path to .keys.json with {openrouter, deepseek, anthropic} sections",
    )
    parser.add_argument(
        "--measure_dir", default=str(DEFAULT_MEASURE_DIR),
        help="Path to measure directory with filter2.json files",
    )
    parser.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent judge calls (default: 15)",
    )
    parser.add_argument(
        "--rollout_id", type=int, default=None,
        help="DSPy rollout ID — bypass cache and generate fresh (default: use cache)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
