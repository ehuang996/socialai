"""Generate responses from multiple models for the collected final dataset.

Sends each user_input to multiple models via OpenRouter (through DSPy for caching)
and records their assistant_response.

Input: JSONL from stage5 data_preprocessing.py (user_input, measure list, etc.)
Output: JSONL with model_responses dict containing each model's response.

DSPy caching: responses are cached to disk automatically. Re-running with the same
input/model/temperature returns cached results (reproducibility). Use --rollout_id
to force fresh generations while still caching the new results.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from tqdm import tqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models where we request low reasoning effort
REASONING_EFFORT_MODELS = {"openai/o4-mini", "x-ai/grok-3-mini-beta"}

# OpenAI reasoning models where DSPy requires temperature=1.0 and max_tokens>=16000
OPENAI_REASONING_MODELS = {"openai/o4-mini"}

# initial run
MODELS = {
    "grok3_mini_beta": "x-ai/grok-3-mini-beta",
    "gpt4o_mini": "openai/gpt-4o-mini",
    "gemini2_flash_001": "google/gemini-2.0-flash-001",
    "claude_sonnet_4": "anthropic/claude-sonnet-4",
}

# 2nd run (with newer models)
MODELS2 = {
    # OpenAI
    "o4_mini": "openai/o4-mini",
    "gpt5_3": "openai/gpt-5.3-chat",
    "gpt5_4": "openai/gpt-5.4",
    "gpt5_4_pro": "openai/gpt-5.4-pro",
    # Anthropic
    "claude_haiku": "anthropic/claude-haiku-4.5",
    "claude_sonnet": "anthropic/claude-sonnet-4.6",
    "claude_opus": "anthropic/claude-opus-4.6",
    # Google
    "gemini3_flash": "google/gemini-3-flash-preview",
    "gemini3_1_pro": "google/gemini-3.1-pro-preview",
    # xAI
    "grok4": "x-ai/grok-4.20-beta",
}


def create_lm(model_id, api_key, max_tokens, temperature, timeout=180):
    """Create a DSPy LM instance for an OpenRouter model."""
    # OpenAI reasoning models require temperature=1.0 and max_tokens >= 16000
    if model_id in OPENAI_REASONING_MODELS:
        return dspy.LM(
            model=f"openrouter/{model_id}",
            api_key=api_key,
            api_base=OPENROUTER_BASE_URL,
            max_tokens=max(max_tokens, 16000),
            temperature=1.0,
            cache=True,
            timeout=timeout,
        )
    return dspy.LM(
        model=f"openrouter/{model_id}",
        api_key=api_key,
        api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=True,
        timeout=timeout,
    )


def call_model(lm, user_input, model_id, rollout_id=None):
    """Call a model via DSPy and return the response text."""
    messages = [{"role": "user", "content": user_input}]
    config = {}
    if rollout_id is not None:
        config["rollout_id"] = rollout_id
    if model_id in REASONING_EFFORT_MODELS:
        config["extra_body"] = {"reasoning": {"effort": "low"}}

    response = lm(messages=messages, **config)
    if not response or not response[0]:
        raise ValueError("Model returned empty response")
    result = response[0]
    # DSPy reasoning models may return a dict with 'content' key
    if isinstance(result, dict):
        result = result.get("content", "") or result.get("text", "") or str(result)
    return result.strip()


def _call_model_with_retry(col_name, model_id, lm, user_input, rollout_id, max_retries=3):
    """Call a single model with retries. Returns (col_name, result_dict)."""
    for attempt in range(max_retries):
        try:
            raw = call_model(lm, user_input, model_id, rollout_id)
            return col_name, {"assistant_response": raw}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return col_name, {"error": str(e)}


def process_row(row, lms, models, rollout_id):
    """Send user_input to all models in parallel and return the result dict."""
    user_input = row.get("user_input", "")
    if not user_input:
        return None

    model_results = {}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                _call_model_with_retry, col_name, model_id, lms[col_name], user_input, rollout_id
            ): col_name
            for col_name, model_id in models.items()
        }
        for future in as_completed(futures):
            col_name, result = future.result()
            model_results[col_name] = result

    return {
        "user_input": user_input,
        "original_assistant_response": row.get("assistant_response", ""),
        "timestamp": row.get("timestamp", ""),
        "measure": row.get("measure", []),
        "model_responses": model_results,
    }


def run(args):
    # Load API key
    if args.key:
        with open(args.key) as f:
            api_key = f.read().strip()
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise ValueError("No API key provided. Use --key or set OPENROUTER_API_KEY.")

    # Select models
    model_sets = {"1": MODELS, "2": MODELS2, "all": {**MODELS, **MODELS2}}
    models = model_sets.get(args.model_set, MODELS)
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        models = {k: v for k, v in models.items() if k in selected}

    # Create DSPy LM instances (one per model for correct caching)
    lms = {}
    for col_name, model_id in models.items():
        lms[col_name] = create_lm(model_id, api_key, args.max_tokens, args.temperature)

    # Load completed rows for resumption (supports partial model completion)
    completed_rows = {}  # user_input -> output row
    output_path = args.output
    try:
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                completed_rows[row["user_input"]] = row
    except FileNotFoundError:
        pass

    # Load input rows and figure out which need (re)processing
    rows = []
    rows_needing_update = []  # rows with some but not all models
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            ui = row["user_input"]
            if ui in completed_rows:
                existing = completed_rows[ui]
                existing_models = set(existing.get("model_responses", {}).keys())
                missing = set(models.keys()) - existing_models
                if missing:
                    rows_needing_update.append((row, existing, missing))
            else:
                rows.append(row)

    print(f"Processing {len(rows)} new rows, updating {len(rows_needing_update)} partial rows "
          f"(skipped {len(completed_rows) - len(rows_needing_update)} fully completed)")
    print(f"Models: {list(models.keys())}")
    print(f"DSPy caching: enabled (rollout_id={args.rollout_id})")

    rollout_id = args.rollout_id if args.rollout_id is not None else None

    # Process new rows (append to file)
    with open(output_path, "a") as f_out:
        for row in tqdm(rows, desc="New rows"):
            result = process_row(row, lms, models, rollout_id)
            if result:
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                completed_rows[result["user_input"]] = result

    # Update partial rows (need to rewrite file)
    if rows_needing_update:
        print(f"Updating {len(rows_needing_update)} rows with missing models...")
        for input_row, existing_row, missing in tqdm(rows_needing_update, desc="Updating"):
            missing_models = {k: v for k, v in models.items() if k in missing}
            missing_lms = {k: v for k, v in lms.items() if k in missing}
            result = process_row(input_row, missing_lms, missing_models, rollout_id)
            if result:
                existing_row["model_responses"].update(result["model_responses"])

        # Rewrite the output file with updated rows
        all_rows = list(completed_rows.values())
        with open(output_path, "w") as f_out:
            for row in all_rows:
                f_out.write(json.dumps(row) + "\n")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from multiple models for collected final dataset."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from stage5 data_preprocessing.py")
    parser.add_argument("--output", required=True, help="Output JSONL with model responses")
    parser.add_argument("--key", default="", help="Path to OpenRouter API key file")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens per response (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--model_set", type=str, default="1", choices=["1", "2", "all"],
                        help="Model set to use: 1 (original 4), 2 (new 10), or all (all 14) (default: 1)")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names to filter from the set")
    parser.add_argument("--rollout_id", type=int, default=None,
                        help="DSPy rollout ID — bypass cache and generate fresh (default: use cache)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
