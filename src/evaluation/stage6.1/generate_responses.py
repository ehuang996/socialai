"""Stage 6.1: Generate model responses for single-turn conversations.

Sends each user_input to multiple models via OpenRouter (through DSPy for caching)
and records their assistant_response.

Input: Single-turn JSONL from split_turns.py (Stage 5.5)
Output: JSONL with model_responses dict containing each model's response.

DSPy caching: responses are cached to disk automatically. Re-running with the same
input/model/temperature returns cached results (reproducibility). Use --rollout_id
to force fresh generations while still caching the new results.
"""

import argparse
import json
import os

import dspy
from tqdm import tqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models that support reasoning effort (reasoning models)
REASONING_MODELS = {"openai/o4-mini", "x-ai/grok-3-mini-beta"}

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


def create_lm(model_id, api_key, max_tokens, temperature):
    """Create a DSPy LM instance for an OpenRouter model."""
    return dspy.LM(
        model=f"openrouter/{model_id}",
        api_key=api_key,
        api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=True,
    )


def call_model(lm, user_input, model_id, rollout_id=None):
    """Call a model via DSPy and return the response text."""
    messages = [{"role": "user", "content": user_input}]
    config = {}
    if rollout_id is not None:
        config["rollout_id"] = rollout_id
    if model_id in REASONING_MODELS:
        config["extra_body"] = {"reasoning": {"effort": "low"}}

    response = lm(messages=messages, **config)
    if not response or not response[0]:
        raise ValueError("Model returned empty response")
    return response[0].strip()


def process_row(row, lms, models, rollout_id):
    """Send user_input to each model and return the result dict."""
    user_input = row.get("user_input", "")
    if not user_input:
        return None

    model_results = {}
    for col_name, model_id in models.items():
        lm = lms[col_name]
        try:
            raw = call_model(lm, user_input, model_id, rollout_id)
            model_results[col_name] = {"assistant_response": raw}
        except Exception as e:
            model_results[col_name] = {"error": str(e)}

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
    model_sets = {"1": MODELS, "2": MODELS2}
    models = model_sets.get(args.model_set, MODELS)
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        models = {k: v for k, v in models.items() if k in selected}

    # Create DSPy LM instances (one per model for correct caching)
    lms = {}
    for col_name, model_id in models.items():
        lms[col_name] = create_lm(model_id, api_key, args.max_tokens, args.temperature)

    # Load completed user_inputs for resumption
    completed = set()
    output_path = args.output
    try:
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                completed.add(row["user_input"])
    except FileNotFoundError:
        pass

    # Load input rows
    rows = []
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            if row["user_input"] not in completed:
                rows.append(row)

    print(f"Processing {len(rows)} rows (skipped {len(completed)} completed)")
    print(f"Models: {list(models.keys())}")
    print(f"DSPy caching: enabled (rollout_id={args.rollout_id})")

    rollout_id = args.rollout_id if args.rollout_id is not None else None

    with open(output_path, "a") as f_out:
        for row in tqdm(rows):
            result = process_row(row, lms, models, rollout_id)
            if result:
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from multiple models for collected final dataset."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from collect_final.py")
    parser.add_argument("--output", required=True, help="Output JSONL with model responses")
    parser.add_argument("--key", default="", help="Path to OpenRouter API key file")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Max tokens per response (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--model_set", type=str, default="1", choices=["1", "2"],
                        help="Model set to use: 1 (original 4) or 2 (new 10) (default: 1)")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names to filter from the set")
    parser.add_argument("--rollout_id", type=int, default=None,
                        help="DSPy rollout ID — bypass cache and generate fresh (default: use cache)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
