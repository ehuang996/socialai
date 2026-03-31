"""Split final dataset into single-turn and multi-turn conversations.

Uses Opus 4.6 via OpenRouter to classify every conversation. Should be run
AFTER collect_final.py (Stage 5) and BEFORE generate_responses.py (Stage 6).

Single-turn: a clean, standalone user message (no pasted transcript).
Multi-turn: contains role markers (User:, Assistant:, etc.) from a pasted
conversation transcript — the user's first message embeds prior exchanges.

Output:
  - <output_dir>/single_turn_<stem>.jsonl  (clean user messages)
  - <output_dir>/multi_turn_<stem>.jsonl   (pasted transcripts)
  - <output_dir>/split_report_<stem>.json  (classification details)
"""

import argparse
import json
import re
import time
from pathlib import Path

import openai

CLASSIFY_SYSTEM_PROMPT = """\
You are classifying a user message from a chatbot conversation.

Your task: determine whether this is a **single-turn** message or a **multi-turn** pasted transcript.

- **single-turn**: The user wrote a standalone message to the chatbot. It may be short or long, \
may contain lists, code, JSON, or other structured content, but it is ONE message from ONE user \
directed at the chatbot.
- **multi-turn**: The user's message contains a pasted conversation transcript — it includes \
dialogue between multiple parties or multiple exchanges with role labels (e.g., "User:", \
"Assistant:", "Human:", "AI:", "Bot:", etc.). The user is sharing or continuing a prior conversation.

Edge cases:
- If the message contains role labels but they are part of instructions (e.g., "When the user \
says X, respond with Y"), classify as **single-turn**.
- If the message quotes a short snippet of dialogue as an example within a larger standalone \
request, classify as **single-turn**.
- If the message is primarily a multi-exchange transcript (even if it ends with a new request), \
classify as **multi-turn**.

Respond with ONLY a JSON object:
{"classification": "single-turn" or "multi-turn", "reasoning": "brief explanation"}"""


def classify_with_llm(
    client: openai.OpenAI, user_input: str, model: str = "anthropic/claude-opus-4-6",
    max_retries: int = 5,
) -> tuple[str, str]:
    """Use Opus to classify a user_input. Returns (classification, reasoning)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input[:8000]},
                ],
                max_tokens=256,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            break
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait}s...", flush=True)
            time.sleep(wait)
    else:
        return "multi-turn", "All retries failed, defaulting to multi-turn"

    # Parse JSON response
    try:
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", raw)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        return parsed["classification"], parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        # Fallback: look for keywords
        if "single-turn" in raw.lower():
            return "single-turn", f"Fallback parse: {raw[:200]}"
        elif "multi-turn" in raw.lower():
            return "multi-turn", f"Fallback parse: {raw[:200]}"
        return "multi-turn", f"Could not parse, defaulting to multi-turn: {raw[:200]}"


def main():
    parser = argparse.ArgumentParser(description="Split final dataset into single/multi-turn.")
    parser.add_argument("--input", type=str, required=True, help="Path to final JSONL (e.g., data/final_247.jsonl)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as input file)")
    parser.add_argument("--key", type=str, default=".openrouter_key", help="Path to OpenRouter API key file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    # Load data
    rows = []
    with open(input_path) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} conversations from {input_path}", flush=True)

    # Setup OpenRouter client
    api_key = Path(args.key).read_text().strip()
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Check for existing report to resume from
    stem = input_path.stem
    report_path = output_dir / f"split_report_{stem}.json"
    cached_details = {}
    if report_path.exists():
        with open(report_path) as f:
            prev_report = json.load(f)
        for d in prev_report.get("details", []):
            cached_details[d["preview"]] = d
        print(f"Resuming: loaded {len(cached_details)} cached classifications from {report_path}", flush=True)

    # Classify every row with Opus
    single_turn = []
    multi_turn = []
    details = []

    for i, row in enumerate(rows):
        user_input = row["user_input"]
        preview = user_input[:200]

        # Use cached classification if available
        if preview in cached_details:
            classification = cached_details[preview]["classification"]
            reasoning = cached_details[preview]["reasoning"]
            tag = "cached"
        else:
            classification, reasoning = classify_with_llm(client, user_input)
            tag = "new"

        detail = {
            "index": i,
            "classification": classification,
            "reasoning": reasoning,
            "preview": preview,
        }
        details.append(detail)

        if classification == "single-turn":
            single_turn.append(row)
        else:
            multi_turn.append(row)

        print(f"  [{i+1}/{len(rows)}] {classification} ({tag}) — {reasoning[:80]}", flush=True)

        # Save report incrementally so resume works even if we crash
        if tag == "new" and (i + 1) % 10 == 0:
            partial_report = {
                "total": len(rows),
                "single_turn_count": len(single_turn),
                "multi_turn_count": len(multi_turn),
                "details": details,
            }
            with open(report_path, "w") as f:
                json.dump(partial_report, f, indent=2)
            print(f"    (checkpoint saved at {i+1}/{len(rows)})", flush=True)

    # Derive output filenames from input filename
    single_path = output_dir / f"single_turn_{stem}.jsonl"
    multi_path = output_dir / f"multi_turn_{stem}.jsonl"

    # Write outputs
    with open(single_path, "w") as f:
        for row in single_turn:
            f.write(json.dumps(row) + "\n")

    with open(multi_path, "w") as f:
        for row in multi_turn:
            f.write(json.dumps(row) + "\n")

    report = {
        "total": len(rows),
        "single_turn_count": len(single_turn),
        "multi_turn_count": len(multi_turn),
        "details": details,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults:", flush=True)
    print(f"  Single-turn: {len(single_turn)} → {single_path}", flush=True)
    print(f"  Multi-turn:  {len(multi_turn)} → {multi_path}", flush=True)
    print(f"  Report:      {report_path}", flush=True)


if __name__ == "__main__":
    main()
