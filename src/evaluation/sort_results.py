"""Sort Stage 7.2 judge results by user_input → measure → model family/recency."""

import argparse
import json

from src.evaluation._eval_common import ALL_MEASURES

MEASURE_ORDER = ALL_MEASURES

# Model order: grouped by family (provider), most recent / largest first within each.
# Keep in sync with the 23 API/direct columns plus local-HF qwen3_1_7b/qwen3_4b.
MODEL_ORDER = [
    # OpenAI (newest → oldest)
    "gpt_5_5",
    "gpt_5_4",
    "gpt_4o",
    "gpt_4o_mini",
    # Google
    "gemini_3_1_pro",
    "gemini_3_flash",
    "gemini_2_flash_001",
    # Anthropic
    "claude_opus_4_7",
    "claude_opus_4_6",
    "claude_opus_4_6_t2k",
    "claude_opus_4_6_t5k",
    "claude_opus_4_6_t10k",
    "claude_sonnet_4",
    # xAI
    "grok_4_3",
    "grok_4",
    "grok_3",
    # DeepSeek
    "deepseek_v4_pro",
    "deepseek_v4_flash",
    "deepseek_chat",
    # Qwen (max → smallest)
    "qwen3_6_max_preview",
    "qwen3_32b",
    "qwen3_14b",
    "qwen3_8b",
    "qwen3_4b",
    "qwen3_1_7b",
]

MEASURE_RANK = {m: i for i, m in enumerate(MEASURE_ORDER)}
MODEL_RANK = {m: i for i, m in enumerate(MODEL_ORDER)}


def sort_key(row):
    return (
        row["_input_rank"],
        MEASURE_RANK.get(row["measure"], 999),
        MODEL_RANK.get(row["model_name"], 999),
    )


def main():
    parser = argparse.ArgumentParser(description="Sort Stage 7.2 judge results.")
    parser.add_argument("--input", default="data/eval_judge_results.jsonl")
    parser.add_argument(
        "--output", default="data/eval_judge_results.jsonl",
        help="Output path (default: overwrite input)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]

    input_order = {}
    for row in rows:
        ui = row["user_input"]
        if ui not in input_order:
            input_order[ui] = len(input_order)
        row["_input_rank"] = input_order[ui]

    rows.sort(key=sort_key)

    with open(args.output, "w") as f:
        for row in rows:
            del row["_input_rank"]
            f.write(json.dumps(row) + "\n")

    print(f"Sorted {len(rows)} rows → {args.output}")
    print(f"  {len(input_order)} unique inputs")
    print(f"  {len(set(r['measure'] for r in rows))} measures")
    print(f"  {len(set(r['model_name'] for r in rows))} models")


if __name__ == "__main__":
    main()
