"""Sort stage7_1_eval_results.jsonl by user_input, measure, then model family/recency."""

import argparse
import json

# Measure order: numeric prefix, then alphabetical within prefix
MEASURE_ORDER = [
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

# Model order: grouped by family, most recent first within each family
MODEL_ORDER = [
    # OpenAI family (newest → oldest)
    "gpt5_4_pro",
    "gpt5_4",
    "gpt5_3",
    "o4_mini",
    "gpt4o_mini",
    # Gemini family (newest → oldest)
    "gemini3_1_pro",
    "gemini3_flash",
    "gemini2_flash_001",
    # Claude family (newest → oldest)
    "claude_opus",
    "claude_sonnet",
    "claude_sonnet_4",
    "claude_haiku",
    # Grok family (newest → oldest)
    "grok4",
    "grok3_mini_beta",
]

MEASURE_RANK = {m: i for i, m in enumerate(MEASURE_ORDER)}
MODEL_RANK = {m: i for i, m in enumerate(MODEL_ORDER)}


def sort_key(row):
    # 1. Group by user_input (stable: first appearance order, assigned below)
    # 2. Measure order
    # 3. Model family + recency order
    return (
        row["_input_rank"],
        MEASURE_RANK.get(row["measure"], 999),
        MODEL_RANK.get(row["model_name"], 999),
    )


def main():
    parser = argparse.ArgumentParser(description="Sort stage 7.1 eval results.")
    parser.add_argument(
        "--input",
        default="data/stage7_1_eval_results.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/stage7_1_eval_results.jsonl",
        help="Output path (default: overwrite input)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f]

    # Assign input rank by first appearance order
    input_order = {}
    for row in rows:
        ui = row["user_input"]
        if ui not in input_order:
            input_order[ui] = len(input_order)
        row["_input_rank"] = input_order[ui]

    rows.sort(key=sort_key)

    # Remove temp field and write
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
