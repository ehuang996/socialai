"""Find intersection of chitchat_keep and category_keep from Stage 3 output."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Filter Stage 3 output to rows where both chitchat and category checks pass.")
    parser.add_argument("--input", required=True, help="Path to Stage 3 output JSONL")
    parser.add_argument("--output", required=True, help="Path to write filtered JSONL")
    parser.add_argument("--model", default="gpt_4o_mini", help="Model key in model_responses (default: gpt_4o_mini)")
    args = parser.parse_args()

    total = 0
    chitchat_only = 0
    category_only = 0
    both_keep = 0
    neither = 0
    errors = 0

    kept_rows = []

    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            total += 1

            model_resp = row.get("model_responses", {}).get(args.model, {})
            if "error" in model_resp:
                errors += 1
                continue

            c_keep = model_resp.get("chitchat_keep", False)
            k_keep = model_resp.get("category_keep", False)

            if c_keep and k_keep:
                both_keep += 1
                kept_rows.append(row)
            elif c_keep:
                chitchat_only += 1
            elif k_keep:
                category_only += 1
            else:
                neither += 1

    with open(args.output, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Total rows:        {total}")
    print(f"Errors:            {errors}")
    print(f"Both KEEP:         {both_keep}")
    print(f"Chitchat only:     {chitchat_only}")
    print(f"Category only:     {category_only}")
    print(f"Neither:           {neither}")
    print(f"\nWrote {both_keep} rows to {args.output}")


if __name__ == "__main__":
    main()
