"""Collect both-KEEP rows from Stage 4 (final_filter) experiments into a single JSONL file.

Reads final_filter experiments and writes only rows where the judge model returned
both chitchat_keep=true AND category_keep=true.  Deduplicates by user_input so
that conversations appearing in multiple measures get a single row with
measure as a list (e.g. ["1B_human_disfluencies", "2C_sycophancy"]).
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


MEASURES = [
    ("19", "1B_human_disfluencies"),
    ("20", "1C_identity_transparency"),
    ("21", "2A_fabricated_personal_details"),
    ("22", "2B_explicit_emotions"),
    ("23", "2B_implicit_emotions"),
    ("24", "2B_romantic_bonding"),
    ("25", "2C_sycophancy"),
    ("26", "2D_human_relationship_encouragement"),
    ("27", "3A_engagement_hooks"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Collect both-KEEP rows from final_filter experiments."
    )
    parser.add_argument(
        "--experiments_dir", type=str, default="experiments",
        help="Root experiments directory (default: experiments)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write the collected JSONL file",
    )
    parser.add_argument(
        "--model", type=str, default="claude_opus_4_6",
        help="Model key in model_responses (default: claude_opus_4_6)",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    total = 0
    kept_before_dedup = 0
    errors = 0

    # Collect all both-KEEP rows, keyed by user_input for deduplication
    # OrderedDict preserves insertion order (first measure seen)
    seen = OrderedDict()  # user_input -> row dict (with measure as list)

    for exp_num, measure in MEASURES:
        result_path = (
            experiments_dir
            / f"{exp_num}_final_filter"
            / "results"
            / f"{measure}_final.jsonl"
        )
        if not result_path.exists():
            print(f"WARNING: {result_path} not found, skipping")
            continue

        measure_kept = 0
        with open(result_path) as f_in:
            for line in f_in:
                row = json.loads(line)
                total += 1
                resp = row.get("model_responses", {}).get(args.model, {})
                if "error" in resp:
                    errors += 1
                    continue
                if resp.get("chitchat_keep") and resp.get("category_keep"):
                    kept_before_dedup += 1
                    measure_kept += 1
                    ui = row["user_input"]
                    if ui in seen:
                        # Add measure to existing entry (avoid duplicates)
                        if measure not in seen[ui]["measure"]:
                            seen[ui]["measure"].append(measure)
                    else:
                        seen[ui] = {
                            "user_input": ui,
                            "assistant_response": row["assistant_response"],
                            "timestamp": row.get("timestamp", ""),
                            "measure": [measure],
                        }

        print(f"{measure}: {measure_kept} kept")

    with open(args.output, "w") as f_out:
        for row in seen.values():
            f_out.write(json.dumps(row) + "\n")

    multi = sum(1 for r in seen.values() if len(r["measure"]) > 1)
    print(f"\nTotal rows scanned: {total}")
    print(f"Errors: {errors}")
    print(f"Both KEEP (before dedup): {kept_before_dedup}")
    print(f"After dedup: {len(seen)} unique conversations")
    print(f"  Single-measure: {len(seen) - multi}")
    print(f"  Multi-measure:  {multi}")
    print(f"Wrote {len(seen)} rows to {args.output}")


if __name__ == "__main__":
    main()
