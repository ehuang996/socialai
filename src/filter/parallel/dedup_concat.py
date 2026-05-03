"""Concatenate per-shard partial files into one output, deduplicating by
`conversation_hash` (keeping the first occurrence). Used as the concat step in
the SLURM dependency chain since Stage 2 partials may contain duplicates from
the (now-fixed) shard-then-filter resumption bug.

Usage:
    uv run --no-sync python src/filter/parallel/dedup_concat.py \\
        --output_base data/2A_fabricated_personal_information_scores \\
        --output_file experiments/25_low_quality_filter/results/2A_fabricated_personal_information_scores.jsonl

After writing the output, deletes the per-shard part files so re-runs start
clean (matches the behavior of the simple `cat … && rm` concat the original
pipeline used).
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_base", required=True,
                    help="Per-shard prefix without _part_N.jsonl suffix.")
    ap.add_argument("--output_file", required=True,
                    help="Path to the concatenated, deduplicated output file.")
    args = ap.parse_args()

    shards = sorted(glob.glob(f"{args.output_base}_part_*.jsonl"))
    if not shards:
        raise SystemExit(f"No shard files matching {args.output_base}_part_*.jsonl")

    seen: set[str] = set()
    total_in = 0
    total_out = 0
    Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as out:
        for s in shards:
            with open(s) as f:
                for line in f:
                    total_in += 1
                    try:
                        h = json.loads(line)["conversation_hash"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if h in seen:
                        continue
                    seen.add(h)
                    out.write(line if line.endswith("\n") else line + "\n")
                    total_out += 1

    for s in shards:
        os.remove(s)

    dropped = total_in - total_out
    print(
        f"Dedup concat: {len(shards)} shards -> {args.output_file}\n"
        f"  rows in:  {total_in:,}\n"
        f"  rows out: {total_out:,} unique\n"
        f"  dropped duplicates: {dropped:,}"
    )


if __name__ == "__main__":
    main()
