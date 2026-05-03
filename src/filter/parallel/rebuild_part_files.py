"""Rebuild per-shard partial files from a fully-concatenated stage output.

When a Stage 2 concat job runs (and rm's the per-shard files) but the output
ended up incomplete because some shards were PREEMPTED, we need to redo those
shards. To preserve the existing work, we redistribute the rows already in the
concat output back into per-shard files (using each row's `conversation_hash`
position in the original Stage 1 input, mod num_shards). Then make-up runs for
the missing shards resume from those files via Judge.load_completed_ids().

Usage:
    uv run --no-sync python src/filter/parallel/rebuild_part_files.py \\
        --input_path experiments/00_coarse_filter/results/1B_intentional_human_speech_coarse.jsonl \\
        --concat_output experiments/16_low_quality_filter/results/2B_emotion_expression_scores.jsonl \\
        --output_base data/2B_emotion_expression_scores \\
        --num_shards 8
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True,
                    help="Stage 1 (coarse) input JSONL, used to map hash -> position.")
    ap.add_argument("--concat_output", required=True,
                    help="Existing concatenated stage output (whose rows we re-distribute).")
    ap.add_argument("--output_base", required=True,
                    help="Per-shard prefix; writes <base>_part_<N>.jsonl.")
    ap.add_argument("--num_shards", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading input hash -> position from {args.input_path} ...")
    hash_pos: dict[str, int] = {}
    with open(args.input_path) as f:
        for i, line in enumerate(f):
            try:
                h = json.loads(line)["conversation_hash"]
            except (json.JSONDecodeError, KeyError):
                continue
            hash_pos[h] = i
    print(f"  {len(hash_pos):,} input rows indexed")

    # Open per-shard files in write mode (truncate any existing)
    out_dir = Path(args.output_base).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        N: open(f"{args.output_base}_part_{N}.jsonl", "w")
        for N in range(args.num_shards)
    }
    counts = [0] * args.num_shards
    missing = 0

    print(f"Distributing rows from {args.concat_output} ...")
    with open(args.concat_output) as f:
        for line in f:
            try:
                h = json.loads(line)["conversation_hash"]
            except (json.JSONDecodeError, KeyError):
                continue
            pos = hash_pos.get(h)
            if pos is None:
                missing += 1
                continue
            sid = pos % args.num_shards
            files[sid].write(line)
            counts[sid] += 1

    for f in files.values():
        f.close()

    total = sum(counts)
    print(f"\nWrote {total:,} rows across {args.num_shards} shards")
    for N, c in enumerate(counts):
        print(f"  shard {N}: {c:,} rows")
    if missing:
        print(f"WARNING: {missing} rows had hashes not in input (skipped)")


if __name__ == "__main__":
    main()
