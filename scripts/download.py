"""Download WildChat from HuggingFace, filter English, deduplicate, and save to JSONL.

Default dataset is `allenai/WildChat-4.8M` (the public, non-gated release — ~3.2M
conversations). The full 4.8M version lives in the request-only repo
`allenai/WildChat-4.8M-Full` and is not used here. The legacy `allenai/WildChat-1M`
is also supported via --dataset_name.

Run once (no SLURM array) via:
    sbatch slurm/run_download.sbatch
Or directly:
    uv run python scripts/download.py --output_path data/wildchat_raw.jsonl
"""
import argparse
import json
import os
from pathlib import Path

HF_CACHE = "/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface"


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess WildChat to JSONL.")
    parser.add_argument("--output_path", type=str, default="data/wildchat_raw.jsonl")
    parser.add_argument("--dataset_name", type=str, default="allenai/WildChat-4.8M")
    parser.add_argument("--hf_cache", type=str, default=HF_CACHE)
    args = parser.parse_args()

    # Set HF cache before importing datasets
    os.environ["HF_HOME"] = args.hf_cache

    from datasets import load_dataset

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset_name} (cache: {args.hf_cache})...")
    dataset = load_dataset(args.dataset_name, split="train", cache_dir=args.hf_cache)
    print(f"Total rows: {len(dataset)}")

    # Filter English-only conversations
    dataset = dataset.filter(lambda x: x["language"] == "English")
    print(f"After English filter: {len(dataset)}")

    # Deduplicate by conversation_hash (preserving first occurrence)
    seen_hashes: set = set()
    unique_indices = []
    for i, h in enumerate(dataset["conversation_hash"]):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    dataset = dataset.select(unique_indices)
    print(f"After deduplication: {len(dataset)}")

    # Write to JSONL
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        for row in dataset:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"Done. Wrote {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
