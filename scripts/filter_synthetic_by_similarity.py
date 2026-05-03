"""Filter synthetic_data_full.jsonl by cosine similarity between user_input and source_input.

Keeps rows where sim(user_input, source_input) >= threshold (default 0.75),
writes them to synthetic_data.jsonl, and prints per-measure counts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_data_full.jsonl")
    parser.add_argument("--output", default="data/synthetic_data.jsonl")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    print(f"loaded {len(rows)} rows from {args.input}")

    model = SentenceTransformer(args.model)

    user_inputs = [r["user_input"] for r in rows]
    source_inputs = [r["source_input"] for r in rows]

    # NOTE: [pedagogical] normalize_embeddings=True gives unit vectors, so the dot
    # product equals cosine similarity — no need to divide by norms.
    user_emb = model.encode(user_inputs, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    source_emb = model.encode(source_inputs, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)

    sims = (user_emb * source_emb).sum(axis=1)

    kept = [r for r, s in zip(rows, sims) if s >= args.threshold]
    print(f"kept {len(kept)} / {len(rows)} rows with cosine sim >= {args.threshold}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.output}")

    # NOTE: [thought process] `measure` is a list (a row can be flagged for multiple
    # categories), so a row contributes to every category it lists.
    counts = Counter()
    for row in kept:
        for measure in row["measure"]:
            counts[measure] += 1

    print("\nper-measure counts (rows kept):")
    for measure, count in sorted(counts.items()):
        print(f"  {measure}: {count}")


if __name__ == "__main__":
    main()
