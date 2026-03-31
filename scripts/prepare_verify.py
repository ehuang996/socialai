"""Convert single_turn_104.jsonl to the format expected by Stage 1 (coarse_filter).

Stage 1 reads `conversation` (list of dicts) and `conversation_hash`.
This script adds those fields so single_turn_104.jsonl can be fed directly
into the pipeline starting at Stage 1.

Usage:
    uv run python scripts/prepare_verify.py
Output:
    data/single_turn_104_prepared.jsonl
"""

import hashlib
import json
from pathlib import Path

IN_PATH = Path("single_turn_104.jsonl")
OUT_PATH = Path("data/single_turn_104_prepared.jsonl")


def main():
    rows = []
    with open(IN_PATH) as f:
        for line in f:
            row = json.loads(line)
            user_input = row["user_input"]
            assistant_response = row["assistant_response"]

            row["conversation_hash"] = hashlib.sha256(
                (user_input + "\x00" + assistant_response).encode()
            ).hexdigest()
            row["conversation"] = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response},
            ]
            row.setdefault("timestamp", "")
            rows.append(row)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Prepared {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
