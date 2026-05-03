"""Merge local-HF Stage 7.1 response shards into the main response JSONL.

The local generator writes one model column per row in part files like:

    data/eval_responses_local_qwen_qwen3_4b_part_0.jsonl

This script folds those cells into data/eval_responses.jsonl while preserving
the same row schema produced by generate_responses.py. It can also initialize
the main response file from data/final_dataset.jsonl if the API/direct sweep
has not produced a file yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def response_shell(row: dict) -> dict:
    return {
        "user_input": row["user_input"],
        "measure": row.get("measure", []),
        "synthetic": row.get("synthetic", False),
        "language": row.get("language", "English"),
        "model_responses": row.get("model_responses", {}),
    }


def expand_part_specs(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        matches = sorted(Path(p) for p in glob.glob(spec))
        if not matches:
            raise SystemExit(f"No local response part files matched: {spec}")
        paths.extend(matches)
    return paths


def atomic_write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge local-HF Stage 7.1 response part files."
    )
    parser.add_argument("--input", default="data/final_dataset.jsonl")
    parser.add_argument("--base-output", default="data/eval_responses.jsonl")
    parser.add_argument(
        "--local-parts",
        nargs="*",
        default=["data/eval_responses_local_qwen_*.jsonl"],
        help="Glob(s) for local response part files.",
    )
    parser.add_argument("--output", default="data/eval_responses.jsonl")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model cells even if they contain assistant_response.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    base_output = Path(args.base_output)
    output_path = Path(args.output)

    input_rows = [response_shell(row) for row in read_jsonl(input_path)]
    if not input_rows:
        raise SystemExit(f"No rows found in {input_path}")

    row_order = [row["user_input"] for row in input_rows]
    rows_by_input = {row["user_input"]: row for row in input_rows}

    base_rows = read_jsonl(base_output)
    for row in base_rows:
        user_input = row.get("user_input")
        if user_input not in rows_by_input:
            row_order.append(user_input)
        rows_by_input[user_input] = response_shell(row)

    part_paths = expand_part_specs(args.local_parts)
    merged = Counter()
    skipped_success = Counter()
    seen_part_rows = 0

    for part_path in part_paths:
        for local_row in read_jsonl(part_path):
            seen_part_rows += 1
            user_input = local_row.get("user_input")
            if user_input not in rows_by_input:
                raise SystemExit(
                    f"{part_path} contains user_input not present in {input_path}: "
                    f"{str(user_input)[:120]}"
                )
            responses = local_row.get("model_responses", {})
            if not isinstance(responses, dict):
                continue

            dest = rows_by_input[user_input].setdefault("model_responses", {})
            for model_name, model_payload in responses.items():
                existing = dest.get(model_name, {})
                existing_ok = isinstance(existing, dict) and bool(
                    existing.get("assistant_response")
                )
                incoming_ok = isinstance(model_payload, dict) and bool(
                    model_payload.get("assistant_response")
                )
                if existing_ok and not incoming_ok and not args.overwrite:
                    skipped_success[model_name] += 1
                    continue
                dest[model_name] = model_payload
                merged[model_name] += 1

    output_rows = [rows_by_input[user_input] for user_input in row_order]
    atomic_write_jsonl(output_rows, output_path)

    print(f"Read {len(input_rows)} input rows from {input_path}")
    print(f"Read {len(base_rows)} base response rows from {base_output}")
    print(f"Read {seen_part_rows} local part rows from {len(part_paths)} file(s)")
    print(f"Wrote {len(output_rows)} merged response rows -> {output_path}")
    print("Merged cells:")
    for model_name, count in sorted(merged.items()):
        print(f"  {model_name}: {count}")
    if skipped_success:
        print("Preserved existing successful cells:")
        for model_name, count in sorted(skipped_success.items()):
            print(f"  {model_name}: {count}")


if __name__ == "__main__":
    main()
