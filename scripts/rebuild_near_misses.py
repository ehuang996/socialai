"""Rebuild data/near_misses.jsonl from Stage 3 XOR.

Reads every `experiments/<NN>_high_quality_filter/results/<measure>_high_quality.jsonl`
(Stage 3, GPT-4o-mini judge) and keeps rows where exactly one of
`chitchat_keep` / `category_keep` is true. Then:

  Phase 1 — dedup by `user_input`, merging `measure` into a list.
  Phase 3 — semantic dedup via sentence-transformers/all-MiniLM-L6-v2
            cosine similarity (default threshold 0.85).

Skips Stage 5's Phase 2 (single/multi-turn splitting) per design.

Output schema (slim): {user_input, measure, synthetic: false, language: "English"}.

Usage:
    uv run python scripts/rebuild_near_misses.py
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

STAGE3_MODEL = "gpt_4o_mini"


def discover_stage3_experiments(experiments_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (exp_num, measure, result_path) for each high_quality_filter run."""
    out = []
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^(\d+)_high_quality_filter$", d.name)
        if not m:
            continue
        results_dir = d / "results"
        if not results_dir.exists():
            continue
        for f in sorted(results_dir.iterdir()):
            fm = re.match(r"^(.+)_high_quality\.jsonl$", f.name)
            if fm:
                out.append((m.group(1), fm.group(1), f))
                break
    return out


def collect_xor(measures: list[tuple[str, str, Path]]) -> tuple[list[dict], dict]:
    """Phase 1: collect XOR rows from Stage 3, dedup by user_input.

    Returns (rows, stats). Each row is {user_input, measure: [list]}.
    """
    total = 0
    errors = 0
    kept_before_dedup = 0
    seen: "OrderedDict[str, dict]" = OrderedDict()

    for exp_num, measure, path in measures:
        measure_scanned = 0
        measure_kept = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                total += 1
                measure_scanned += 1
                resp = row.get("model_responses", {}).get(STAGE3_MODEL, {})
                if "error" in resp:
                    errors += 1
                    continue
                ck = resp.get("chitchat_keep")
                catk = resp.get("category_keep")
                if ck is None or catk is None:
                    errors += 1
                    continue
                if bool(ck) != bool(catk):  # XOR
                    kept_before_dedup += 1
                    measure_kept += 1
                    ui = row["user_input"]
                    if ui in seen:
                        if measure not in seen[ui]["measure"]:
                            seen[ui]["measure"].append(measure)
                    else:
                        seen[ui] = {"user_input": ui, "measure": [measure]}
        print(f"  {measure:42s}: scanned {measure_scanned:>6d}   XOR kept {measure_kept:>6d}")

    rows = list(seen.values())
    multi = sum(1 for r in rows if len(r["measure"]) > 1)
    stats = {
        "total_scanned": total,
        "errors": errors,
        "xor_before_user_input_dedup": kept_before_dedup,
        "after_user_input_dedup": len(rows),
        "multi_measure_rows": multi,
    }
    return rows, stats


def semantic_dedupe(rows: list[dict], threshold: float) -> tuple[list[dict], list[tuple[int, int, float]]]:
    """Phase 3: cosine-similarity dedup with sentence-transformers."""
    if not rows:
        return rows, []

    import numpy as np  # noqa: F401  (used indirectly via encode)
    from sentence_transformers import SentenceTransformer

    print(f"\nEmbedding {len(rows)} user_inputs with all-MiniLM-L6-v2...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(
        [r["user_input"] for r in rows],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    print("Computing pairwise similarity...", flush=True)
    sim = embs @ embs.T  # cosine since normalized

    keep: list[int] = []
    dropped: list[tuple[int, int, float]] = []
    for i in range(len(rows)):
        dup_of = None
        for j in keep:
            if sim[i, j] >= threshold:
                dup_of = j
                break
        if dup_of is None:
            keep.append(i)
        else:
            dropped.append((i, dup_of, float(sim[i, dup_of])))

    return [rows[i] for i in keep], dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild data/near_misses.jsonl from Stage 3 XOR.",
    )
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--output", type=str, default="data/near_misses.jsonl")
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Cosine-similarity threshold for semantic dedup (default: 0.85)",
    )
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Phase 1 — Collect Stage 3 XOR")
    print("=" * 64)
    measures = discover_stage3_experiments(experiments_dir)
    if not measures:
        raise SystemExit(f"No high_quality_filter experiments found in {experiments_dir}")
    print(f"Found {len(measures)} high_quality_filter experiments:")
    for exp_num, measure, path in measures:
        print(f"  {exp_num}_high_quality_filter -> {measure}")
    print()
    rows, phase1_stats = collect_xor(measures)
    print()
    for k, v in phase1_stats.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 64)
    print(f"Phase 3 — Semantic dedup (threshold={args.threshold})")
    print("=" * 64)
    deduped, dropped = semantic_dedupe(rows, args.threshold)
    print(f"Kept: {len(deduped)}  Dropped: {len(dropped)}")

    # Tag + emit slim schema
    out_rows = [
        {
            "user_input": r["user_input"],
            "measure": r["measure"],
            "synthetic": False,
            "language": "English",
        }
        for r in deduped
    ]
    with open(output_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_rows)} rows to {output_path}")

    report = {
        "source": "Stage 3 XOR (high_quality_filter / gpt_4o_mini)",
        "threshold": args.threshold,
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        **phase1_stats,
        "after_semantic_dedup": len(deduped),
        "semantic_dropped": len(dropped),
    }
    report_path = output_path.with_name(output_path.stem + ".dedup_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Dedup report: {report_path}")


if __name__ == "__main__":
    main()
