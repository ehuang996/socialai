"""Rebuild near_misses from NEW Stage 3 + Stage 4 XOR (the four re-run measures).

Differences from v1 (`scripts/rebuild_near_misses.py`):
  1. Sources: NEW Stage 3 / Stage 4 outputs at /project2/jessetho_1732 for the 4
     re-run measures (2A_fabricated_personal_information, 2B_emotion_expression
     (relaxed), 2C_flattery_tone, 2D_human_relationship_encouragement).
  2. Adds Stage 4 XOR (Opus 4.6) on top of Stage 3 XOR (GPT-4o-mini).
  3. 2A subsampling rule: Stage 3 XOR for 2A is huge (~59K), but only the
     `chitchat=False, category=True` partition is meaningful as near-miss
     candidates (the inverse partition has ~52K trivially-non-fabricated
     chitchat). Cap that partition at 7K rows.
  4. Dedup by user_input (merge measures into a list), then semantic cosine
     dedup at threshold 0.85.

Output: data/near_misses_v2.jsonl, schema {user_input, measure, source: [...]}.

Usage:
    cd /project2/jessetho_1732/wangzhu/socialai
    .venv/bin/python scripts/rebuild_near_misses_v2.py
"""
import argparse
import json
import random
from collections import OrderedDict, Counter
from pathlib import Path

# (measure, stage3_path, stage4_path)
NEW_SOURCES = [
    ("2A_fabricated_personal_information",
     "experiments/26_high_quality_filter/results/2A_fabricated_personal_information_high_quality.jsonl",
     "experiments/27_final_filter/results/2A_fabricated_personal_information_final.jsonl"),
    ("2B_emotion_expression",
     "experiments/28_high_quality_filter/results/2B_emotion_expression_high_quality.jsonl",  # relaxed
     "experiments/29_final_filter/results/2B_emotion_expression_final.jsonl"),               # relaxed
    ("2C_flattery_tone",
     "experiments/20_high_quality_filter/results/2C_flattery_tone_high_quality.jsonl",
     "experiments/21_final_filter/results/2C_flattery_tone_final.jsonl"),
    ("2D_human_relationship_encouragement",
     "experiments/23_high_quality_filter/results/2D_human_relationship_encouragement_high_quality.jsonl",
     "experiments/24_final_filter/results/2D_human_relationship_encouragement_final.jsonl"),
]
S3_MODEL = "gpt_4o_mini"
S4_MODEL = "claude_opus_4_6"

# 2A subsampling
SUBSAMPLE_2A_S3_CATEGORY_TRUE = 7000
SUBSAMPLE_SEED = 42


def collect_stage_xor(path: str, measure: str, model_key: str, stage_label: str,
                      filter_2a_category_only: bool = False) -> tuple[list[dict], dict]:
    """Read a Stage 3/4 file, return XOR rows and a stats dict."""
    if not Path(path).exists():
        return [], {"missing": True, "path": path}
    cat_only = []   # chitchat=False, category=True (interesting failure)
    chit_only = []  # chitchat=True, category=False
    err = 0
    total = 0
    for line in open(path):
        r = json.loads(line)
        mr = r.get("model_responses", {}).get(model_key, {})
        cc, ck = mr.get("chitchat_keep"), mr.get("category_keep")
        if cc is None or ck is None or "error" in mr:
            err += 1
            continue
        total += 1
        if cc and not ck:
            chit_only.append(r)
        elif (not cc) and ck:
            cat_only.append(r)
        # both / neither: skipped

    # 2A Stage 3 special rule: drop chitchat-only XOR; cap category-only at 7K
    if filter_2a_category_only:
        random.seed(SUBSAMPLE_SEED)
        before = len(cat_only)
        if len(cat_only) > SUBSAMPLE_2A_S3_CATEGORY_TRUE:
            cat_only = random.sample(cat_only, SUBSAMPLE_2A_S3_CATEGORY_TRUE)
        kept = cat_only
        stats = {
            "stage": stage_label, "total": total, "errors": err,
            "chitchat_only": len(chit_only), "category_only": before,
            "kept_after_2a_rule": len(kept),
        }
    else:
        kept = chit_only + cat_only
        stats = {
            "stage": stage_label, "total": total, "errors": err,
            "chitchat_only": len(chit_only), "category_only": len(cat_only),
            "xor_total": len(kept),
        }
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/near_misses_v2.jsonl")
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    print("=" * 70)
    print("Phase 1 — Collect Stage 3 + Stage 4 XOR per measure")
    print("=" * 70)
    raw: list[tuple[dict, str, str]] = []  # (row, measure, source_label)
    overall_stats: dict[str, dict] = {}

    for measure, s3_path, s4_path in NEW_SOURCES:
        print(f"\n## {measure}")
        # Stage 3 XOR
        is_2a = measure == "2A_fabricated_personal_information"
        s3_rows, s3_stats = collect_stage_xor(
            s3_path, measure, S3_MODEL, "stage3", filter_2a_category_only=is_2a)
        print(f"  Stage 3: {s3_stats}")
        for r in s3_rows:
            raw.append((r, measure, "stage3"))
        # Stage 4 XOR
        s4_rows, s4_stats = collect_stage_xor(
            s4_path, measure, S4_MODEL, "stage4", filter_2a_category_only=False)
        print(f"  Stage 4: {s4_stats}")
        for r in s4_rows:
            raw.append((r, measure, "stage4"))
        overall_stats[measure] = {"stage3": s3_stats, "stage4": s4_stats}

    print(f"\nTotal raw XOR rows collected: {len(raw):,}")

    # Phase 2 — dedup by user_input, merge measure + source labels
    print("\n" + "=" * 70)
    print("Phase 2 — Dedup by user_input")
    print("=" * 70)
    by_ui: "OrderedDict[str, dict]" = OrderedDict()
    for row, measure, source in raw:
        ui = row["user_input"]
        if ui in by_ui:
            entry = by_ui[ui]
            if measure not in entry["measure"]:
                entry["measure"].append(measure)
            label = f"{measure}:{source}"
            if label not in entry["source"]:
                entry["source"].append(label)
        else:
            by_ui[ui] = {
                "user_input": ui,
                "measure": [measure],
                "source": [f"{measure}:{source}"],
                "synthetic": False,
                "language": "English",
            }
    deduped_by_ui = list(by_ui.values())
    multi = sum(1 for r in deduped_by_ui if len(r["measure"]) > 1)
    print(f"  After user_input dedup: {len(deduped_by_ui):,} unique inputs")
    print(f"  Multi-measure rows:     {multi:,}")

    # Phase 3 — semantic dedup
    print("\n" + "=" * 70)
    print(f"Phase 3 — Semantic dedup (cosine ≥ {args.threshold})")
    print("=" * 70)
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding {len(deduped_by_ui)} user_inputs ...")
    embs = model.encode([r["user_input"] for r in deduped_by_ui],
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True)
    print("Computing pairwise similarity ...")
    sim = embs @ embs.T
    keep_idx = []
    dropped = []
    for i in range(len(deduped_by_ui)):
        dup = next((j for j in keep_idx if sim[i, j] >= args.threshold), None)
        if dup is None:
            keep_idx.append(i)
        else:
            dropped.append((i, dup, float(sim[i, dup])))
    final = [deduped_by_ui[i] for i in keep_idx]
    print(f"  Kept: {len(final):,}   Dropped: {len(dropped):,}")

    # Sort: by canonical measure order on first measure, then by user_input
    ORDER = ["1B_intentional_human_speech", "1B_human_pronoun",
             "1C_identity_transparency", "2A_fabricated_personal_information",
             "2B_emotion_expression", "2C_deference", "2C_flattery_tone",
             "2D_human_relationship_encouragement", "3A_engagement_hooks"]
    O = {m: i for i, m in enumerate(ORDER)}
    for r in final:
        r["measure"].sort(key=lambda m: O.get(m, 999))
        r["source"].sort()
    final.sort(key=lambda r: O.get(r["measure"][0], 999) if r["measure"] else 999)

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote → {out_path}  ({len(final):,} rows)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL DISTRIBUTION")
    print("=" * 70)
    counts = Counter()
    src_counts = Counter()
    for r in final:
        for m in r["measure"]:
            counts[m] += 1
        for s in r["source"]:
            src_counts[s] += 1
    print(f"\nPer-measure tag counts (rows count once per tag):")
    for m, c in counts.most_common():
        print(f"  {c:>6,}  {m}")
    print(f"  ----- {sum(counts.values()):>6,}  TOTAL TAGS")
    print(f"\nPer-source counts (measure:stageN):")
    for s, c in sorted(src_counts.items()):
        print(f"  {c:>6,}  {s}")

    # Dedup report
    rep = {
        "threshold": args.threshold,
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "stage_stats": overall_stats,
        "raw_xor_rows": len(raw),
        "after_user_input_dedup": len(deduped_by_ui),
        "after_semantic_dedup": len(final),
        "semantic_dropped": len(dropped),
        "per_measure_tags_final": dict(counts),
        "per_source_final": dict(src_counts),
    }
    rep_path = out_path.with_suffix(".dedup_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nReport → {rep_path}")


if __name__ == "__main__":
    main()
