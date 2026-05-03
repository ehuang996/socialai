"""Stage 5 v2 — combined NEW + OLD final_filter outputs, batched single/multi
classification, NO chitchat veto.

Differences from v1 (`data_preprocessing.py`):
  1. Reads from a hand-curated list of (source_path, measure) pairs spanning
     BOTH project locations (NEW jessetho_1732 + OLD robinjia_875), so we mix
     re-run new-prompt measures with measures that didn't get re-run.
  2. NO chitchat veto. v1's veto dropped any user_input with chitchat_keep=false
     in any measure; v2 keeps every both-keep row regardless.
  3. Batched single/multi classification: one Opus 4.6 call labels 10 messages
     at a time instead of one. ~10x fewer API calls. DSPy disk cache.
  4. Phase 3 (semantic dedup) runs only on the single-turn subset, same as v1.

Usage:
    cd /project2/jessetho_1732/wangzhu/socialai
    .venv/bin/python src/data_preprocessing/data_preprocessing_v2.py \\
        --output data/final_v2.jsonl \\
        --key .openrouter_key
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

# DSPy cache configured before import
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")
import dspy


# Sources: (final_filter_jsonl_path, measure_name)
# OLD = robinjia_875 (1B/1C/3A still using old prompts), NEW = jessetho_1732 (2A/2B/2C_flattery/2D rerun)
OLD = "/project2/robinjia_875/ehuang97/socialai/experiments"
NEW = "/project2/jessetho_1732/wangzhu/socialai/experiments"
SOURCES = [
    (f"{OLD}/23_final_filter/results/1B_intentional_human_speech_final.jsonl", "1B_intentional_human_speech"),
    (f"{OLD}/24_final_filter/results/1B_human_pronoun_final.jsonl",          "1B_human_pronoun"),
    (f"{OLD}/25_final_filter/results/1C_identity_transparency_final.jsonl",  "1C_identity_transparency"),
    (f"{OLD}/33_final_filter/results/3A_engagement_hooks_final.jsonl",       "3A_engagement_hooks"),
    (f"{NEW}/27_final_filter/results/2A_fabricated_personal_information_final.jsonl",   "2A_fabricated_personal_information"),
    (f"{NEW}/29_final_filter/results/2B_emotion_expression_final.jsonl",                "2B_emotion_expression"),
    (f"{NEW}/21_final_filter/results/2C_flattery_tone_final.jsonl",                     "2C_flattery_tone"),
    (f"{NEW}/24_final_filter/results/2D_human_relationship_encouragement_final.jsonl",  "2D_human_relationship_encouragement"),
]
MODEL_KEY = "claude_opus_4_6"

BATCH_SIZE = 10
MAX_MSG_CHARS = 6000  # truncate per-message text fed to the classifier

CLASSIFY_SYSTEM_PROMPT = """\
You are classifying user messages from chatbot conversations.

For EACH numbered message below, decide whether it is **single-turn** or **multi-turn**:

- **single-turn**: A standalone message from ONE user to a chatbot. May be short or long, may contain lists, code, JSON, or short quoted snippets — but it is ONE message from ONE user directed at the chatbot.
- **multi-turn**: Contains a pasted conversation transcript — multiple exchanges between parties with role labels such as "User:", "Assistant:", "Human:", "AI:", "Bot:", "System:". The user is sharing or continuing a prior conversation rather than writing fresh.

Edge cases:
- Role labels INSIDE INSTRUCTIONS (e.g., "When the user says X, respond with Y") → single-turn.
- A short quoted dialogue snippet WITHIN a larger standalone request → single-turn.
- Primarily a multi-exchange transcript (even if it ends with a new request) → multi-turn.

Respond with ONLY a JSON object. The "classifications" array must have exactly the same number of entries as input messages, in the same order:
{
  "classifications": [
    {"id": 1, "classification": "single-turn", "reasoning": "..."},
    {"id": 2, "classification": "multi-turn", "reasoning": "..."}
  ]
}"""


# ---------------------------------------------------------------------------
# Phase 1 — Collect and dedupe (NO chitchat veto)
# ---------------------------------------------------------------------------

def collect_rows() -> tuple[list[dict], dict]:
    """Read all 8 sources, keep both-keep rows, dedupe by user_input."""
    seen: "OrderedDict[str, dict]" = OrderedDict()
    per_measure_kept: dict[str, int] = {}
    per_measure_in_file: dict[str, int] = {}
    for path, measure in SOURCES:
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping {measure}")
            per_measure_kept[measure] = 0
            continue
        kept = total = 0
        for line in open(path):
            row = json.loads(line)
            total += 1
            mr = row.get("model_responses", {}).get(MODEL_KEY, {})
            if "error" in mr:
                continue
            cc = mr.get("chitchat_keep")
            ck = mr.get("category_keep")
            if not (cc and ck):
                continue
            kept += 1
            ui = row["user_input"]
            if ui in seen:
                if measure not in seen[ui]["measure"]:
                    seen[ui]["measure"].append(measure)
            else:
                seen[ui] = {
                    "user_input": ui,
                    "assistant_response": row.get("assistant_response", ""),
                    "timestamp": row.get("timestamp", ""),
                    "measure": [measure],
                }
        per_measure_kept[measure] = kept
        per_measure_in_file[measure] = total
        print(f"  {measure}: {kept:>4} both-keep / {total:>4} total in source file")
    return list(seen.values()), per_measure_kept


# ---------------------------------------------------------------------------
# Phase 2 — Batched single/multi classification with Opus 4.6
# ---------------------------------------------------------------------------

def parse_classifications(raw: str, n_expected: int) -> list[tuple[str, str]] | None:
    """Parse Opus's JSON response with n_expected classifications. Returns None if malformed."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try to find the first { ... } block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    items = data.get("classifications") or []
    if len(items) != n_expected:
        return None
    out = []
    for item in items:
        c = item.get("classification", "").strip().lower()
        if c not in ("single-turn", "multi-turn"):
            return None
        out.append((c, item.get("reasoning", "")))
    return out


def classify_batch(lm: dspy.LM, messages: list[str], max_retries: int = 3) -> list[tuple[str, str]]:
    """Send up to BATCH_SIZE messages in one prompt; return per-message (label, reasoning)."""
    user_block = "\n\n".join(
        f"[{i+1}] {(m or '')[:MAX_MSG_CHARS]}" for i, m in enumerate(messages)
    )
    payload = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Messages:\n\n{user_block}"},
    ]
    last_raw = ""
    for attempt in range(max_retries):
        try:
            kwargs = {"rollout_id": attempt} if attempt > 0 else {}
            response = lm(messages=payload, **kwargs)
            raw = response[0]
            if isinstance(raw, dict):
                raw = raw.get("content", "") or raw.get("text", "") or str(raw)
            last_raw = raw.strip()
            parsed = parse_classifications(last_raw, len(messages))
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  batch retry {attempt+1}/{max_retries} after {type(e).__name__}: {e}", flush=True)
            time.sleep(2 ** attempt)
    # Fallback: all multi-turn (conservative — multi-turn rows are kept untouched)
    print(f"  WARN: batch parse failed after {max_retries} attempts; defaulting all to multi-turn", flush=True)
    print(f"  raw: {last_raw[:300]}")
    return [("multi-turn", "parse-failed")] * len(messages)


def classify_all(rows: list[dict], lm: dspy.LM, report_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify every row in batches of BATCH_SIZE. Returns (single_turn, multi_turn, per-row details)."""
    # Resume from cached classifications keyed on user_input prefix (matches v1 behavior).
    cached: dict[str, dict] = {}
    if report_path.exists():
        try:
            prev = json.loads(report_path.read_text())
            for d in prev.get("details", []):
                cached[d["preview"]] = d
            if cached:
                print(f"Resuming: loaded {len(cached)} cached classifications from {report_path}")
        except (json.JSONDecodeError, KeyError):
            pass

    single, multi, details = [], [], []
    pending_idxs: list[int] = []
    pending_msgs: list[str] = []

    def flush_pending():
        if not pending_idxs:
            return
        results = classify_batch(lm, pending_msgs)
        for idx, (label, reasoning) in zip(pending_idxs, results):
            row = rows[idx]
            preview = row["user_input"][:200]
            d = {
                "index": idx,
                "classification": label,
                "reasoning": reasoning,
                "preview": preview,
                "measure": row["measure"],
            }
            details.append(d)
            (single if label == "single-turn" else multi).append(row)
        pending_idxs.clear()
        pending_msgs.clear()
        # checkpoint after each batch
        report_path.write_text(json.dumps(
            {"total": len(rows), "details": details}, indent=2))

    for i, row in enumerate(rows):
        preview = row["user_input"][:200]
        if preview in cached:
            label = cached[preview]["classification"]
            reasoning = cached[preview]["reasoning"]
            d = {"index": i, "classification": label, "reasoning": f"(cached) {reasoning[:200]}",
                 "preview": preview, "measure": row["measure"]}
            details.append(d)
            (single if label == "single-turn" else multi).append(row)
            print(f"  [{i+1}/{len(rows)}] {label} (cached)", flush=True)
            continue
        pending_idxs.append(i)
        pending_msgs.append(row["user_input"])
        if len(pending_idxs) >= BATCH_SIZE:
            print(f"  classifying batch ending at row {i+1}/{len(rows)}", flush=True)
            flush_pending()

    flush_pending()  # final partial batch

    # final report
    report_path.write_text(json.dumps({
        "total": len(rows),
        "single_turn_count": len(single),
        "multi_turn_count": len(multi),
        "batch_size": BATCH_SIZE,
        "details": details,
    }, indent=2))
    return single, multi, details


# ---------------------------------------------------------------------------
# Phase 3 — Semantic dedup of single-turn (unchanged from v1)
# ---------------------------------------------------------------------------

def semantic_dedup(rows: list[dict], threshold: float, report_path: Path) -> list[dict]:
    if not rows:
        report_path.write_text(json.dumps({"threshold": threshold, "kept_count": 0, "dropped_count": 0, "dropped": []}, indent=2))
        return []
    import numpy as np
    from sentence_transformers import SentenceTransformer
    print(f"\nSemantic dedup ({len(rows)} rows, threshold={threshold})", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode([r["user_input"] for r in rows],
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sim = embs @ embs.T
    keep_idx = []
    dropped = []
    for i in range(len(rows)):
        dup_of = next((j for j in keep_idx if sim[i, j] >= threshold), None)
        if dup_of is None:
            keep_idx.append(i)
        else:
            dropped.append((i, dup_of, float(sim[i, dup_of])))
    report = {
        "threshold": threshold,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "original_count": len(rows),
        "kept_count": len(keep_idx),
        "dropped_count": len(dropped),
        "dropped": sorted(
            [{"dropped_preview": rows[d]["user_input"][:200],
              "kept_preview": rows[k]["user_input"][:200],
              "similarity": round(s, 4)} for d, k, s in dropped],
            key=lambda x: -x["similarity"]),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  kept {len(keep_idx)}, dropped {len(dropped)}", flush=True)
    return [rows[i] for i in keep_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Combined output JSONL path; siblings get the split + reports.")
    ap.add_argument("--key", default=".openrouter_key", help="OpenRouter API key file.")
    ap.add_argument("--dedupe_threshold", type=float, default=0.85,
                    help="Cosine threshold for Phase 3 single-turn dedup (default: 0.85).")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    single_path = out_dir / f"single_turn_{stem}.jsonl"
    multi_path = out_dir / f"multi_turn_{stem}.jsonl"
    split_report = out_dir / f"split_report_{stem}.json"
    dedup_report = out_dir / f"dedup_report_{stem}.json"

    print("=" * 60)
    print("Phase 1: Collect both-keep rows (NO chitchat veto)")
    print("=" * 60)
    rows, per_measure_kept = collect_rows()
    print(f"\nTotal post-dedup unique user_inputs: {len(rows)}")
    multi_measure = sum(1 for r in rows if len(r["measure"]) > 1)
    print(f"  Single-measure: {len(rows) - multi_measure}")
    print(f"  Multi-measure:  {multi_measure}")
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  Wrote → {out_path}")

    print("\n" + "=" * 60)
    print(f"Phase 2: Single/multi-turn classification (Opus 4.6, batches of {BATCH_SIZE})")
    print("=" * 60)
    api_key = Path(args.key).read_text().strip()
    lm = dspy.LM(
        model="openrouter/anthropic/claude-opus-4-6",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=4096,  # batch of 10 needs more space than one
        temperature=0,
        cache=True,
        timeout=180,
    )
    single, multi, details = classify_all(rows, lm, split_report)
    # write slim single-turn (user_input + measure) and full multi-turn
    with open(single_path, "w") as f:
        for r in single:
            f.write(json.dumps({"user_input": r["user_input"], "measure": r["measure"]}) + "\n")
    with open(multi_path, "w") as f:
        for r in multi:
            f.write(json.dumps(r) + "\n")
    print(f"\nSingle-turn: {len(single)} → {single_path}")
    print(f"Multi-turn:  {len(multi)} → {multi_path}")
    print(f"Report:       {split_report}")

    print("\n" + "=" * 60)
    print(f"Phase 3: Semantic dedup of single-turn (cosine ≥ {args.dedupe_threshold})")
    print("=" * 60)
    # Re-read slim single-turn file so we dedup on the same shape v1 used
    deduped = semantic_dedup(single, args.dedupe_threshold, dedup_report)
    with open(single_path, "w") as f:
        for r in deduped:
            slim = {"user_input": r["user_input"], "measure": r["measure"]}
            f.write(json.dumps(slim) + "\n")
    print(f"\nFinal single-turn (deduped): {len(deduped)} → {single_path}")

    # Tag both files with synthetic / language for downstream compatibility
    print("\n" + "=" * 60)
    print("Tagging rows with synthetic=false, language=English")
    print("=" * 60)
    for path in (out_path, single_path, multi_path):
        if not path.exists():
            continue
        rs = [json.loads(l) for l in open(path)]
        for r in rs:
            r["synthetic"] = False
            r["language"] = "English"
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(f"  tagged {len(rs)} rows in {path.name}")

    # Final per-measure × turn-type distribution
    print("\n" + "=" * 60)
    print("FINAL DISTRIBUTION")
    print("=" * 60)
    from collections import Counter
    s_counts = Counter()
    m_counts = Counter()
    for r in deduped:
        for m in r["measure"]:
            s_counts[m] += 1
    for r in multi:
        for m in r["measure"]:
            m_counts[m] += 1
    measures_in_order = [m for _, m in SOURCES]
    print(f"{'measure':<42} {'single (deduped)':>18} {'multi':>10}")
    print("-" * 75)
    for m in measures_in_order:
        print(f"{m:<42} {s_counts[m]:>18} {m_counts[m]:>10}")
    print("-" * 75)
    print(f"{'Sum across measures (rows count once per measure tag)':<42} {sum(s_counts.values()):>18} {sum(m_counts.values()):>10}")
    print(f"\nUnique rows: single (deduped) = {len(deduped)},  multi = {len(multi)}")

    print(f"\nFiles:")
    print(f"  combined (post Phase 1):  {out_path}")
    print(f"  single-turn (final, deduped): {single_path}")
    print(f"  multi-turn:               {multi_path}")
    print(f"  split report:             {split_report}")
    print(f"  dedup report:             {dedup_report}")


if __name__ == "__main__":
    main()
