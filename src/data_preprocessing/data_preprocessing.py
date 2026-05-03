"""Stage 5 — Data Preprocessing.

Merges the former Stage 5 (collect_final) and Stage 5.5 (split_turns) into a
single preprocessing step, followed by a data-cleaning pass.

Phase 1 — Collect & Deduplicate
    Reads all `<NN>_final_filter` experiments and keeps rows matching the
    selected pool, then deduplicates by `user_input`:
      - `--pool both_keep` (default): rows where the judge returned both
        `chitchat_keep=true` AND `category_keep=true` (positives).
      - `--pool xor`: rows where exactly one of the two keeps is true
        (negatives, used as few-shot negatives for synthetic generation).
    The global chitchat veto is applied only in `both_keep` mode (domain gate
    for positives). In `xor` mode the veto is skipped, since XOR rows include
    `chitchat_keep=false` by definition.
    Conversations appearing in multiple measures get a single row with
    `measure` as a list.

Phase 2 — Split Single-Turn vs Multi-Turn
    Uses Claude Opus 4.6 via OpenRouter to classify each `user_input` as either
    a clean single-turn message or a pasted multi-turn transcript. Writes two
    additional JSONL files alongside the combined output.

Phase 3 — Data Cleaning
    Semantic deduplication of the single-turn file using sentence-transformers
    (all-MiniLM-L6-v2) + cosine similarity (threshold 0.90 by default). Then
    tags every row in all three output files with `synthetic: false` and
    `language: "English"` to prepare for later synthetic-data mixing.

Outputs (given --output data/final.jsonl):
  - data/final.jsonl                  (combined, deduplicated)
  - data/single_turn_final.jsonl      (clean user messages, semantic-deduped)
  - data/multi_turn_final.jsonl       (pasted transcripts)
  - data/split_report_final.json      (classification details)
  - data/dedup_report_final.json      (semantic-dedup decisions)
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

# Point DSPy disk cache at the project-local cache/dspy/ directory.
# Must be set BEFORE `import dspy` since dspy reads the env var at import time.
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")

import dspy


# ---------------------------------------------------------------------------
# Phase 1: Collect & Deduplicate
# ---------------------------------------------------------------------------

def discover_final_filter_experiments(experiments_dir: Path) -> list[tuple[str, str]]:
    """Auto-discover final_filter experiments by scanning the experiments directory.

    Returns list of (exp_num, measure_name) tuples sorted by experiment number.
    """
    measures = []
    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^(\d+)_final_filter$", d.name)
        if not m:
            continue
        exp_num = m.group(1)
        results_dir = d / "results"
        if not results_dir.exists():
            continue
        for f in results_dir.iterdir():
            fm = re.match(r"^(.+)_final\.jsonl$", f.name)
            if fm:
                measures.append((exp_num, fm.group(1)))
                break
    return measures


def collect_and_dedupe(
    experiments_dir: Path,
    output_path: Path,
    model: str,
    experiments_arg: str | None,
    pool: str = "both_keep",
) -> list[dict]:
    """Collect rows from final_filter experiments and deduplicate.

    pool:
      - "both_keep": chitchat_keep AND category_keep (positives). The global
        chitchat-keep veto is applied: any user_input with chitchat_keep=false
        in at least one measure is dropped entirely, enforcing the chitchat
        domain gate.
      - "xor":       exactly one of chitchat_keep / category_keep (negatives).
        No chitchat veto — by definition XOR includes chitchat_keep=false rows,
        and applying a global veto would zero out half of the XOR pool.
    """
    if pool not in {"both_keep", "xor"}:
        raise ValueError(f"pool must be 'both_keep' or 'xor', got {pool!r}")
    apply_veto = pool == "both_keep"
    if experiments_arg:
        measures = []
        for entry in experiments_arg.split(","):
            exp_num, measure = entry.strip().split(":")
            measures.append((exp_num, measure))
    else:
        measures = discover_final_filter_experiments(experiments_dir)
        if not measures:
            raise SystemExit(f"ERROR: No final_filter experiments found in {experiments_dir}")

    print(f"Found {len(measures)} final_filter experiments:")
    for exp_num, measure in measures:
        print(f"  {exp_num}_final_filter -> {measure}")
    print()

    # Pass 1: build the global chitchat-keep veto set (both_keep mode only).
    # If any measure's judge returned chitchat_keep=false for a user_input,
    # that user_input is dropped from ALL measures, since chitchat_keep is a
    # measure-independent property of the conversation. Errors are NOT vetoes.
    # Skipped in xor mode: XOR includes chitchat_keep=false by design.
    vetoed: set[str] = set()
    if apply_veto:
        for exp_num, measure in measures:
            result_path = (
                experiments_dir
                / f"{exp_num}_final_filter"
                / "results"
                / f"{measure}_final.jsonl"
            )
            if not result_path.exists():
                continue
            with open(result_path) as f_in:
                for line in f_in:
                    row = json.loads(line)
                    resp = row.get("model_responses", {}).get(model, {})
                    if "error" in resp:
                        continue
                    if resp.get("chitchat_keep") is False:
                        vetoed.add(row["user_input"])
        print(f"Veto set: {len(vetoed)} user_inputs with chitchat_keep=false in at least one measure\n")
    else:
        print("Chitchat veto: skipped (pool=xor)\n")

    total = 0
    kept_before_dedup = 0
    errors = 0
    dropped_by_veto = 0

    # OrderedDict preserves insertion order (first measure seen)
    seen: "OrderedDict[str, dict]" = OrderedDict()

    for exp_num, measure in measures:
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
                if row["user_input"] in vetoed:
                    dropped_by_veto += 1
                    continue
                resp = row.get("model_responses", {}).get(model, {})
                if "error" in resp:
                    errors += 1
                    continue
                ck = resp.get("chitchat_keep")
                catk = resp.get("category_keep")
                if pool == "both_keep":
                    passes = bool(ck) and bool(catk)
                else:  # "xor"
                    passes = bool(ck) != bool(catk)
                if passes:
                    kept_before_dedup += 1
                    measure_kept += 1
                    ui = row["user_input"]
                    if ui in seen:
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

    rows = list(seen.values())
    with open(output_path, "w") as f_out:
        for row in rows:
            f_out.write(json.dumps(row) + "\n")

    multi = sum(1 for r in rows if len(r["measure"]) > 1)
    label = "Both KEEP" if pool == "both_keep" else "XOR KEEP"
    print(f"\nTotal rows scanned: {total}")
    print(f"Errors: {errors}")
    print(f"Dropped by chitchat veto: {dropped_by_veto}")
    print(f"{label} (before dedup): {kept_before_dedup}")
    print(f"After dedup: {len(rows)} unique conversations")
    print(f"  Single-measure: {len(rows) - multi}")
    print(f"  Multi-measure:  {multi}")
    print(f"Wrote {len(rows)} rows to {output_path}\n")

    return rows


# ---------------------------------------------------------------------------
# Phase 2: Split Single-Turn vs Multi-Turn
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """\
You are classifying a user message from a chatbot conversation.

Your task: determine whether this is a **single-turn** message or a **multi-turn** pasted transcript.

- **single-turn**: The user wrote a standalone message to the chatbot. It may be short or long, \
may contain lists, code, JSON, or other structured content, but it is ONE message from ONE user \
directed at the chatbot.
- **multi-turn**: The user's message contains a pasted conversation transcript — it includes \
dialogue between multiple parties or multiple exchanges with role labels (e.g., "User:", \
"Assistant:", "Human:", "AI:", "Bot:", etc.). The user is sharing or continuing a prior conversation.

Edge cases:
- If the message contains role labels but they are part of instructions (e.g., "When the user \
says X, respond with Y"), classify as **single-turn**.
- If the message quotes a short snippet of dialogue as an example within a larger standalone \
request, classify as **single-turn**.
- If the message is primarily a multi-exchange transcript (even if it ends with a new request), \
classify as **multi-turn**.

Respond with ONLY a JSON object:
{"classification": "single-turn" or "multi-turn", "reasoning": "brief explanation"}"""


def classify_with_llm(
    lm: dspy.LM,
    user_input: str,
    max_retries: int = 5,
) -> tuple[str, str]:
    """Use Opus (via DSPy + disk cache) to classify a user_input.

    Returns (classification, reasoning). DSPy caches successful responses to
    cache/dspy/, so re-running with the same user_input is free.
    """
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_input[:8000]},
    ]
    raw = ""
    for attempt in range(max_retries):
        try:
            response = lm(messages=messages)
            if not response or not response[0]:
                raise ValueError("empty response")
            raw = response[0]
            if isinstance(raw, dict):
                raw = raw.get("content", "") or raw.get("text", "") or str(raw)
            raw = raw.strip()
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait}s...", flush=True)
            time.sleep(wait)
    else:
        return "multi-turn", "All retries failed, defaulting to multi-turn"

    try:
        text = re.sub(r"^```(?:json)?\s*", "", raw)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        return parsed["classification"], parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        if "single-turn" in raw.lower():
            return "single-turn", f"Fallback parse: {raw[:200]}"
        elif "multi-turn" in raw.lower():
            return "multi-turn", f"Fallback parse: {raw[:200]}"
        return "multi-turn", f"Could not parse, defaulting to multi-turn: {raw[:200]}"


def split_turns(rows: list[dict], combined_path: Path, key_path: Path) -> None:
    """Classify each row as single/multi-turn and write split files + report."""
    output_dir = combined_path.parent
    stem = combined_path.stem

    api_key = Path(key_path).read_text().strip()
    lm = dspy.LM(
        model="openrouter/anthropic/claude-opus-4-6",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=256,
        temperature=0,
        cache=True,
        timeout=120,
    )

    # Resume from existing report if present
    report_path = output_dir / f"split_report_{stem}.json"
    cached_details: dict[str, dict] = {}
    if report_path.exists():
        with open(report_path) as f:
            prev_report = json.load(f)
        for d in prev_report.get("details", []):
            cached_details[d["preview"]] = d
        print(f"Resuming: loaded {len(cached_details)} cached classifications from {report_path}", flush=True)

    single_turn: list[dict] = []
    multi_turn: list[dict] = []
    details: list[dict] = []

    for i, row in enumerate(rows):
        user_input = row["user_input"]
        preview = user_input[:200]

        if preview in cached_details:
            classification = cached_details[preview]["classification"]
            reasoning = cached_details[preview]["reasoning"]
            tag = "cached"
        else:
            classification, reasoning = classify_with_llm(lm, user_input)
            tag = "new"

        details.append({
            "index": i,
            "classification": classification,
            "reasoning": reasoning,
            "preview": preview,
        })

        if classification == "single-turn":
            single_turn.append(row)
        else:
            multi_turn.append(row)

        print(f"  [{i+1}/{len(rows)}] {classification} ({tag}) — {reasoning[:80]}", flush=True)

        # Incremental checkpoint so we can resume on crash
        if tag == "new" and (i + 1) % 10 == 0:
            partial_report = {
                "total": len(rows),
                "single_turn_count": len(single_turn),
                "multi_turn_count": len(multi_turn),
                "details": details,
            }
            with open(report_path, "w") as f:
                json.dump(partial_report, f, indent=2)
            print(f"    (checkpoint saved at {i+1}/{len(rows)})", flush=True)

    single_path = output_dir / f"single_turn_{stem}.jsonl"
    multi_path = output_dir / f"multi_turn_{stem}.jsonl"

    with open(single_path, "w") as f:
        for row in single_turn:
            slim = {k: row[k] for k in ("user_input", "measure") if k in row}
            f.write(json.dumps(slim) + "\n")
    with open(multi_path, "w") as f:
        for row in multi_turn:
            f.write(json.dumps(row) + "\n")

    report = {
        "total": len(rows),
        "single_turn_count": len(single_turn),
        "multi_turn_count": len(multi_turn),
        "details": details,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSplit results:", flush=True)
    print(f"  Single-turn: {len(single_turn)} → {single_path}", flush=True)
    print(f"  Multi-turn:  {len(multi_turn)} → {multi_path}", flush=True)
    print(f"  Report:      {report_path}", flush=True)


# ---------------------------------------------------------------------------
# Phase 3: Data Cleaning (semantic dedup + metadata tagging)
# ---------------------------------------------------------------------------

def clean_data(combined_path: Path, threshold: float) -> None:
    """Phase 3: semantic dedup (single-turn only) + add synthetic/language tags.

    Step 1: embed every `user_input` in `single_turn_{stem}.jsonl` with
    sentence-transformers/all-MiniLM-L6-v2, compute pairwise cosine similarity,
    and drop any row whose similarity to an earlier kept row is >= `threshold`.
    Rewrites the single-turn file in place and writes a `dedup_report_{stem}.json`.

    Step 2: tag every row in the combined, single-turn, and multi-turn files
    with `synthetic=False` and `language="English"`. Idempotent — re-running on
    already-tagged files is a no-op.

    If `--no_split` was used upstream (so no single-turn / multi-turn files
    exist), Step 1 is skipped and Step 2 only tags files that exist.
    """
    output_dir = combined_path.parent
    stem = combined_path.stem
    single_path = output_dir / f"single_turn_{stem}.jsonl"
    multi_path = output_dir / f"multi_turn_{stem}.jsonl"

    # --- Step 1: semantic dedup on single-turn only ---
    if single_path.exists():
        # Lazy imports so the module still imports cleanly before `uv sync`.
        import numpy as np
        from sentence_transformers import SentenceTransformer

        with open(single_path) as f:
            rows = [json.loads(line) for line in f]
        print(f"\nPhase 3.1 — semantic dedup on {len(rows)} single-turn rows "
              f"(threshold={threshold})", flush=True)

        if len(rows) == 0:
            print("  (empty file, skipping)", flush=True)
            dropped: list[tuple[int, int, float]] = []
            keep: list[int] = []
        else:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embs = model.encode(
                [r["user_input"] for r in rows],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            sim = embs @ embs.T  # cosine, since normalized

            keep = []
            dropped = []  # (dropped_idx, kept_idx, similarity)
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

            print(f"  Kept: {len(keep)}  Dropped: {len(dropped)}", flush=True)
            deduped_rows = [rows[i] for i in keep]
            with open(single_path, "w") as f:
                for r in deduped_rows:
                    f.write(json.dumps(r) + "\n")

        report = {
            "threshold": threshold,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "original_count": len(rows),
            "kept_count": len(keep),
            "dropped_count": len(dropped),
            "dropped": sorted(
                [
                    {
                        "dropped_preview": rows[d]["user_input"][:200],
                        "kept_preview": rows[k]["user_input"][:200],
                        "similarity": round(s, 4),
                    }
                    for d, k, s in dropped
                ],
                key=lambda x: x["similarity"],
                reverse=True,
            ),
        }
        dedup_report_path = output_dir / f"dedup_report_{stem}.json"
        with open(dedup_report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Dedup report: {dedup_report_path}", flush=True)
    else:
        print(f"\nPhase 3.1 — skipped (no {single_path.name}; did you run with --no_split?)",
              flush=True)

    # --- Step 2: add synthetic/language metadata to all 3 files ---
    def tag_file(path: Path) -> None:
        if not path.exists():
            return
        with open(path) as f:
            rs = [json.loads(line) for line in f]
        for r in rs:
            r["synthetic"] = False
            r["language"] = "English"
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(f"  Tagged {len(rs)} rows in {path.name}", flush=True)

    print("Phase 3.2 — tagging rows with synthetic=false, language=English", flush=True)
    tag_file(combined_path)
    tag_file(single_path)
    tag_file(multi_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5 — Data preprocessing: collect both-KEEP rows from "
                    "final_filter experiments, deduplicate by user_input, then "
                    "split into single-turn / multi-turn subsets via Opus 4.6.",
    )
    parser.add_argument(
        "--experiments_dir", type=str, default="experiments",
        help="Root experiments directory (default: experiments)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write the combined deduplicated JSONL "
             "(split files are derived from this stem).",
    )
    parser.add_argument(
        "--model", type=str, default="claude_opus_4_6",
        help="Model key in model_responses for Phase 1 (default: claude_opus_4_6)",
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated list of experiment_num:measure pairs to use "
             "(e.g. '19:1B_intentional_human_speech,20:1C_identity_transparency'). "
             "If not provided, auto-discovers all final_filter experiments.",
    )
    parser.add_argument(
        "--key", type=str, default=".openrouter_key",
        help="Path to OpenRouter API key file (used for Phase 2 splitting)",
    )
    parser.add_argument(
        "--no_split", action="store_true",
        help="Skip Phase 2 (single/multi-turn splitting); only collect & deduplicate.",
    )
    parser.add_argument(
        "--pool", type=str, default="both_keep", choices=["both_keep", "xor"],
        help="Which Stage 4 rows to collect. 'both_keep' = chitchat_keep AND "
             "category_keep (positives, default; chitchat veto applied). "
             "'xor' = exactly one of the two (negatives, used as synthetic-"
             "generation few-shot negatives; chitchat veto skipped since XOR "
             "includes chitchat_keep=false by definition).",
    )
    parser.add_argument(
        "--dedupe_threshold", type=float, default=0.85,
        help="Cosine-similarity threshold for Phase 3 semantic dedup on the "
             "single-turn file (default: 0.85). Pairs with similarity >= threshold "
             "are considered duplicates and the later occurrence is dropped.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Collect & Deduplicate")
    print("=" * 60)
    rows = collect_and_dedupe(
        experiments_dir=Path(args.experiments_dir),
        output_path=output_path,
        model=args.model,
        experiments_arg=args.experiments,
        pool=args.pool,
    )

    if args.no_split:
        print("Skipping Phase 2 (--no_split set).")
    else:
        print("=" * 60)
        print("Phase 2: Split Single-Turn vs Multi-Turn")
        print("=" * 60)
        split_turns(rows=rows, combined_path=output_path, key_path=Path(args.key))

    print("=" * 60)
    print("Phase 3: Data Cleaning (semantic dedup + metadata tagging)")
    print("=" * 60)
    clean_data(combined_path=output_path, threshold=args.dedupe_threshold)


if __name__ == "__main__":
    main()
