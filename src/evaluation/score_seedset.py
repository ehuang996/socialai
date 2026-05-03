"""One-shot evaluator for `data/new_seedset.jsonl`.

Phase A — Generate model responses for each user_input on:
  - openai/gpt-4o
  - anthropic/claude-sonnet-4
  - google/gemini-2.0-flash-001
DSPy-cached, so re-runs are free.

Phase B — Judge each (input, response) under ALL 9 active measures using
Claude Opus 4.6 (regardless of the row's labeled measure). 322 × 3 × 9 = 8,694
judgments. Same DSPy disk cache.

Parallelism: ThreadPoolExecutor with N_WORKERS=4 across both phases.

Output: data/seedset_scored.jsonl — one row per (user_input, model, measure)
triple with the judge verdict.

Usage:
    cd /project2/jessetho_1732/wangzhu/socialai
    .venv/bin/python src/evaluation/score_seedset.py --key .openrouter_key
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# DSPy cache configured before import
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")
import dspy

from src.evaluation._eval_common import (
    ALL_MEASURES as ACTIVE_MEASURES,
    extract_text,
    load_judge_prompts as _load_judge_prompts,
    normalize_keep,
    parse_judge_response,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
N_WORKERS = 4

# Models to test (Phase A) — short_name → openrouter model id
RESPONSE_MODELS = {
    "gpt_4o":          "openai/gpt-4o",
    "claude_sonnet_4": "anthropic/claude-sonnet-4",
    "gemini_2_flash":  "google/gemini-2.0-flash-001",
}
JUDGE_MODEL = "anthropic/claude-opus-4.6"


def load_judge_prompts() -> dict[str, str]:
    return _load_judge_prompts(ACTIVE_MEASURES)


def call_model(lm: dspy.LM, user_input: str) -> str:
    response = lm(messages=[{"role": "user", "content": user_input}])
    return extract_text(response)


def call_judge(lm: dspy.LM, system_prompt: str, conversation: str, attempt: int = 0) -> dict:
    config = {"rollout_id": attempt} if attempt > 0 else {}
    response = lm(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation},
    ], **config)
    raw = extract_text(response)
    try:
        parsed = normalize_keep(parse_judge_response(raw))
    except Exception as e:
        parsed = {"error": "parse_failed", "raw": raw[:500], "exc": str(e)}
    return {"raw_judge_response": raw, "judge_output": parsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/new_seedset.jsonl")
    ap.add_argument("--output", default="data/seedset_scored.jsonl")
    ap.add_argument("--key", default=".openrouter_key")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    api_key = Path(args.key).read_text().strip()
    rows = [json.loads(l) for l in open(args.input)]
    print(f"Loaded {len(rows)} rows from {args.input}")

    # Build DSPy LMs (one per model id)
    lms = {
        short: dspy.LM(model=f"openrouter/{model_id}",
                       api_key=api_key, api_base=OPENROUTER_BASE_URL,
                       max_tokens=2048, temperature=0.7,
                       cache=True, timeout=180)
        for short, model_id in RESPONSE_MODELS.items()
    }
    judge_lm = dspy.LM(model=f"openrouter/{JUDGE_MODEL}",
                       api_key=api_key, api_base=OPENROUTER_BASE_URL,
                       max_tokens=1024, temperature=0,
                       cache=True, timeout=120)
    judge_prompts = load_judge_prompts()
    print(f"Loaded {len(judge_prompts)} judge prompts: {list(judge_prompts)}")

    # Resume support: read any existing rows from output
    completed = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                completed.add((r["user_input"], r["model_name"], r["measure"]))
            except Exception:
                continue
        print(f"Resuming: {len(completed)} (input, model, measure) triples already in {out_path}")

    # Phase A — generate responses per (row, model). Cache responses in memory.
    print("\n" + "="*70)
    print("Phase A: generating responses (3 models × {} rows = {} calls)".format(
        len(rows), len(rows) * 3))
    print("="*70)
    response_cache: dict[tuple[str, str], str] = {}

    def gen_one(row_idx: int, model_short: str):
        ui = rows[row_idx]["user_input"]
        try:
            resp = call_model(lms[model_short], ui)
        except Exception as e:
            resp = f"__ERROR__: {type(e).__name__}: {e}"
        response_cache[(ui, model_short)] = resp
        return row_idx, model_short

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i in range(len(rows)):
            for short in RESPONSE_MODELS:
                futures.append(ex.submit(gen_one, i, short))
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(f"  Phase A: {done}/{len(futures)}")
    print(f"Phase A done. Cached {len(response_cache)} responses.")

    # Phase B — judge each (row, model) response under all 9 measures
    print("\n" + "="*70)
    n_total = len(rows) * len(RESPONSE_MODELS) * len(ACTIVE_MEASURES)
    print("Phase B: judging on Opus 4.6 ({} rows × 3 models × 9 measures = {} judgments)".format(
        len(rows), n_total))
    print("="*70)

    def judge_one(row_idx: int, model_short: str, measure: str):
        ui = rows[row_idx]["user_input"]
        labeled_measure = rows[row_idx].get("measure", [])
        resp = response_cache.get((ui, model_short), "")
        if resp.startswith("__ERROR__"):
            return {
                "user_input": ui, "labeled_measure": labeled_measure,
                "model_name": model_short, "model_response": resp,
                "measure": measure,
                "raw_judge_response": "", "judge_output": {"error": "model_error"},
            }
        conversation = f"USER: {ui}\nASSISTANT: {resp}"
        try:
            result = call_judge(judge_lm, judge_prompts[measure], conversation)
        except Exception as e:
            return {
                "user_input": ui, "labeled_measure": labeled_measure,
                "model_name": model_short, "model_response": resp,
                "measure": measure,
                "raw_judge_response": "", "judge_output": {"error": f"{type(e).__name__}: {e}"},
            }
        return {
            "user_input": ui, "labeled_measure": labeled_measure,
            "model_name": model_short, "model_response": resp,
            "measure": measure,
            "raw_judge_response": result["raw_judge_response"],
            "judge_output": result["judge_output"],
        }

    todo = []
    for i in range(len(rows)):
        for short in RESPONSE_MODELS:
            for m in ACTIVE_MEASURES:
                if (rows[i]["user_input"], short, m) in completed:
                    continue
                todo.append((i, short, m))
    print(f"To do: {len(todo)} (skipping {n_total - len(todo)} already-cached triples)")

    with open(out_path, "a") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(judge_one, *t) for t in todo]
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                f_out.write(json.dumps(r) + "\n")
                f_out.flush()
                done += 1
                if done % 200 == 0 or done == len(futures):
                    print(f"  Phase B: {done}/{len(futures)}")

    print(f"\nWrote scored output → {out_path}")

    # Summary report
    print("\n" + "="*70)
    print("SUMMARY — keep counts per (model × measure)")
    print("="*70)
    from collections import Counter
    counts = Counter()  # (model, measure, keep) → count
    err = Counter()
    total = 0
    for line in open(out_path):
        r = json.loads(line)
        total += 1
        jo = r.get("judge_output", {})
        if "error" in jo or "keep" not in jo:
            err[(r["model_name"], r["measure"])] += 1
            continue
        counts[(r["model_name"], r["measure"], jo["keep"])] += 1
    print(f"Total judged: {total}\n")
    print(f"{'measure':<42} {'gpt_4o':>14} {'claude_s4':>14} {'gemini_2f':>14}")
    print("-" * 90)
    for m in ACTIVE_MEASURES:
        cells = []
        for model in RESPONSE_MODELS:
            kt = counts.get((model, m, True), 0)
            kf = counts.get((model, m, False), 0)
            e = err.get((model, m), 0)
            cells.append(f"{kt:>3}/{kt+kf+e:<4}")
        print(f"{m:<42} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
    print("\n(format: keep=true / total)")


if __name__ == "__main__":
    main()
