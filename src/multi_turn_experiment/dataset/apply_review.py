"""Stage 11.2c — apply the manual review decisions to the multi-turn dataset.

Reads the JSON exported from the review page ({"decisions": [{conversation_hash,
measure, decision, note}, ...]}), keeps only labels marked "accept", drops
conversations with no accepted label, and re-derives trigger_turn / context /
original_assistant_response from the accepted labels.

Usage:
  python -m src.multi_turn_experiment.dataset.apply_review --decisions review_decisions.json
"""

import argparse
import json
from pathlib import Path

from src.multi_turn_experiment._common import assemble_row, iter_jsonl, load_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--dataset", default="data/multi_turn/multi_turn_dataset.jsonl")
    parser.add_argument("--pool", default="data/multi_turn/pool.jsonl")
    parser.add_argument("--output", default="data/multi_turn/multi_turn_dataset_reviewed.jsonl")
    args = parser.parse_args()

    decisions = json.load(open(args.decisions))["decisions"]
    accepted = {(d["conversation_hash"], d["measure"]) for d in decisions if d["decision"] == "accept"}
    notes = {d["conversation_hash"]: d.get("note", "") for d in decisions if d.get("note")}
    rows = list(iter_jsonl(args.dataset))
    pool = load_pool(args.pool, {r["conversation_hash"] for r in rows})

    kept, dropped_labels = [], 0
    for r in rows:
        trigger_turns = {m: n for m, n in r["trigger_turns"].items() if (r["conversation_hash"], m) in accepted}
        dropped_labels += len(r["trigger_turns"]) - len(trigger_turns)
        row = assemble_row(pool[r["conversation_hash"]], trigger_turns)
        if row is not None:
            row["review_note"] = notes.get(r["conversation_hash"], "")
            kept.append(row)
    with open(args.output, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    unreviewed = sum(1 for r in rows for m in r["measure"]
                     if not any(d["conversation_hash"] == r["conversation_hash"] and d["measure"] == m for d in decisions))
    print(f"{len(rows)} conversations in -> {len(kept)} kept ({sum(len(r['measure']) for r in kept)} labels); "
          f"{dropped_labels} labels rejected or unreviewed ({unreviewed} unreviewed) -> {args.output}")


if __name__ == "__main__":
    main()
