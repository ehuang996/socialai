"""Stage 11.2b — build the manual-review page for the multi-turn dataset.

Writes one self-contained HTML file (docs/multi_turn_review/index.html, for
GitHub Pages) that embeds every dataset conversation with its full WildChat
transcript, the triggering assistant reply highlighted, and the Stage 2/3/4
judge reasoning behind each label. The reviewer accepts or rejects each
(conversation, measure) label in the browser; decisions autosave to
localStorage and export as JSON for `apply_review.py`.

Usage:
  python -m src.multi_turn_experiment.dataset.build_review_page
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.multi_turn_experiment._common import iter_jsonl, load_pool

TEMPLATE = Path(__file__).with_name("review_template.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/multi_turn/multi_turn_dataset.jsonl")
    parser.add_argument("--pool", default="data/multi_turn/pool.jsonl")
    parser.add_argument("--filter_dir", default="data/multi_turn/filter")
    parser.add_argument("--output", default="docs/multi_turn_review/index.html")
    parser.add_argument("--artifact", action="store_true",
                        help="emit only the page content (no doctype/html/head/body) for hosts that wrap it")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.dataset))
    pool = load_pool(args.pool, {r["conversation_hash"] for r in rows})

    # Judge reasoning per (hash, turn, measure) from the three stage files.
    reasoning = defaultdict(dict)
    for measure in {m for r in rows for m in r["measure"]}:
        for stage, suffix, col in [("stage2", "low_quality", None), ("stage3", "high_quality", "gpt_4o_mini"),
                                   ("stage4", "final", "claude_opus_4_6")]:
            for v in iter_jsonl(Path(args.filter_dir) / f"{measure}_{suffix}.jsonl"):
                verdict = v if col is None else v["model_responses"].get(col, {})
                reasoning[(v["conversation_hash"], v["turn_index"], measure)][stage] = {
                    k: verdict.get(k) for k in ("reasoning", "keep", "chitchat_reasoning", "chitchat_keep",
                                                "category_reasoning", "category_keep") if k in verdict}

    conversations = []
    for r in rows:
        p = pool[r["conversation_hash"]]
        conversations.append({
            "hash": r["conversation_hash"],
            "model": p.get("model", ""), "timestamp": p.get("timestamp", ""), "turns": p["turn"],
            "conversation": p["conversation"],          # the full WildChat transcript, all turns
            "measures": r["measure"],
            "trigger_turns": r["trigger_turns"],
            "labels": [{"measure": m, "turn": n, **reasoning[(r["conversation_hash"], n, m)]}
                       for m, n in r["trigger_turns"].items()],
        })

    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(conversations, ensure_ascii=False))
    if args.artifact:
        head = html.split("<head>", 1)[1].split("</head>", 1)[0].replace('<meta charset="utf-8">', "")
        head = "\n".join(l for l in head.splitlines() if "viewport" not in l)
        body = html.split("<body>", 1)[1].split("</body>", 1)[0]
        html = head + body
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(html)
    print(f"{len(conversations)} conversations, {sum(len(c['labels']) for c in conversations)} labels -> {args.output} "
          f"({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
