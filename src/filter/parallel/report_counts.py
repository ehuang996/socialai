"""Report keep counts at every stage for the measures recorded in state.json.

Reads `src/filter/parallel/state.json` (written by submit_pipeline.py), opens
each stage's output JSONL, and prints a table of:
    Stage 2 keep=true / total / errors
    Stage 3 chitchat / category / both / total / errors
    Stage 4 chitchat / category / both / total / errors

Also reports SLURM job state for each (array, concat) JID so partial runs are
easy to spot.

Usage (from project root):
    uv run python src/filter/parallel/report_counts.py
"""

import json
import subprocess
from pathlib import Path

STATE_PATH = Path("src/filter/parallel/state.json")


def squeue_state(jids: list[str]) -> dict[str, str]:
    """Query SLURM for the state of each JID. Empty string = not in queue (likely done)."""
    if not jids:
        return {}
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-o", "%i %T", "-j", ",".join(jids)],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {jid: "?" for jid in jids}
    states: dict[str, str] = {jid: "" for jid in jids}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            jid_full, st = parts[0], parts[1]
            jid = jid_full.split("_")[0]  # array tasks may be JID_TASKID
            states[jid] = st
    return states


def count_s2(path: Path) -> dict:
    if not path.exists():
        return {"_missing": True}
    total = keep = err = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            err += 1
            continue
        total += 1
        k = r.get("keep")
        if k is True:
            keep += 1
        elif k is None or "error" in r:
            err += 1
    return {"total": total, "keep": keep, "errors": err}


def count_dual(path: Path, model_key: str) -> dict:
    if not path.exists():
        return {"_missing": True}
    total = chit = cat = both = err = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            err += 1
            continue
        total += 1
        mr = r.get("model_responses", {}).get(model_key, {})
        cc, ck = mr.get("chitchat_keep"), mr.get("category_keep")
        if cc is None or ck is None or "error" in mr:
            err += 1
            continue
        if cc:
            chit += 1
        if ck:
            cat += 1
        if cc and ck:
            both += 1
    return {"total": total, "chitchat_keep": chit, "category_keep": cat,
            "both": both, "errors": err}


def fmt_state(jid: str, st: dict[str, str]) -> str:
    s = st.get(jid, "")
    return s if s else "DONE"


def main():
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found. Run submit_pipeline.py first.")
        return

    state = json.loads(STATE_PATH.read_text())
    measures = state["measures"]
    by_m = state["by_measure"]

    # Collect all JIDs across all measures
    all_jids: list[str] = []
    for m in measures:
        for stage_key in ("stage2", "stage3", "stage4"):
            s = by_m[m][stage_key]
            all_jids.extend([s["array_jid"], s["concat_jid"]])
    sq = squeue_state(all_jids)

    print(f"{'measure':<42} {'stage':<8} {'array':>9} {'concat':>9} {'output rows / keeps':<60}")
    print("-" * 130)
    for m in measures:
        for stage_key, label in [("stage2", "S2"), ("stage3", "S3"), ("stage4", "S4")]:
            s = by_m[m][stage_key]
            arr_st = fmt_state(s["array_jid"], sq)
            cat_st = fmt_state(s["concat_jid"], sq)
            path = Path(s["output_file"])
            if stage_key == "stage2":
                cts = count_s2(path)
                if "_missing" in cts:
                    summary = "(no output yet)"
                else:
                    summary = f"total={cts['total']:,} keep={cts['keep']:,} err={cts['errors']}"
            else:
                model_key = "gpt_4o_mini" if stage_key == "stage3" else "claude_opus_4_6"
                cts = count_dual(path, model_key)
                if "_missing" in cts:
                    summary = "(no output yet)"
                else:
                    summary = (f"total={cts['total']:,} "
                               f"chit={cts['chitchat_keep']:,} cat={cts['category_keep']:,} "
                               f"both={cts['both']:,} err={cts['errors']}")
            print(f"{m:<42} {label:<8} {arr_st:>9} {cat_st:>9} {summary}")
        print()


if __name__ == "__main__":
    main()
