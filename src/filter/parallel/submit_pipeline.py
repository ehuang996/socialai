"""Submit Stage 2 → 3 → 4 pipeline runs for one or more measures with SLURM
dependency chains, then write the resulting job IDs and output paths to a
state file for later polling and reporting.

Usage (from project root):
    uv run python src/filter/parallel/submit_pipeline.py \\
        --measures 2A_fabricated_personal_information,2B_emotion_expression,2C_flattery_tone,2D_human_relationship_encouragement \\
        --stage1_output experiments/00_coarse_filter/results/1B_intentional_human_speech_coarse.jsonl \\
        --key .openrouter_key \\
        --shards_s2 8 --shards_s34 4

Calls pipeline.sh under the hood for each (measure, stage) and parses the JIDs
+ output paths from its stdout. Stage 3's array job depends on Stage 2's concat;
Stage 4's array job depends on Stage 3's concat. SLURM resolves the chain
once the dependencies clear, so all 12 chains can be queued at once.

State written to src/filter/parallel/state.json — keyed by measure, with
per-stage entries {array_jid, concat_jid, exp_dir, output_file}.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_MEASURES = [
    "2A_fabricated_personal_information",
    "2B_emotion_expression",
    "2C_flattery_tone",
    "2D_human_relationship_encouragement",
]
STATE_PATH = Path("src/filter/parallel/state.json")


def call_pipeline(
    measure: str,
    stage: str,
    input_path: str,
    shards: int,
    key: str | None = None,
    depends_on: str | None = None,
) -> dict:
    """Invoke pipeline.sh and parse its stdout for JIDs + paths."""
    cmd = [
        "bash", "pipeline.sh",
        "--measure", measure,
        "--stage", stage,
        "--shards", str(shards),
        "--input", input_path,
    ]
    if stage in ("high_quality_filter", "final_filter"):
        if not key:
            raise ValueError(f"key required for {stage}")
        cmd += ["--key", key]
    if depends_on:
        cmd += ["--depends-on", depends_on]

    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = result.stdout
    err = result.stderr
    if result.returncode != 0:
        print(f"pipeline.sh failed (rc={result.returncode}):\n--- stdout ---\n{out}\n--- stderr ---\n{err}", file=sys.stderr)
        raise RuntimeError(f"pipeline.sh failed for {measure}/{stage}")
    print(out, flush=True)

    array_jid = re.search(r"Array job submitted:\s*(\d+)", out)
    concat_jid = re.search(r"Concat job submitted:\s*(\d+)", out)
    exp_dir = re.search(r"Experiment:\s*(\S+)", out)
    output_file = re.search(r"Final output:\s*(\S+)", out)

    if not (array_jid and concat_jid and exp_dir and output_file):
        raise RuntimeError(
            f"Could not parse pipeline.sh output for {measure}/{stage}.\n"
            f"Stdout was:\n{out}"
        )

    return {
        "array_jid": array_jid.group(1),
        "concat_jid": concat_jid.group(1),
        "exp_dir": exp_dir.group(1),
        "output_file": output_file.group(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--measures",
        default=",".join(DEFAULT_MEASURES),
        help="Comma-separated measure names to run.",
    )
    ap.add_argument(
        "--stage1_output",
        required=True,
        help="Stage 1 (coarse filter) output JSONL — input for Stage 2.",
    )
    ap.add_argument("--key", default=".openrouter_key", help="OpenRouter API key file.")
    ap.add_argument("--shards_s2", type=int, default=8, help="Stage 2 shards.")
    ap.add_argument("--shards_s34", type=int, default=4, help="Stage 3/4 shards.")
    args = ap.parse_args()

    measures = [m.strip() for m in args.measures.split(",") if m.strip()]
    state: dict = {"measures": measures, "by_measure": {}}

    for m in measures:
        print(f"\n========= {m} =========")
        s2 = call_pipeline(m, "low_quality_filter", args.stage1_output, args.shards_s2)
        s3 = call_pipeline(
            m, "high_quality_filter", s2["output_file"], args.shards_s34,
            key=args.key, depends_on=s2["concat_jid"],
        )
        s4 = call_pipeline(
            m, "final_filter", s3["output_file"], args.shards_s34,
            key=args.key, depends_on=s3["concat_jid"],
        )
        state["by_measure"][m] = {"stage2": s2, "stage3": s3, "stage4": s4}

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nWrote state to {STATE_PATH}")

    # Compact summary
    print("\nJob chains (array_jid → concat_jid):")
    for m, st in state["by_measure"].items():
        print(f"  {m}")
        for stage_key in ("stage2", "stage3", "stage4"):
            s = st[stage_key]
            print(f"    {stage_key}: {s['array_jid']} → {s['concat_jid']}  | {s['output_file']}")


if __name__ == "__main__":
    main()
