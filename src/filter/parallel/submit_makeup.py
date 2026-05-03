"""Submit make-up arrays for failed/preempted Stage 2 shards, then chain Stage 3+4
with proper dependencies. Resumes from each shard's existing partial file via
Judge.load_completed_ids().

Each measure's spec:
  - missing_shards: list[int] — shards that need a re-run (FAILED, PREEMPTED, or
    silent-quota-error COMPLETED)
  - original_array_jid: str — the original array; concat waits for BOTH that
    array AND the make-up (afterany so PREEMPTED tasks don't block)
  - exp_dirs: dict with stage2/3/4 paths

After this runs, state.json is patched with the new make-up + chain JIDs.

Usage:
    .venv/bin/python src/filter/parallel/submit_makeup.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/project2/jessetho_1732/wangzhu/socialai")
KEY = ".openrouter_key"
STAGE1_INPUT = "experiments/00_coarse_filter/results/1B_intentional_human_speech_coarse.jsonl"
STATE_PATH = PROJECT / "src/filter/parallel/state.json"

# After fixing the shard-then-filter bug in Judge.run() (the order was reversed,
# causing each shard to reprocess rows from other shards' slices on resume),
# every shard's partial file is stale wrt its own slice — re-run all 8 with
# resumption. Each shard now only needs the ~5–7K rows from its true slice
# that aren't already in its part file.
MEASURES = {
    "2A_fabricated_personal_information": {
        "missing": list(range(8)),
        "original_array": "3309903",
        "exp_s2": "experiments/25_low_quality_filter",
    },
    "2B_emotion_expression": {
        "missing": list(range(8)),
        "original_array": "3309836",
        "exp_s2": "experiments/16_low_quality_filter",
    },
    "2C_flattery_tone": {
        "missing": list(range(8)),
        "original_array": "3309842",
        "exp_s2": "experiments/19_low_quality_filter",
    },
    "2D_human_relationship_encouragement": {
        "missing": list(range(8)),
        "original_array": "3309848",
        "exp_s2": "experiments/22_low_quality_filter",
    },
}


def sbatch(cmd: list[str]) -> str:
    """Run sbatch and return parsed JID."""
    print("$ " + " ".join(cmd), flush=True)
    out = subprocess.check_output(cmd, text=True).strip()
    print(out, flush=True)
    return out  # --parsable returns just the JID


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def submit_one_measure(measure: str, cfg: dict) -> dict:
    """Submit make-up array + new concat + new Stage 3+4 chain for one measure."""
    missing = cfg["missing"]
    original = cfg["original_array"]
    exp_s2 = cfg["exp_s2"]
    array_spec = ",".join(str(s) for s in missing)
    output_base = f"data/{measure}_scores"
    output_file = f"{exp_s2}/results/{measure}_scores.jsonl"

    # 1. Make-up Stage 2 array — same OUTPUT_BASE so per-shard files resume
    env = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=low_quality_filter",
        f"EXPERIMENT_DIR={exp_s2}",
        f"INPUT_PATH={STAGE1_INPUT}",
        f"OUTPUT_BASE={output_base}",
        "EXTRA_ARGS=--concurrency_limit 64",
    ])
    makeup = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_makeup_s2",
        "--array", array_spec,
        "--output", f"{exp_s2}/logs/job_%A_%a.out",
        "--error", f"{exp_s2}/logs/job_%A_%a.err",
        "--export", env,
        "slurm/run_vllm_stage.sbatch",
    ])

    # 2. Concat — depends on BOTH make-up + original (afterany since some
    #    original tasks may be PREEMPTED/FAILED but we still want to concat the
    #    8 part files once writes are quiesced). Uses dedup_concat.py to drop
    #    duplicate conversation_hashes left by the prior shard-after-filter bug.
    concat_cmd = (
        "module purge && module load gcc/13.3.0 && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "uv run --no-sync python src/filter/parallel/dedup_concat.py "
        f"--output_base {output_base} --output_file {output_file}"
    )
    concat = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_concat",
        "--partition", "nlp",
        "--account", "robinjia_875",
        "--chdir", str(PROJECT),
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--dependency", f"afterany:{makeup}:{original}",
        "--output", f"{exp_s2}/logs/concat_%j.out",
        "--wrap", concat_cmd,
    ])

    # 3. Stage 3 array — afterok on concat
    exp_s3_num = int(exp_s2.split("/")[-1].split("_")[0]) + 1
    exp_s3 = f"experiments/{exp_s3_num:02d}_high_quality_filter"
    Path(f"{exp_s3}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"{exp_s3}/results").mkdir(parents=True, exist_ok=True)
    s3_output_base = f"data/{measure}_high_quality"
    s3_output_file = f"{exp_s3}/results/{measure}_high_quality.jsonl"
    env_s3 = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=high_quality_filter",
        f"EXPERIMENT_DIR={exp_s3}",
        f"INPUT_PATH={output_file}",
        f"OUTPUT_BASE={s3_output_base}",
        f"KEY={KEY}",
        "EXTRA_ARGS=",
    ])
    s3_arr = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_s3",
        "--array", "0-3",
        "--output", f"{exp_s3}/logs/job_%A_%a.out",
        "--error", f"{exp_s3}/logs/job_%A_%a.err",
        "--export", env_s3,
        "--dependency", f"afterok:{concat}",
        "slurm/run_api_stage.sbatch",
    ])
    # IMPORTANT: do NOT use `echo … -> FILE` here — bash interprets the `>` as a
    # redirection and *overwrites* FILE with the echo line, corrupting the
    # output. Use `wc -l FILE` (which writes to stdout/the .out log) instead.
    s3_concat_cmd = (
        f"cat {s3_output_base}_part_*.jsonl > {s3_output_file} && "
        f"rm {s3_output_base}_part_*.jsonl && "
        f"wc -l {s3_output_file}"
    )
    s3_concat = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_s3_concat",
        "--partition", "nlp",
        "--account", "robinjia_875",
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--dependency", f"afterok:{s3_arr}",
        "--output", f"{exp_s3}/logs/concat_%j.out",
        "--wrap", s3_concat_cmd,
    ])

    # 4. Stage 4 array — afterok on Stage 3 concat
    exp_s4_num = exp_s3_num + 1
    exp_s4 = f"experiments/{exp_s4_num:02d}_final_filter"
    Path(f"{exp_s4}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"{exp_s4}/results").mkdir(parents=True, exist_ok=True)
    s4_output_base = f"data/{measure}_final"
    s4_output_file = f"{exp_s4}/results/{measure}_final.jsonl"
    env_s4 = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=final_filter",
        f"EXPERIMENT_DIR={exp_s4}",
        f"INPUT_PATH={s3_output_file}",
        f"OUTPUT_BASE={s4_output_base}",
        f"KEY={KEY}",
        "EXTRA_ARGS=",
    ])
    s4_arr = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_s4",
        "--array", "0-3",
        "--output", f"{exp_s4}/logs/job_%A_%a.out",
        "--error", f"{exp_s4}/logs/job_%A_%a.err",
        "--export", env_s4,
        "--dependency", f"afterok:{s3_concat}",
        "slurm/run_api_stage.sbatch",
    ])
    # See note on s3_concat_cmd: avoid `echo … -> FILE` (bash redirects, clobbers FILE).
    s4_concat_cmd = (
        f"cat {s4_output_base}_part_*.jsonl > {s4_output_file} && "
        f"rm {s4_output_base}_part_*.jsonl && "
        f"wc -l {s4_output_file}"
    )
    s4_concat = sbatch([
        "sbatch", "--parsable",
        "--job-name", f"{measure}_s4_concat",
        "--partition", "nlp",
        "--account", "robinjia_875",
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--dependency", f"afterok:{s4_arr}",
        "--output", f"{exp_s4}/logs/concat_%j.out",
        "--wrap", s4_concat_cmd,
    ])

    return {
        "stage2": {"makeup_array_jid": makeup, "original_array_jid": original,
                   "concat_jid": concat, "exp_dir": exp_s2,
                   "output_file": output_file, "missing_shards": missing},
        "stage3": {"array_jid": s3_arr, "concat_jid": s3_concat,
                   "exp_dir": exp_s3, "output_file": s3_output_file},
        "stage4": {"array_jid": s4_arr, "concat_jid": s4_concat,
                   "exp_dir": exp_s4, "output_file": s4_output_file},
    }


def main():
    state: dict = {"measures": list(MEASURES), "by_measure": {}}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())

    for measure, cfg in MEASURES.items():
        print(f"\n========= make-up: {measure} (missing shards {cfg['missing']}) =========")
        result = submit_one_measure(measure, cfg)
        state["by_measure"][measure] = result

    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nWrote state to {STATE_PATH}")
    print("\n=== Job chain summary ===")
    for m, r in state["by_measure"].items():
        s2, s3, s4 = r["stage2"], r["stage3"], r["stage4"]
        print(f"\n{m}:")
        print(f"  S2 makeup: {s2['makeup_array_jid']} (depends on original {s2['original_array_jid']})")
        print(f"  S2 concat: {s2['concat_jid']}")
        print(f"  S3 array:  {s3['array_jid']}  →  concat {s3['concat_jid']}")
        print(f"  S4 array:  {s4['array_jid']}  →  concat {s4['concat_jid']}")


if __name__ == "__main__":
    main()
