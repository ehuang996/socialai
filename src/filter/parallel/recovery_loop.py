"""Recovery / completion orchestrator for the 4 measures' Stage 2/3/4 chains.

Design
------
SLURM `--requeue` does NOT bring back tasks preempted on `nlp` (PreemptMode=CANCEL),
so we replace it with an explicit poll-and-resubmit loop. For each measure we
track every Stage 2 array submission. A shard is "done" if any of its array
attempts ended in COMPLETED across submissions. If a shard is terminal but not
COMPLETED (PREEMPTED, FAILED, …) we re-submit a make-up array for just that
shard. When all 8 shards are COMPLETED we run dedup_concat for Stage 2 and then
chain Stage 3 + Stage 4 (with the wc-l-instead-of-echo concat fix).

Run:
    cd /project2/jessetho_1732/wangzhu/socialai
    nohup .venv/bin/python src/filter/parallel/recovery_loop.py \\
        > logs/recovery_loop.log 2>&1 &

State persists in src/filter/parallel/recovery_state.json so the script is
restartable.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/project2/jessetho_1732/wangzhu/socialai")
KEY = ".openrouter_key"
STAGE1_INPUT = "experiments/00_coarse_filter/results/1B_intentional_human_speech_coarse.jsonl"
STATE_PATH = PROJECT / "src/filter/parallel/recovery_state.json"
NUM_SHARDS = 8
NUM_API_SHARDS = 4
POLL_SECONDS = 60

# Initial config — Stage 2 array IDs known so far per measure.
INITIAL_MEASURES = {
    "2A_fabricated_personal_information": {
        "stage2_arrays": ["3309903", "3312348"],
        "stage2_exp": "experiments/25_low_quality_filter",
        # 2A's Stage 2 is already fully COMPLETED + concat done; Stage 3 array
        # already running. We only need to submit a corrected Stage 3 concat +
        # Stage 4 chain when Stage 3 array finishes.
        "stage2_done": True,
        "stage2_concat_done": True,
        "stage3_array_jid": "3312350",  # already running with fixed slicing
        "stage3_concat_jid": None,
        "stage4_array_jid": None,
        "stage4_concat_jid": None,
    },
    "2B_emotion_expression": {
        "stage2_arrays": ["3309836", "3312354"],
        "stage2_exp": "experiments/16_low_quality_filter",
        # 2B Stage 2 concat already ran on incomplete data; per-shard files
        # rebuilt by rebuild_part_files.py. Need to re-run shards 0,1 to fill
        # the gap, then re-run dedup_concat (overwrites 522K with 573K), then
        # full Stage 3 chain (DSPy cache makes Stage 3 fast).
        "stage2_done": False,
        "stage2_concat_done": False,
        "stage3_array_jid": None,
        "stage3_concat_jid": None,
        "stage4_array_jid": None,
        "stage4_concat_jid": None,
    },
    "2C_flattery_tone": {
        "stage2_arrays": ["3309842", "3312360"],
        "stage2_exp": "experiments/19_low_quality_filter",
        "stage2_done": False,
        "stage2_concat_done": False,
        "stage3_array_jid": None,
        "stage3_concat_jid": None,
        "stage4_array_jid": None,
        "stage4_concat_jid": None,
    },
    "2D_human_relationship_encouragement": {
        "stage2_arrays": ["3309848", "3312366"],
        "stage2_exp": "experiments/22_low_quality_filter",
        "stage2_done": False,
        "stage2_concat_done": False,
        "stage3_array_jid": None,
        "stage3_concat_jid": None,
        "stage4_array_jid": None,
        "stage4_concat_jid": None,
    },
}

S3_BASE_NUM = {  # exp dir number for stage 3 (= s2 num + 1)
    "2A_fabricated_personal_information": 26,
    "2B_emotion_expression": 17,
    "2C_flattery_tone": 20,
    "2D_human_relationship_encouragement": 23,
}
S4_BASE_NUM = {
    "2A_fabricated_personal_information": 27,
    "2B_emotion_expression": 18,
    "2C_flattery_tone": 21,
    "2D_human_relationship_encouragement": 24,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sacct_array(jid: str) -> dict[int, str]:
    """Return {shard_id: latest_state} for one array master JID."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", jid, "--format=JobID,State", "-np"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {}
    states: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        full_jid = parts[0]
        st = parts[1]
        # array-task ids look like "12345_3" or "12345_3.batch"
        if "." in full_jid:
            continue
        if "_" not in full_jid:
            continue
        try:
            sid = int(full_jid.split("_", 1)[1].split(".")[0])
        except ValueError:
            continue
        # Prefer COMPLETED; otherwise keep the latest non-completed state
        prev = states.get(sid)
        if prev == "COMPLETED":
            continue
        states[sid] = st
    return states


def shard_summary(arrays: list[str]) -> tuple[set[int], set[int]]:
    """For a list of array JIDs, return (completed_shards, in_flight_shards)."""
    completed: set[int] = set()
    in_flight: set[int] = set()
    in_flight_states = {"RUNNING", "PENDING", "REQUEUED", "CONFIGURING", "RESIZING"}
    for jid in arrays:
        for sid, st in sacct_array(jid).items():
            if st == "COMPLETED":
                completed.add(sid)
            elif st in in_flight_states:
                in_flight.add(sid)
    in_flight -= completed
    return completed, in_flight


def sbatch_parsable(args: list[str]) -> str:
    out = subprocess.check_output(
        ["sbatch", "--parsable", *args],
        text=True, cwd=PROJECT,
    ).strip()
    return out


def submit_stage2_makeup(measure: str, exp_s2: str, missing: list[int]) -> str:
    array_spec = ",".join(str(s) for s in missing)
    output_base = f"data/{measure}_scores"
    # MAKEUP_NUM_SHARDS forces JudgeConfig to use the true sharding (8) instead
    # of SLURM_ARRAY_TASK_COUNT, which would equal len(missing) for partial
    # arrays — leading to wrong slices.
    env = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=low_quality_filter",
        f"EXPERIMENT_DIR={exp_s2}",
        f"INPUT_PATH={STAGE1_INPUT}",
        f"OUTPUT_BASE={output_base}",
        "EXTRA_ARGS=--concurrency_limit 64",
        f"MAKEUP_NUM_SHARDS={NUM_SHARDS}",
    ])
    return sbatch_parsable([
        "--job-name", f"{measure}_s2_recovery",
        "--array", array_spec,
        "--output", f"{exp_s2}/logs/job_%A_%a.out",
        "--error", f"{exp_s2}/logs/job_%A_%a.err",
        "--export", env,
        "slurm/run_vllm_stage.sbatch",
    ])


def submit_stage2_concat(measure: str, exp_s2: str, dep_jids: list[str]) -> str:
    output_base = f"data/{measure}_scores"
    output_file = f"{exp_s2}/results/{measure}_scores.jsonl"
    dep_clause = "afterany:" + ":".join(dep_jids) if dep_jids else None
    cmd = (
        "module purge && module load gcc/13.3.0 && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "uv run --no-sync python src/filter/parallel/dedup_concat.py "
        f"--output_base {output_base} --output_file {output_file}"
    )
    args = [
        "--job-name", f"{measure}_s2_concat",
        "--partition", "nlp", "--account", "robinjia_875",
        "--chdir", str(PROJECT),
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--output", f"{exp_s2}/logs/concat_%j.out",
        "--wrap", cmd,
    ]
    if dep_clause:
        args = ["--dependency", dep_clause] + args
    return sbatch_parsable(args)


def submit_stage3(measure: str, dep_jid: str | None, s2_output_file: str) -> tuple[str, str]:
    """Submit Stage 3 array + concat. Returns (array_jid, concat_jid)."""
    s3_num = S3_BASE_NUM[measure]
    exp_s3 = f"experiments/{s3_num:02d}_high_quality_filter"
    (PROJECT / exp_s3 / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT / exp_s3 / "results").mkdir(parents=True, exist_ok=True)
    output_base = f"data/{measure}_high_quality"
    output_file = f"{exp_s3}/results/{measure}_high_quality.jsonl"
    env = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=high_quality_filter",
        f"EXPERIMENT_DIR={exp_s3}",
        f"INPUT_PATH={s2_output_file}",
        f"OUTPUT_BASE={output_base}",
        f"KEY={KEY}",
        "EXTRA_ARGS=",
    ])
    arr_args = [
        "--job-name", f"{measure}_s3",
        "--array", f"0-{NUM_API_SHARDS - 1}",
        "--output", f"{exp_s3}/logs/job_%A_%a.out",
        "--error", f"{exp_s3}/logs/job_%A_%a.err",
        "--export", env,
        "slurm/run_api_stage.sbatch",
    ]
    if dep_jid:
        arr_args = ["--dependency", f"afterok:{dep_jid}"] + arr_args
    arr_jid = sbatch_parsable(arr_args)
    # FIXED concat: wc -l, no `echo … -> FILE` (which bash redirects, clobbering FILE)
    concat_cmd = (
        f"cat {output_base}_part_*.jsonl > {output_file} && "
        f"rm {output_base}_part_*.jsonl && "
        f"wc -l {output_file}"
    )
    concat_jid = sbatch_parsable([
        "--dependency", f"afterok:{arr_jid}",
        "--job-name", f"{measure}_s3_concat",
        "--partition", "nlp", "--account", "robinjia_875",
        "--chdir", str(PROJECT),
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--output", f"{exp_s3}/logs/concat_%j.out",
        "--wrap", concat_cmd,
    ])
    return arr_jid, concat_jid


def submit_stage4(measure: str, dep_jid: str, s3_output_file: str) -> tuple[str, str]:
    s4_num = S4_BASE_NUM[measure]
    exp_s4 = f"experiments/{s4_num:02d}_final_filter"
    (PROJECT / exp_s4 / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT / exp_s4 / "results").mkdir(parents=True, exist_ok=True)
    output_base = f"data/{measure}_final"
    output_file = f"{exp_s4}/results/{measure}_final.jsonl"
    env = ",".join([
        "ALL",
        f"MEASURE={measure}",
        "STAGE=final_filter",
        f"EXPERIMENT_DIR={exp_s4}",
        f"INPUT_PATH={s3_output_file}",
        f"OUTPUT_BASE={output_base}",
        f"KEY={KEY}",
        "EXTRA_ARGS=",
    ])
    arr_jid = sbatch_parsable([
        "--dependency", f"afterok:{dep_jid}",
        "--job-name", f"{measure}_s4",
        "--array", f"0-{NUM_API_SHARDS - 1}",
        "--output", f"{exp_s4}/logs/job_%A_%a.out",
        "--error", f"{exp_s4}/logs/job_%A_%a.err",
        "--export", env,
        "slurm/run_api_stage.sbatch",
    ])
    concat_cmd = (
        f"cat {output_base}_part_*.jsonl > {output_file} && "
        f"rm {output_base}_part_*.jsonl && "
        f"wc -l {output_file}"
    )
    concat_jid = sbatch_parsable([
        "--dependency", f"afterok:{arr_jid}",
        "--job-name", f"{measure}_s4_concat",
        "--partition", "nlp", "--account", "robinjia_875",
        "--chdir", str(PROJECT),
        "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
        "--output", f"{exp_s4}/logs/concat_%j.out",
        "--wrap", concat_cmd,
    ])
    return arr_jid, concat_jid


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"measures": INITIAL_MEASURES}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def is_finished(state: dict, measure: str) -> bool:
    m = state["measures"][measure]
    return m["stage4_concat_jid"] is not None


def step_measure(state: dict, measure: str) -> None:
    m = state["measures"][measure]
    if is_finished(state, measure):
        return
    s2_output_file = f"{m['stage2_exp']}/results/{measure}_scores.jsonl"

    # --- Stage 2 → Stage 3 path ---
    if not m["stage2_done"]:
        completed, in_flight = shard_summary(m["stage2_arrays"])
        if len(completed) == NUM_SHARDS:
            print(f"[{measure}] Stage 2 all 8 shards COMPLETED. ", flush=True)
            m["stage2_done"] = True
        elif not in_flight:
            missing = sorted(set(range(NUM_SHARDS)) - completed)
            if missing:
                jid = submit_stage2_makeup(measure, m["stage2_exp"], missing)
                print(f"[{measure}] Submitted Stage 2 make-up for shards {missing} → {jid}", flush=True)
                m["stage2_arrays"].append(jid)

    if m["stage2_done"] and not m["stage2_concat_done"]:
        # Run dedup_concat (no upstream deps since arrays are all terminal)
        jid = submit_stage2_concat(measure, m["stage2_exp"], dep_jids=[])
        print(f"[{measure}] Submitted Stage 2 dedup_concat (re-merge) → {jid}", flush=True)
        m["stage2_concat_done"] = True
        m["stage2_concat_jid"] = jid
        # Stage 3 chain depends on this concat finishing
        s3_arr, s3_concat = submit_stage3(measure, dep_jid=jid, s2_output_file=s2_output_file)
        print(f"[{measure}] Submitted Stage 3: array {s3_arr} → concat {s3_concat}", flush=True)
        m["stage3_array_jid"] = s3_arr
        m["stage3_concat_jid"] = s3_concat
        s4_arr, s4_concat = submit_stage4(measure, dep_jid=s3_concat,
                                          s3_output_file=f"experiments/{S3_BASE_NUM[measure]:02d}_high_quality_filter/results/{measure}_high_quality.jsonl")
        print(f"[{measure}] Submitted Stage 4: array {s4_arr} → concat {s4_concat}", flush=True)
        m["stage4_array_jid"] = s4_arr
        m["stage4_concat_jid"] = s4_concat
        return

    # 2A path: Stage 2 already done + concat done, but Stage 3 array was submitted
    # earlier and downstream chain (concat + Stage 4) hasn't been submitted yet.
    if (m["stage2_concat_done"] and m["stage3_array_jid"] is not None
            and m["stage3_concat_jid"] is None):
        # Submit Stage 3 concat and Stage 4 chain depending on the existing array
        output_base = f"data/{measure}_high_quality"
        s3_num = S3_BASE_NUM[measure]
        exp_s3 = f"experiments/{s3_num:02d}_high_quality_filter"
        s3_output_file = f"{exp_s3}/results/{measure}_high_quality.jsonl"
        concat_cmd = (
            f"cat {output_base}_part_*.jsonl > {s3_output_file} && "
            f"rm {output_base}_part_*.jsonl && "
            f"wc -l {s3_output_file}"
        )
        s3_concat = sbatch_parsable([
            "--dependency", f"afterok:{m['stage3_array_jid']}",
            "--job-name", f"{measure}_s3_concat",
            "--partition", "nlp", "--account", "robinjia_875",
            "--chdir", str(PROJECT),
            "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=0:30:00",
            "--output", f"{exp_s3}/logs/concat_%j.out",
            "--wrap", concat_cmd,
        ])
        print(f"[{measure}] Submitted Stage 3 concat (fixed wrap) → {s3_concat}", flush=True)
        m["stage3_concat_jid"] = s3_concat
        s4_arr, s4_concat = submit_stage4(measure, dep_jid=s3_concat,
                                          s3_output_file=s3_output_file)
        print(f"[{measure}] Submitted Stage 4: array {s4_arr} → concat {s4_concat}", flush=True)
        m["stage4_array_jid"] = s4_arr
        m["stage4_concat_jid"] = s4_concat


def main():
    state = load_state()
    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== iteration {iteration} @ {time.strftime('%H:%M:%S')} ===", flush=True)
        for measure in list(state["measures"].keys()):
            try:
                step_measure(state, measure)
            except Exception as e:
                print(f"[{measure}] ERROR in step: {e}", flush=True)
        save_state(state)
        if all(is_finished(state, m) for m in state["measures"]):
            print("All 4 measures finished (Stage 4 chains submitted). Loop done.", flush=True)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
