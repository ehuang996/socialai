"""Unified entrypoint: dispatches to measure.<measure>.<stage>.cli() or .main().

Usage:
    uv run python src/filter/run.py --measure anthropomorphism --stage coarse_filter [judge args...]
    uv run python src/filter/run.py --measure anthropomorphism --stage low_quality_filter [judge args...]
    uv run python src/filter/run.py --measure anthropomorphism --stage high_quality_filter [args...]

The --experiment_dir flag triggers a snapshot of src/ into that directory before running.
"""

import argparse
import importlib
import os
import re
import shutil
import sys
from pathlib import Path


def try_concat_shards(output_path: str, num_shards: int, combined_dir: Path | None = None) -> None:
    """Concatenate _part_N shard files into a single output file and delete the parts.

    Called after the last shard (by ID) finishes. If any shard file is missing
    (i.e. still running or failed), concat is skipped and a message is printed.

    combined_dir: directory for the merged file (defaults to same dir as shard files).
    """
    p = Path(output_path)
    base = re.sub(r"_part_\d+$", "", p.stem)  # e.g. wildchat_scores_part_3 -> wildchat_scores
    shard_files = [p.parent / f"{base}_part_{i}{p.suffix}" for i in range(num_shards)]

    missing = [f for f in shard_files if not f.exists()]
    if missing:
        print(f"[run.py] Concat skipped: {len(missing)} shard file(s) not yet present: {missing}")
        return

    dest_dir = combined_dir if combined_dir is not None else p.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    combined = dest_dir / f"{base}{p.suffix}"
    with open(combined, "w") as out:
        for shard_file in shard_files:
            with open(shard_file) as f:
                shutil.copyfileobj(f, out)

    for shard_file in shard_files:
        shard_file.unlink()

    total = sum(1 for _ in open(combined))
    print(f"[run.py] Concatenated {num_shards} shards -> {combined} ({total} rows)")


def main():
    # Parse only our own args; leave the rest for the submodule
    top_parser = argparse.ArgumentParser(add_help=False)
    top_parser.add_argument("--measure", required=True,
                            help="Measure name (e.g., anthropomorphism, sycophancy)")
    top_parser.add_argument("--stage", required=True,
                            help="Stage name matching the Python file (e.g., coarse_filter, low_quality_filter, high_quality_filter)")
    top_parser.add_argument("--experiment_dir", default=None,
                            help="If set, copy src/ into <experiment_dir>/src before running")

    our_args, remaining = top_parser.parse_known_args()

    # Snapshot src/ into the experiment directory — only on shard 0 (or non-array jobs)
    # to avoid race conditions when multiple SLURM array tasks start simultaneously.
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    should_snapshot = task_id is None or task_id == "0"
    if our_args.experiment_dir is not None and should_snapshot:
        src_dir = Path(__file__).parent  # i.e., src/filter/
        dest_dir = Path(our_args.experiment_dir) / "src"
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
        print(f"[run.py] Snapshotted src/ -> {dest_dir}")

    # Dynamically import measure.<measure>.<stage>.
    # If not found, fall back to measure.base.<stage> and inject --measure_dir
    # so the generic implementation can locate the measure's JSON config files.
    measure_module_path = f"measure.{our_args.measure}.{our_args.stage}"
    base_module_path = f"measure.base.{our_args.stage}"
    try:
        module = importlib.import_module(measure_module_path)
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(base_module_path)
            measure_dir = str(Path(__file__).parent / "measure" / our_args.measure)
            remaining = ["--measure_dir", measure_dir] + remaining
            print(f"[run.py] Using base implementation for '{our_args.stage}' with measure_dir={measure_dir}")
        except ModuleNotFoundError as e:
            print(f"ERROR: Could not import '{measure_module_path}' or '{base_module_path}': {e}",
                  file=sys.stderr)
            sys.exit(1)

    # Strip our flags from sys.argv so the submodule's argparse sees only its own flags.
    # Done here (after potential --measure_dir injection) so cli() sees the complete args.
    sys.argv = [sys.argv[0]] + remaining

    # Call cli() for Judge subclasses, main() for other modules
    if hasattr(module, "cli"):
        module.cli()
    elif hasattr(module, "main"):
        module.main()
    else:
        print(f"ERROR: '{module_path}' has neither cli() nor main().", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
