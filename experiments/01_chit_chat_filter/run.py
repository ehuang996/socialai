"""Stage 1: Chit-chat filter for WildChat conversations.

Passthrough to ChitChatJudge CLI. Run via the SLURM array script or directly:

    uv run python experiments/01_chit_chat_filter/run.py \
        --input_path data/wildchat_raw.jsonl \
        --output_path data/wildchat_chit_chat_part_0.jsonl \
        --prompt-version v5

See run_chit_chat.sbatch for the full SLURM array job.
"""
from pipeline.chit_chat import ChitChatJudge

if __name__ == "__main__":
    ChitChatJudge.cli()
