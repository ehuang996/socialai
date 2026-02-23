"""Stage 2: Anthropomorphic behavior judge for chit-chat conversations.

Passthrough to AnthropomorphicJudge CLI. Run via the SLURM array script or directly:

    uv run python experiments/02_anthropomorphic_judge/run.py \
        --input_path data/wildchat_chit_chat.jsonl \
        --output_path data/wildchat_scores_part_0.jsonl

Prerequisites: data/wildchat_chit_chat.jsonl must exist (output of experiment 03).
See run_anthropomorphic.sbatch for the full SLURM array job.
"""
from pipeline.anthropomorphic import AnthropomorphicJudge

if __name__ == "__main__":
    AnthropomorphicJudge.cli()
