#!/bin/bash
#SBATCH --job-name=wildchat_download
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

mkdir -p logs data

module purge
module load gcc/13.3.0

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR

export HF_HOME=/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface
export TMPDIR=/project2/robinjia_875/wangzhu/eric_huang/.cache/tmp
mkdir -p $TMPDIR

uv run python scripts/download.py \
    --dataset_name "allenai/WildChat-4.8M" \
    --output_path data/wildchat_raw.jsonl

echo "Download complete."
