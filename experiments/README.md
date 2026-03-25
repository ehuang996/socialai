# Experiments

Each numbered folder is a self-contained experiment with its own sbatch scripts, results, and logs.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 01 | `01_chit_chat_filter` | Active | Filter WildChat-1M for chit-chat (downloads data, 6-shard SLURM array) |
| 02 | `02_anthropomorphic_judge` | Active | Detect anthropomorphic behavior in chit-chat (6-shard SLURM array) |
| 03 | `03_openrouter_eval` | Active | Evaluate seed prompts against frontier models via OpenRouter API |

## Convention
- Create new numbered folders (`04_xxx/`, `05_xxx/`, ...) for new experiments — don't edit old ones.
- Each folder contains: sbatch scripts, `results/`, `logs/`, `figures/`, and `README.md` (observations).
- Sbatch scripts call `src/filter/run.py --measure <measure> --stage <stage> --experiment_dir experiments/<folder>`, which snapshots `src/filter/` into the experiment folder for version control.
- This README should only contain brief descriptions of each experiment. Detailed setup, results, and observations belong in each experiment's own `README.md`.
