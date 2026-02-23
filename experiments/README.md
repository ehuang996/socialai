# Experiments

Each numbered folder is a self-contained experiment with its own `run.py`, results, and logs.

| # | Name | Status | Description |
|---|------|--------|-------------|
| 01 | `01_chit_chat_filter` | Active | Filter WildChat-1M for chit-chat (downloads data, 6-shard SLURM array) |
| 02 | `02_anthropomorphic_judge` | Active | Detect anthropomorphic behavior in chit-chat (6-shard SLURM array) |
| 03 | `03_openrouter_eval` | Active | Evaluate seed prompts against GPT-5.2 and Gemini 3 Pro via OpenRouter |

## Convention
- Create new numbered folders (`01_xxx/`, `02_xxx/`, ...) for new experiments — don't edit old ones.
- Each folder contains: `run.py`, `results/`, `logs/`, `figures/`, and `README.md` (observations).
- This README should only contain brief descriptions of each experiment. Detailed setup, results, and observations belong in each experiment's own `README.md`.
