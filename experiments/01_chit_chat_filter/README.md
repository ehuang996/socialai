# 03 — Chit-Chat Filter

**Status:** Active
**Goal:** Filter WildChat-1M for casual chit-chat conversations using an LLM-as-a-judge (ChitChatJudge, prompt v5).

## Pipeline

### Step 0: Download & Preprocess (once)

Downloads `allenai/WildChat-1M` from the shared HuggingFace cache, keeps only English conversations, deduplicates by `conversation_hash`, and writes `data/wildchat_raw.jsonl`.

```bash
sbatch experiments/01_chit_chat_filter/run_download.sbatch
```

Monitor: `squeue -u $USER`
Expected output: `data/wildchat_raw.jsonl` (~500k–700k rows)

### Step 1: Chit-Chat Filter (6-shard array job)

Runs `ChitChatJudge` with prompt v5 on `wildchat_raw.jsonl` across 6 parallel SLURM tasks. Only rows where the judge returns `keep=true` are written.

```bash
sbatch --array=0-5 experiments/01_chit_chat_filter/run_chit_chat.sbatch
```

Monitor: `squeue -u $USER`
Outputs: `data/wildchat_chit_chat_part_{0..5}.jsonl`

### Step 2: Concatenate Shards

After all 6 array tasks complete:

```bash
cat data/wildchat_chit_chat_part_*.jsonl > data/wildchat_chit_chat.jsonl
wc -l data/wildchat_chit_chat.jsonl
```

This combined file is the input for experiment 04.

## Resources

- GPU: 1× A6000 per shard
- Partition: `nlp_hiprio`
- Memory: 40G per shard
- vLLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Concurrency: 100 async workers per shard

## Results

(To be filled after run)
