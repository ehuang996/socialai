# 04 — Anthropomorphic Judge

**Status:** Active
**Goal:** Detect anthropomorphic AI behavior in chit-chat conversations filtered by experiment 03.

## Prerequisites

`data/wildchat_chit_chat.jsonl` must exist. Generate it by running experiment 03 and concatenating the shards:

```bash
cat data/wildchat_chit_chat_part_*.jsonl > data/wildchat_chit_chat.jsonl
```

## Pipeline

### Step 1: Run the Judge (6-shard array job)

```bash
sbatch --array=0-5 experiments/02_anthropomorphic_judge/run_anthropomorphic.sbatch
```

Monitor: `squeue -u $USER`
Outputs: `data/wildchat_scores_part_{0..5}.jsonl`

### Step 2: Concatenate Shards

After all 6 array tasks complete:

```bash
cat data/wildchat_scores_part_*.jsonl > data/wildchat_scores.jsonl
wc -l data/wildchat_scores.jsonl
```

## Output Format

Each line in the output JSONL (old-format for compatibility):

```json
{
  "hash": "<conversation_hash>",
  "user_input": "<first user turn>",
  "assistant_response": "<first assistant turn>",
  "timestamp": "<ISO timestamp>",
  "speaker_reasoning": "...",
  "actual_speaker": "chatbot or persona name",
  "anthropomorphism_reasoning": "...",
  "anthropomorphic_score": -1 | 0 | 1
}
```

Scores: `-1` = robotic/transparent, `0` = neutral, `1` = human-simulating.

## Resources

- GPU: 1× A6000 per shard
- Partition: `nlp_hiprio`
- Memory: 40G per shard
- vLLM model: `Qwen/Qwen3-VL-8B-Instruct`
- Concurrency: 64 async workers per shard

## Results

(To be filled after run)
