#!/bin/bash
# Stage 7.1 — Generate API/direct model responses for the full evaluation set.
#
# Inputs:  data/final_dataset.jsonl (969 rows: seedset + manual-pass synthetic)
# Output:  data/eval_responses.jsonl
# Models:  all 23 API/direct entries in MODEL_REGISTRY
# Keys:    .keys.json ({openrouter, deepseek, anthropic}), or legacy
#          .openrouter_key + optional .deepseek_key fallback
#
# Total scale: 23 × 969 = 22,287 generations.
# Run scripts/run_stage7_1_local_qwen.sh afterward to add qwen3_1_7b and
# qwen3_4b locally, bringing the combined Stage 7.1 output to 25 models.
# DSPy disk-cached, so re-runs are free; the script also resumes from any
# partially-written output file.
set -e

cd /project2/robinjia_875/ehuang97/socialai

uv run python -m src.evaluation.generate_responses \
    --input data/final_dataset.jsonl \
    --output data/eval_responses.jsonl \
    --keys .keys.json

echo "=== Done ==="
wc -l data/eval_responses.jsonl
