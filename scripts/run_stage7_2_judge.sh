#!/bin/bash
# Stage 7.2 — LLM-as-judge evaluation of every labelled (input, model, measure) triple.
#
# Inputs:  data/eval_responses.jsonl (Stage 7.1 output)
# Output:  data/eval_judge_results.jsonl
# Judge:   anthropic/claude-opus-4.6 via OpenRouter (no thinking, temperature=0)
# Keys:    .keys.json or legacy .openrouter_key fallback
#
# Total scale after local-Qwen merge:
# 3,147 labelled input-measure pairs × 25 models = 78,675 judge calls.
# DSPy disk-cached, so re-runs are free; resumes from any partially-written
# output file by skipping (user_input, measure, model_name) triples already present.
set -e

cd /project2/robinjia_875/ehuang97/socialai

uv run python -m src.evaluation.judge_responses \
    --input data/eval_responses.jsonl \
    --output data/eval_judge_results.jsonl \
    --keys .keys.json \
    --concurrency 30

echo "=== Done ==="
wc -l data/eval_judge_results.jsonl
