#!/bin/bash
# Run Stage 6 for single_turn_final_452.jsonl with all 14 models
set -e

cd /project2/robinjia_875/ehuang97/socialai

uv run python src/evaluation/stage6_generate_responses/generate_responses.py \
    --input data/single_turn_final_452.jsonl \
    --output data/single_turn_model_responses_452.jsonl \
    --key .openrouter_key \
    --model_set all

echo "=== Done ==="
wc -l data/single_turn_model_responses_452.jsonl
