"""
Test all User Input rows from anthropomorphism_seedset_v1.csv against
GPT-5.2 and Gemini 3 Pro via OpenRouter API.

Usage:
    python run_openrouter_eval.py
"""

import argparse
import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "gpt_5_2": "openai/gpt-5.2",
    "gemini_3_pro": "google/gemini-3-pro-preview",
}

CHECKPOINT_EVERY = 50  # save progress every N rows


def load_api_key(key_path: str) -> str:
    with open(key_path) as f:
        return f.read().strip()


async def query_model(client: AsyncOpenAI, model_id: str, user_input: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_input}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


async def process_row(client: AsyncOpenAI, row: pd.Series, semaphore: asyncio.Semaphore) -> dict:
    user_input = row["User Input"]
    tasks = {
        col: query_model(client, model_id, user_input, semaphore)
        for col, model_id in MODELS.items()
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def run(input_csv: str, output_csv: str, key_path: str, concurrency: int):
    api_key = load_api_key(key_path)
    client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    semaphore = asyncio.Semaphore(concurrency)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Resume from checkpoint if output already exists
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_count = len(existing)
        print(f"Resuming from checkpoint: {done_count} rows already done")
        results_so_far = existing.to_dict("records")
        df = df.iloc[done_count:].reset_index(drop=True)
    else:
        done_count = 0
        results_so_far = []

    total = len(df)
    print(f"Rows remaining: {total}")

    async def process_with_progress(idx: int, row: pd.Series) -> dict:
        model_responses = await process_row(client, row, semaphore)
        return {
            "User Input": row["User Input"],
            "Original Assistant Response": row["Assistant Response"],
            "Actual Speaker": row["Actual Speaker"],
            **{f"GPT-5.2 Response": model_responses["gpt_5_2"],
               "Gemini 3 Pro Response": model_responses["gemini_3_pro"]},
        }

    tasks = [process_with_progress(i, row) for i, row in df.iterrows()]

    batch_results = []
    for i, coro in enumerate(tqdm_asyncio.as_completed(tasks, total=total, desc="Querying models")):
        result = await coro
        batch_results.append(result)

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == total:
            # Merge with previous results and save checkpoint
            all_results = results_so_far + batch_results
            pd.DataFrame(all_results).to_csv(output_csv, index=False)
            print(f"\nCheckpoint saved: {len(all_results)} rows total -> {output_csv}")

    all_results = results_so_far + batch_results
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\nDone! {len(all_results)} rows saved to {output_csv}")

    results_dir = os.path.dirname(output_csv)
    gpt_jsonl = os.path.join(results_dir, "gpt5.2.jsonl")
    gemini_jsonl = os.path.join(results_dir, "gemini3pro.jsonl")

    with open(gpt_jsonl, "w", encoding="utf-8") as fg, open(gemini_jsonl, "w", encoding="utf-8") as fm:
        for row in all_results:
            base = {
                "user_input": row["User Input"],
                "original_assistant_response": row["Original Assistant Response"],
                "actual_speaker": row["Actual Speaker"],
            }
            fg.write(json.dumps({**base, "new_model_response": row["GPT-5.2 Response"]}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({**base, "new_model_response": row["Gemini 3 Pro Response"]}, ensure_ascii=False) + "\n")

    print(f"JSONL files written: {gpt_jsonl}, {gemini_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate user inputs via OpenRouter")
    parser.add_argument("--input", default="../../anthropomorphism_seedset_v1.csv", help="Input CSV path")
    parser.add_argument("--output", default="results/raw.csv", help="Output CSV path")
    parser.add_argument("--key", default="../../key.sh", help="Path to file containing OpenRouter API key")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    args = parser.parse_args()

    asyncio.run(run(args.input, args.output, args.key, args.concurrency))


if __name__ == "__main__":
    main()
