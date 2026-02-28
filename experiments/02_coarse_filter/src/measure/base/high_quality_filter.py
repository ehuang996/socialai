"""Generic high-quality filter — reads config from <measure_dir>/filter3.json.

Used as a fallback when a measure folder does not provide its own high_quality_filter.py.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHECKPOINT_EVERY = 50


def load_config(measure_dir: str) -> dict:
    with open(Path(measure_dir) / "filter3.json") as f:
        return json.load(f)


def load_api_key(key_path: str) -> str:
    with open(key_path) as f:
        return f.read().strip()


async def query_model(client, model_id, user_input, system_prompt, semaphore):
    async with semaphore:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_input})
            response = await client.chat.completions.create(model=model_id, messages=messages)
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"


async def process_row(client, row, models, system_prompt, semaphore):
    user_input = row["User Input"]
    tasks = {
        col: query_model(client, model_id, user_input, system_prompt, semaphore)
        for col, model_id in models.items()
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def run(input_csv: str, output_csv: str, key_path: str, concurrency: int, measure_dir: str):
    config = load_config(measure_dir)
    system_prompt = config.get("system_prompt", "")
    models = config["models"]

    client = AsyncOpenAI(api_key=load_api_key(key_path), base_url=OPENROUTER_BASE_URL)
    semaphore = asyncio.Semaphore(concurrency)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    if "low_quality_score" in df.columns:
        before = len(df)
        df = df[df["low_quality_score"] == 1].reset_index(drop=True)
        print(f"Filtered to low_quality_score=1: {len(df)}/{before} rows")

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

    async def process_with_progress(idx, row):
        model_responses = await process_row(client, row, models, system_prompt, semaphore)
        col_names = {col: col.replace("_", " ").title() + " Response" for col in models}
        return {
            "User Input": row["User Input"],
            "Original Assistant Response": row["Assistant Response"],
            "Actual Speaker": row["Actual Speaker"],
            **{col_names[col]: model_responses[col] for col in models},
        }

    tasks = [process_with_progress(i, row) for i, row in df.iterrows()]
    batch_results = []
    for i, coro in enumerate(tqdm_asyncio.as_completed(tasks, total=total, desc="Querying models")):
        result = await coro
        batch_results.append(result)
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == total:
            all_results = results_so_far + batch_results
            pd.DataFrame(all_results).to_csv(output_csv, index=False)
            print(f"\nCheckpoint saved: {len(all_results)} rows -> {output_csv}")

    all_results = results_so_far + batch_results
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\nDone! {len(all_results)} rows saved to {output_csv}")

    results_dir = os.path.dirname(output_csv)
    for col, model_id in models.items():
        jsonl_path = os.path.join(results_dir, f"{col}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in all_results:
                col_name = col.replace("_", " ").title() + " Response"
                f.write(json.dumps({
                    "user_input": row["User Input"],
                    "original_assistant_response": row["Original Assistant Response"],
                    "actual_speaker": row["Actual Speaker"],
                    "new_model_response": row.get(col_name, ""),
                }, ensure_ascii=False) + "\n")
        print(f"JSONL written: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate user inputs via OpenRouter")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="results/raw.csv", help="Output CSV path")
    parser.add_argument("--key", required=True, help="Path to file containing OpenRouter API key")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API requests")
    parser.add_argument("--measure_dir", type=str, default="",
                        help="Path to the measure folder containing filter3.json")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.key, args.concurrency, args.measure_dir))


if __name__ == "__main__":
    main()
