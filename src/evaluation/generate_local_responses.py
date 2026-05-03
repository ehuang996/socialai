"""Stage 7.1 local-HF add-on generation through a local vLLM server.

This is used for models that are available on Hugging Face but not currently
routable through OpenRouter. Output rows intentionally match
generate_responses.py:

{
  "user_input": "...",
  "measure": [...],
  "synthetic": false,
  "language": "English",
  "model_responses": {
    "qwen3_4b": {"assistant_response": "..."}
  }
}

Part files from this script are merged into data/eval_responses.jsonl by
merge_local_responses.py before Stage 7.2 judging.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


@dataclass(frozen=True)
class LocalModelConfig:
    hf_model: str
    max_tokens: int = 2048
    temperature: float = 0.7
    extra_body: dict = field(default_factory=dict)


LOCAL_MODEL_REGISTRY: dict[str, LocalModelConfig] = {
    # OpenRouter currently lists these model IDs with 0 active endpoints, so we
    # serve them locally from Hugging Face. Qwen3 thinking is disabled to match
    # the non-thinking OpenRouter Qwen small-model configuration.
    "qwen3_1_7b": LocalModelConfig(
        hf_model="Qwen/Qwen3-1.7B",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3_4b": LocalModelConfig(
        hf_model="Qwen/Qwen3-4B",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def load_completed(output_path: Path, model_col: str) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model_col in row.get("model_responses", {}):
                completed.add(row.get("user_input", ""))
    return completed


def load_input_rows(input_path: Path, shard_id: int, num_shards: int) -> list[dict]:
    with input_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if num_shards > 1:
        return rows[shard_id::num_shards]
    return rows


def extract_message_text(response) -> str:
    message = response.choices[0].message
    content = message.content or ""
    return content.strip()


async def call_with_retry(
    client: AsyncOpenAI,
    row: dict,
    model_col: str,
    served_model_name: str,
    cfg: LocalModelConfig,
    max_retries: int,
) -> dict:
    messages = [{"role": "user", "content": row["user_input"]}]
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=served_model_name,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_body=cfg.extra_body,
            )
            text = extract_message_text(response)
            if not text:
                raise ValueError("Model returned empty response")
            model_result = {"assistant_response": text}
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
            else:
                model_result = {"error": last_error}

    return {
        "user_input": row["user_input"],
        "measure": row.get("measure", []),
        "synthetic": row.get("synthetic", False),
        "language": row.get("language", "English"),
        "model_responses": {model_col: model_result},
    }


async def run_async(args: argparse.Namespace) -> None:
    if args.model_col not in LOCAL_MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown local model column {args.model_col!r}. "
            f"Available: {sorted(LOCAL_MODEL_REGISTRY)}"
        )

    cfg = LOCAL_MODEL_REGISTRY[args.model_col]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard_rows = load_input_rows(Path(args.input), args.shard_id, args.num_shards)
    completed = load_completed(output_path, args.model_col)
    rows = [r for r in shard_rows if r.get("user_input") not in completed]

    print(
        f"[{args.model_col}] shard {args.shard_id}/{args.num_shards}: "
        f"{len(rows)} rows to run (slice size {len(shard_rows)}, "
        f"skipped {len(completed)} completed)"
    )
    print(f"[{args.model_col}] HF model: {cfg.hf_model}")
    print(f"[{args.model_col}] API base: {args.api_base}")

    if not rows:
        print(f"[{args.model_col}] Nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.api_base, api_key="EMPTY", timeout=args.timeout)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row: dict) -> dict:
        async with sem:
            return await call_with_retry(
                client=client,
                row=row,
                model_col=args.model_col,
                served_model_name=args.served_model_name,
                cfg=cfg,
                max_retries=args.max_retries,
            )

    with output_path.open("a") as f_out:
        tasks = [one(row) for row in rows]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=args.model_col):
            result = await coro
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    print(f"[{args.model_col}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage 7.1 responses from a local vLLM-served HF model."
    )
    parser.add_argument("--input", default="data/final_dataset.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_col", required=True, choices=sorted(LOCAL_MODEL_REGISTRY))
    parser.add_argument("--served_model_name", default="")
    parser.add_argument("--api_base", default="http://localhost:8001/v1")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--shard_id", type=int, default=env_int("SLURM_ARRAY_TASK_ID", 0))
    parser.add_argument(
        "--num_shards",
        type=int,
        default=env_int("SLURM_ARRAY_TASK_COUNT", 1),
    )
    args = parser.parse_args()

    if not args.served_model_name:
        args.served_model_name = args.model_col

    t0 = time.time()
    asyncio.run(run_async(args))
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
