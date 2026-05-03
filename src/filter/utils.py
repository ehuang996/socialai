"""Base class for LLM-as-a-judge pipeline using vLLM server + AsyncOpenAI."""

import argparse
import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from param import add_judge_args


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    api_url: str = f"http://localhost:{os.getenv('VLLM_PORT', '8001')}/v1"
    input_path: str = ""
    output_path: str = ""
    concurrency_limit: int = 64
    max_tokens: int = 512
    temperature: float = 0.0
    # Read from env for SLURM array jobs.
    # MAKEUP_NUM_SHARDS lets us override SLURM_ARRAY_TASK_COUNT when submitting
    # a partial-list array (e.g. `--array=0,1`) for a make-up run: SLURM sets
    # SLURM_ARRAY_TASK_COUNT to the *submitted* count (2 here, not the original
    # 8), which would produce a 2-way slice instead of 1/8 slices. The override
    # forces num_shards back to the true sharding.
    shard_id: int = field(default_factory=lambda: int(os.getenv("SLURM_ARRAY_TASK_ID", "0")))
    num_shards: int = field(default_factory=lambda: int(
        os.getenv("MAKEUP_NUM_SHARDS")
        or os.getenv("SLURM_ARRAY_TASK_COUNT", "1")
    ))


class Judge(ABC):
    """Abstract base for LLM-as-a-judge filters.

    Subclasses define system_prompt(), format_conversation(), and judge_type().
    The base handles async inference, sharding, resumption, and JSONL I/O.
    """

    def __init__(self, config: JudgeConfig):
        self.config = config

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def format_conversation(self, row: dict) -> str | None:
        """Format a dataset row into the user message for the judge.

        Return None to skip the row (e.g., too short, too long).
        """
        ...

    @abstractmethod
    def judge_type(self) -> str:
        ...

    def parse_response(self, raw: str) -> dict:
        """Parse JSON from judge output, stripping markdown code fences if present."""
        text = raw.strip()
        if "```" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return {"error": "parse_failed", "raw_output": raw}

    def load_completed_ids(self) -> set[str]:
        """Load conversation hashes already in the output file for resumption."""
        completed = set()
        if not os.path.exists(self.config.output_path):
            return completed
        with open(self.config.output_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["conversation_hash"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed

    async def process_row(self, row: dict, client: AsyncOpenAI, sem: asyncio.Semaphore, f_out):
        """Process a single row: format → API call → parse → write."""
        async with sem:
            user_content = self.format_conversation(row)
            if user_content is None:
                return

            response = await client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            parsed = self.parse_response(raw)

            result = {
                "conversation_hash": row["conversation_hash"],
                **parsed,
                "raw_response": raw,
                "model": self.config.model_name,
                "judge_type": self.judge_type(),
            }
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    async def run(self):
        """Main entry: load input, shard, skip completed, run async inference, write JSONL.

        Order matters: SLICE the input by shard FIRST, THEN filter by completed_ids.
        Doing the filter first and slicing the filtered list (the previous bug) caused
        each shard to process rows that belong to OTHER shards' slices — turning resumed
        runs into ~7× duplicate work, since each shard's completed-id set only covers
        its own part file and not the rows other shards had already done.
        """
        client = AsyncOpenAI(base_url=self.config.api_url, api_key="EMPTY")
        sem = asyncio.Semaphore(self.config.concurrency_limit)
        completed = self.load_completed_ids()

        # Step 1: load all input rows
        all_rows = []
        with open(self.config.input_path) as f:
            for line in f:
                all_rows.append(json.loads(line))

        # Step 2: slice to this shard's deterministic subset BEFORE filtering
        if self.config.num_shards > 1:
            shard_rows = all_rows[self.config.shard_id :: self.config.num_shards]
        else:
            shard_rows = all_rows

        # Step 3: drop rows already completed for this shard
        rows = [r for r in shard_rows if r["conversation_hash"] not in completed]

        print(
            f"[Shard {self.config.shard_id}] Processing {len(rows)} rows "
            f"(slice size {len(shard_rows)}, skipped {len(completed)} completed)"
        )

        with open(self.config.output_path, "a") as f_out:
            tasks = [self.process_row(row, client, sem, f_out) for row in rows]
            await tqdm_asyncio.gather(*tasks)

        print(f"[Shard {self.config.shard_id}] Done.")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """Add common CLI arguments. Subclasses can extend this."""
        add_judge_args(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Judge":
        """Create a Judge instance from parsed CLI args."""
        config = JudgeConfig(
            model_name=args.model_name,
            api_url=args.api_url,
            input_path=args.input_path,
            output_path=args.output_path,
            concurrency_limit=args.concurrency_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return cls(config)

    @classmethod
    def cli(cls):
        """Run as CLI entrypoint."""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        judge = cls.from_args(args)
        asyncio.run(judge.run())
