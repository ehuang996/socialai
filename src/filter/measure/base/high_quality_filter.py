"""Generic high-quality filter (Stage 3) — reads config from <measure_dir>/filter3.json.

Routes through OpenRouter using model IDs from filter3.json. If filter3.json has a
system_prompt, it is used; otherwise the prompt from filter2.json is reused so that
the same evaluation is repeated with a stronger model.

Input/output: JSONL with consistent column names (conversation_hash, user_input,
assistant_response, etc.) matching Stage 2 output.

Caching: API calls go through dspy.LM with disk caching at cache/dspy/. Re-running
with the same (model, system_prompt, conversation, temperature) returns the cached
verdict, so resumption + cache reuse compound (output-file resumption catches whole-
row retries; DSPy cache catches per-call retries within a row).
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

# Point DSPy disk cache at the project-local cache/dspy/ directory.
# Must be set BEFORE `import dspy` since dspy reads the env var at import time.
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")

import dspy
from tqdm.asyncio import tqdm_asyncio
from utils import Judge, JudgeConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class HighQualityFilterJudge(Judge):

    def __init__(self, config: JudgeConfig, measure_dir: str = "", key_path: str = ""):
        super().__init__(config)
        self._measure_dir = Path(measure_dir)

        # Load filter3 config
        with open(self._measure_dir / "filter3.json") as f:
            self._filter3 = json.load(f)

        self._models = self._filter3["models"]

        # Use filter3 system_prompt if non-empty, otherwise fall back to filter2 prompt
        self._system_prompt = self._filter3.get("system_prompt", "").strip()
        if not self._system_prompt:
            with open(self._measure_dir / "filter2.json") as f:
                self._system_prompt = json.load(f)["prompt"]

        # Load OpenRouter API key
        if key_path:
            with open(key_path) as f:
                self._api_key = f.read().strip()
        else:
            self._api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # Build one DSPy LM per model in filter3.json — each carries its own
        # disk-cache namespace via the (model, sampling_kwargs, messages) key.
        self._lms = {
            col_name: dspy.LM(
                model=f"openrouter/{model_id}",
                api_key=self._api_key,
                api_base=OPENROUTER_BASE_URL,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                cache=True,
                timeout=120,
            )
            for col_name, model_id in self._models.items()
        }

    def system_prompt(self) -> str:
        return self._system_prompt

    def format_conversation(self, row: dict) -> str | None:
        user_input = row.get("user_input", "")
        assistant_response = row.get("assistant_response", "")
        if not user_input:
            return None
        return f"USER: {user_input}\nASSISTANT: {assistant_response}"

    def judge_type(self) -> str:
        return self._measure_dir.name

    async def _call_lm(self, lm: dspy.LM, messages: list[dict]) -> str:
        """Invoke a DSPy LM from async context via to_thread (DSPy is sync)."""
        response = await asyncio.to_thread(lm, messages=messages)
        if not response or not response[0]:
            raise ValueError("empty response")
        raw = response[0]
        if isinstance(raw, dict):
            raw = raw.get("content", "") or raw.get("text", "") or str(raw)
        return raw.strip()

    async def process_row(self, row: dict, sem: asyncio.Semaphore, f_out):
        """Query each model in filter3.json and write a single JSONL row with all responses."""
        async with sem:
            user_content = self.format_conversation(row)
            if user_content is None:
                return

            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": user_content},
            ]

            model_results = {}
            for col_name in self._models:
                try:
                    raw = await self._call_lm(self._lms[col_name], messages)
                    parsed = self.parse_response(raw)
                    model_results[col_name] = {
                        "raw_response": raw,
                        **parsed,
                    }
                except Exception as e:
                    model_results[col_name] = {"error": str(e)}

            result = {
                "conversation_hash": row["conversation_hash"],
                "user_input": row.get("user_input", ""),
                "assistant_response": row.get("assistant_response", ""),
                "timestamp": row.get("timestamp", ""),
                "model_responses": model_results,
            }
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    async def run(self):
        """Override base run() to use OpenRouter via DSPy (cached) instead of local vLLM.

        Slice this shard's deterministic subset BEFORE filtering by keep / completed,
        otherwise resumed runs reprocess rows that belong to other shards' slices
        (see Judge.run docstring).
        """
        sem = asyncio.Semaphore(self.config.concurrency_limit)
        completed = self.load_completed_ids()

        all_rows = []
        with open(self.config.input_path) as f:
            for line in f:
                all_rows.append(json.loads(line))

        if self.config.num_shards > 1:
            shard_rows = all_rows[self.config.shard_id :: self.config.num_shards]
        else:
            shard_rows = all_rows

        rows = [
            r for r in shard_rows
            if r.get("keep", False) and r["conversation_hash"] not in completed
        ]

        print(f"[Shard {self.config.shard_id}] Processing {len(rows)} rows "
              f"(slice size {len(shard_rows)}, skipped {len(completed)} completed)")
        print(f"Models: {self._models}")

        with open(self.config.output_path, "a") as f_out:
            tasks = [self.process_row(row, sem, f_out) for row in rows]
            await tqdm_asyncio.gather(*tasks)

        print(f"[Shard {self.config.shard_id}] Done.")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--measure_dir", type=str, default="",
                            help="Path to the measure folder containing filter3.json")
        parser.add_argument("--key", type=str, default="",
                            help="Path to file containing OpenRouter API key")
        # Lower default concurrency for paid API
        for action in parser._actions:
            if action.dest == "concurrency_limit":
                action.default = 10

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HighQualityFilterJudge":
        config = JudgeConfig(
            model_name="openrouter",  # not used directly; models come from filter3.json
            api_url=OPENROUTER_BASE_URL,
            input_path=args.input_path,
            output_path=args.output_path,
            concurrency_limit=args.concurrency_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return cls(config, measure_dir=args.measure_dir, key_path=args.key)


def cli():
    HighQualityFilterJudge.cli()


if __name__ == "__main__":
    cli()
