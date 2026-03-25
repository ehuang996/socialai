"""Generic low-quality filter — reads system prompt from <measure_dir>/filter2.json.

Used as a fallback when a measure folder does not provide its own low_quality_filter.py.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError
from utils import Judge, JudgeConfig


class LowQualityFilterJudge(Judge):

    def __init__(self, config: JudgeConfig, measure_dir: str = ""):
        super().__init__(config)
        self._prompt_file = Path(measure_dir) / "filter2.json"

    def system_prompt(self) -> str:
        with open(self._prompt_file) as f:
            return json.load(f)["prompt"]

    def format_conversation(self, row: dict) -> str | None:
        user_msg = row.get("user_input", "")
        asst_msg = row.get("assistant_response", "")
        if not user_msg:
            return None
        if len(user_msg) + len(asst_msg) > 24000:
            return None
        return f"USER: {user_msg}\nASSISTANT: {asst_msg}"

    def judge_type(self) -> str:
        # Use measure folder name as the score prefix (e.g., "sycophancy_score")
        return Path(self._prompt_file).parent.name

    def load_completed_ids(self) -> set[str]:
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
        async with sem:
            user_content = self.format_conversation(row)
            if user_content is None:
                return
            try:
                response = await client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt()},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except BadRequestError as e:
                print(f"[skip] {row.get('conversation_hash', '?')}: {e}", flush=True)
                return
            raw = response.choices[0].message.content.strip()
            parsed = self.parse_response(raw)
            result = {
                "conversation_hash": row["conversation_hash"],
                "user_input": row.get("user_input", ""),
                "assistant_response": row.get("assistant_response", ""),
                "timestamp": row.get("timestamp", ""),
                **parsed,
            }
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--measure_dir", type=str, default="",
                            help="Path to the measure folder containing filter2.json")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LowQualityFilterJudge":
        config = JudgeConfig(
            model_name=args.model_name,
            api_url=args.api_url,
            input_path=args.input_path,
            output_path=args.output_path,
            concurrency_limit=args.concurrency_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return cls(config, measure_dir=args.measure_dir)


def cli():
    LowQualityFilterJudge.cli()


if __name__ == "__main__":
    cli()
