"""Generic coarse filter (Stage 1) — reads prompts from <measure_dir>/filter1.json.

Used as a fallback when a measure folder does not provide its own coarse_filter.py.
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from utils import Judge, JudgeConfig


class CoarseFilterJudge(Judge):

    def __init__(self, config: JudgeConfig, prompt_version: str = "v1", measure_dir: str = ""):
        super().__init__(config)
        prompts_path = Path(measure_dir) / "filter1.json"
        with open(prompts_path) as f:
            self.prompts = json.load(f)
        self.prompt_version = prompt_version
        if self.prompt_version not in self.prompts:
            raise ValueError(f"Prompt version '{self.prompt_version}' not found. Available: {list(self.prompts.keys())}")

    def system_prompt(self) -> str:
        return self.prompts[self.prompt_version]

    def format_conversation(self, row: dict) -> str | None:
        conversation = row.get("conversation", [])
        if not conversation:
            return None
        user_msg = conversation[0].get("content", "").strip()
        if not user_msg or len(user_msg) < 5:
            return None
        return user_msg[:4000]

    def judge_type(self) -> str:
        return "coarse_filter"

    async def process_row(self, row: dict, client: AsyncOpenAI, sem: asyncio.Semaphore, f_out):
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
            if parsed.get("keep"):
                conversation = row.get("conversation", [])
                user_msg = conversation[0].get("content", "") if len(conversation) > 0 else ""
                asst_msg = conversation[1].get("content", "") if len(conversation) > 1 else ""
                result = {
                    "conversation_hash": row["conversation_hash"],
                    "user_input": user_msg,
                    "assistant_response": asst_msg,
                    "timestamp": str(row.get("timestamp", "")),
                    "filter_reasoning": parsed.get("reasoning", ""),
                }
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        for action in parser._actions:
            if action.dest == "concurrency_limit":
                action.default = 100
        parser.add_argument("--prompt-version", type=str, default="v1",
                            help="Version of the prompt to use (e.g., v1, v5)")
        parser.add_argument("--measure_dir", type=str, default="",
                            help="Path to the measure folder containing filter1.json")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CoarseFilterJudge":
        config = JudgeConfig(
            model_name=args.model_name,
            api_url=args.api_url,
            input_path=args.input_path,
            output_path=args.output_path,
            concurrency_limit=args.concurrency_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return cls(config, prompt_version=args.prompt_version, measure_dir=args.measure_dir)


def cli():
    CoarseFilterJudge.cli()


if __name__ == "__main__":
    cli()
