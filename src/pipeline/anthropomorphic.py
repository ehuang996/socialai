"""Anthropomorphic behavior judge: detects whether an AI assistant exhibits anthropomorphic behavior."""

import argparse
import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI

from pipeline.judge import Judge, JudgeConfig

PROMPT_FILE = str(Path(__file__).parent / "prompts" / "judge_prompt_anthropomorphism.txt")

class AnthropomorphicJudge(Judge):

    def __init__(self, config: JudgeConfig, all_turns: bool = False):
        super().__init__(config)
        self.all_turns = all_turns

    def system_prompt(self) -> str:
        with open(PROMPT_FILE, "r") as f:
            return f.read()

    def format_conversation(self, row: dict) -> str | None:
        conversation = row.get("conversation", [])
        if len(conversation) < 2:
            return None

        if self.all_turns:
            parts = []
            for turn in conversation:
                role = turn["role"].upper()
                parts.append(f"{role}: {turn['content']}")
            text = "\n".join(parts)
        else:
            # First turn only: user message + assistant response
            user_msg = conversation[0].get("content", "")
            asst_msg = conversation[1].get("content", "")

            # Skip if too long (matching old behavior - skip entirely, don't truncate)
            total_len = len(user_msg or "") + len(asst_msg or "")
            if total_len > 24000:
                return None

            text = f"USER: {user_msg}\nASSISTANT: {asst_msg}"

        return text

    def judge_type(self) -> str:
        return "anthropomorphic"

    def load_completed_ids(self) -> set[str]:
        """Load hashes already in output file for resumption (old format uses 'hash' not 'conversation_hash')."""
        import os
        completed = set()
        if not os.path.exists(self.config.output_path):
            return completed
        with open(self.config.output_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["hash"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed

    async def process_row(self, row: dict, client: AsyncOpenAI, sem: asyncio.Semaphore, f_out):
        """Process a single row: format → API call → parse → write (OLD FORMAT)."""
        async with sem:
            conversation = row.get("conversation", [])
            if len(conversation) < 2:
                return

            user_msg = conversation[0].get("content", "")
            asst_msg = conversation[1].get("content", "")

            # Skip if too long (matching old behavior)
            total_len = len(user_msg or "") + len(asst_msg or "")
            if total_len > 24000:
                return

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

            # Match old output format: hash, user_input, assistant_response, timestamp, ...parsed_fields
            result = {
                "hash": row["conversation_hash"],
                "user_input": user_msg,
                "assistant_response": asst_msg,
                "timestamp": str(row.get("timestamp", "")),
                **parsed
            }

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--all_turns", action="store_true", help="Send full conversation instead of first turn only")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AnthropomorphicJudge":
        config = JudgeConfig(
            model_name=args.model_name,
            api_url=args.api_url,
            input_path=args.input_path,
            output_path=args.output_path,
            concurrency_limit=args.concurrency_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return cls(config, all_turns=args.all_turns)


if __name__ == "__main__":
    AnthropomorphicJudge.cli()
