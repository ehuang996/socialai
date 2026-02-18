"""Chit-chat filter judge: classifies whether a conversation is casual chit-chat."""

import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI
from pipeline.judge import Judge


def load_prompts(prompts_file: str = "prompts/chit_chat_prompts.json") -> dict:
    """Load versioned prompts from JSON file."""
    prompts_path = Path(__file__).parent / prompts_file
    with open(prompts_path, 'r') as f:
        return json.load(f)


class ChitChatJudge(Judge):

    def __init__(self, *args, prompt_version: str = "v1", **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = load_prompts()
        self.prompt_version = prompt_version
        if self.prompt_version not in self.prompts:
            raise ValueError(f"Prompt version '{self.prompt_version}' not found. Available: {list(self.prompts.keys())}")

    def system_prompt(self) -> str:
        return self.prompts[self.prompt_version]

    def format_conversation(self, row: dict) -> str | None:
        """Format only the user message (first turn) for filtering."""
        conversation = row.get("conversation", [])
        if not conversation:
            return None
        user_msg = conversation[0].get("content", "").strip()
        if not user_msg or len(user_msg) < 5:
            return None
        # Truncate to 4k characters
        if len(user_msg) > 4000:
            user_msg = user_msg[:4000]
        return user_msg

    def judge_type(self) -> str:
        return "chit_chat"

    async def process_row(self, row: dict, client: AsyncOpenAI, sem: asyncio.Semaphore, f_out):
        """Process a single row: format → API call → parse → write (OLD FORMAT)."""
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

            # Only save if keep=true (matching old behavior)
            if parsed.get("keep"):
                row['filter_reasoning'] = parsed.get('reasoning', '')
                f_out.write(json.dumps(row, default=str) + "\n")
                f_out.flush()

    @classmethod
    def add_args(cls, parser):
        """Override to set concurrency_limit default to 100 (matching old pipeline)."""
        super().add_args(parser)
        # Update the default for concurrency_limit
        for action in parser._actions:
            if action.dest == 'concurrency_limit':
                action.default = 100
        # Add prompt version argument
        parser.add_argument(
            '--prompt-version',
            type=str,
            default='v1',
            help='Version of the prompt to use (e.g., v1, v2)',
        )


if __name__ == "__main__":
    ChitChatJudge.cli()
