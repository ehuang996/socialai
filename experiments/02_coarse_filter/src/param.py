"""Shared argparse helpers for judge CLI entrypoints."""

import argparse
import os


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    """Add standard judge CLI arguments to an existing parser."""
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--api_url",
        type=str,
        default=f"http://localhost:{os.getenv('VLLM_PORT', '8001')}/v1",
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--concurrency_limit", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
