"""Stage 7.1 — Generate API/direct model responses for the evaluation set.

Sends each `user_input` from `data/final_dataset.jsonl` (969 rows) to 23
distinct model evaluations spanning 6 providers (Google, OpenAI, Anthropic,
xAI, DeepSeek, Qwen), including 3 different thinking-budget variants of
claude-opus-4.6.

Total scale: 23 × 969 = 22,287 generations.

Two additional Qwen columns (`qwen3_1_7b`, `qwen3_4b`) are generated locally
with vLLM via generate_local_responses.py, then merged into this output for a
combined 25-model Stage 7.1 file.

Caching:
  - DSPy disk cache (cache/dspy/) makes re-runs free. Cache key includes the
    full request payload, so the three claude-opus-4.6 thinking-budget variants
    each get their own cache entry automatically (different `extra_body`).
  - File-based resumption: rows already in the output JSONL are skipped before
    any LM call is even constructed. Partially-completed rows (some models
    missing) are filled in by re-running only the missing columns.
  - --rollout_id forces fresh generations while still writing them into cache.

Keys:
  Reads `.keys.json` (default path) with sections:
      {"openrouter": "...", "deepseek": "...", "anthropic": "..."}
  OpenRouter is required for every model except the three deepseek_* variants
  (which use the direct DeepSeek API).
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

# Point DSPy disk cache at the project-local cache/dspy/ directory.
# Must be set BEFORE `import dspy` since dspy reads the env var at import time.
os.environ.setdefault("DSPY_CACHEDIR", "cache/dspy")

import dspy
from tqdm import tqdm

from src.evaluation._eval_common import load_keys

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class ModelConfig:
    """One row of the evaluation matrix.

    The dict KEY in MODEL_REGISTRY is the column name used in the output
    JSONL (`model_responses[col]`). It must be unique and snake_case.
    """
    slug: str                              # provider-specific model identifier
    provider: str                          # "openrouter" | "deepseek_direct"
    category: str                          # "target" | "rewriting" | "frontier"
    thinking: bool = False                 # extended thinking enabled?
    thinking_budget: Optional[int] = None  # tokens; required when thinking=True (Anthropic)
    extra_body: dict = field(default_factory=dict)  # provider-specific extras
    max_tokens: int = 2048
    temperature: float = 0.7


# 23 API/direct model evaluations.
# NOTE: qwen3_0_6b is still excluded. qwen3_1_7b and qwen3_4b are excluded
# from this API registry because OpenRouter currently lists those model IDs
# with 0 active endpoints; they are handled by the local-HF add-on instead.
# Thinking-budget variants of the same base model use distinct column keys so
# they each get their own cache entry and downstream judge/analysis column.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ─── Target private models (3) ───────────────────────────────────────
    "gemini_2_flash_001": ModelConfig(
        slug="google/gemini-2.0-flash-001",
        provider="openrouter", category="target",
    ),
    "gpt_4o": ModelConfig(
        slug="openai/gpt-4o",
        provider="openrouter", category="target",
    ),
    "claude_sonnet_4": ModelConfig(
        slug="anthropic/claude-sonnet-4",
        provider="openrouter", category="target",
    ),

    # ─── Rewriting private models (3) ────────────────────────────────────
    "gemini_3_1_pro": ModelConfig(
        slug="google/gemini-3.1-pro-preview",
        provider="openrouter", category="rewriting",
        thinking=True,
        # NOTE: [edge case callout] Gemini extended thinking uses
        # extra_body["thinkingConfig"]["thinkingBudget"]. Set the actual budget
        # once the canonical slug is confirmed.
        extra_body={"thinkingConfig": {"thinkingBudget": -1}},
    ),
    "gpt_5_4": ModelConfig(
        slug="openai/gpt-5.4",
        provider="openrouter", category="rewriting",
    ),
    "claude_opus_4_6": ModelConfig(
        slug="anthropic/claude-opus-4.6",
        provider="openrouter", category="rewriting",
    ),

    # ─── Frontier private models (17) ────────────────────────────────────
    "gemini_3_flash": ModelConfig(
        slug="google/gemini-3-flash-preview",
        provider="openrouter", category="frontier",
    ),
    "gpt_5_5": ModelConfig(
        slug="openai/gpt-5.5",
        provider="openrouter", category="frontier",
        thinking=True,
        extra_body={"reasoning": {"effort": "medium"}},
    ),
    "claude_opus_4_7": ModelConfig(
        slug="anthropic/claude-opus-4.7",
        provider="openrouter", category="frontier",
    ),
    # Three thinking-budget variants of claude-opus-4.6.
    # NOTE: [thought process] Anthropic requires temperature=1.0 and
    # max_tokens >= budget + ~1024 when extended thinking is on.
    "claude_opus_4_6_t2k": ModelConfig(
        slug="anthropic/claude-opus-4.6",
        provider="openrouter", category="frontier",
        thinking=True, thinking_budget=2000,
        max_tokens=2000 + 2048, temperature=1.0,
    ),
    "claude_opus_4_6_t5k": ModelConfig(
        slug="anthropic/claude-opus-4.6",
        provider="openrouter", category="frontier",
        thinking=True, thinking_budget=5000,
        max_tokens=5000 + 2048, temperature=1.0,
    ),
    "claude_opus_4_6_t10k": ModelConfig(
        slug="anthropic/claude-opus-4.6",
        provider="openrouter", category="frontier",
        thinking=True, thinking_budget=10000,
        max_tokens=10000 + 2048, temperature=1.0,
    ),
    "grok_3": ModelConfig(
        slug="x-ai/grok-3",
        provider="openrouter", category="frontier",
    ),
    "grok_4": ModelConfig(
        slug="x-ai/grok-4",
        provider="openrouter", category="frontier",
        thinking=True,
        extra_body={"reasoning": {"effort": "low"}},
    ),
    "grok_4_3": ModelConfig(
        slug="x-ai/grok-4.3",
        provider="openrouter", category="frontier",
        thinking=True,
        extra_body={"reasoning": {"effort": "low"}},
    ),
    "gpt_4o_mini": ModelConfig(
        slug="openai/gpt-4o-mini",
        provider="openrouter", category="frontier",
    ),
    # ── DeepSeek (direct API for all three) ──
    "deepseek_chat": ModelConfig(
        slug="deepseek-chat",
        provider="deepseek_direct", category="frontier",
    ),
    "deepseek_v4_pro": ModelConfig(
        slug="deepseek-v4-pro",
        provider="deepseek_direct", category="frontier",
        thinking=True,
    ),
    "deepseek_v4_flash": ModelConfig(
        slug="deepseek-v4-flash",
        provider="deepseek_direct", category="frontier",
    ),
    # ── Qwen (non-thinking dense models force enable_thinking=False; max-preview is thinking-on) ──
    "qwen3_6_max_preview": ModelConfig(
        slug="qwen/qwen3.6-max-preview",
        provider="openrouter", category="frontier",
        thinking=True,
        extra_body={"enable_thinking": True},
    ),
    "qwen3_8b": ModelConfig(
        slug="qwen/qwen3-8b",
        provider="openrouter", category="frontier",
        extra_body={"enable_thinking": False},
    ),
    "qwen3_14b": ModelConfig(
        slug="qwen/qwen3-14b",
        provider="openrouter", category="frontier",
        extra_body={"enable_thinking": False},
    ),
    "qwen3_32b": ModelConfig(
        slug="qwen/qwen3-32b",
        provider="openrouter", category="frontier",
        extra_body={"enable_thinking": False},
    ),
}


def validate_registry(registry: dict[str, ModelConfig]) -> None:
    """Hard-fail on any Anthropic-thinking entry missing a thinking_budget."""
    for k, c in registry.items():
        if c.thinking and c.slug.startswith("anthropic/") and c.thinking_budget is None:
            raise SystemExit(
                f"{k}: Anthropic thinking is enabled but thinking_budget is None"
            )


def required_keys_for(registry: dict[str, ModelConfig]) -> set[str]:
    """Return the set of key sections needed by the active registry."""
    needed = set()
    for c in registry.values():
        if c.provider == "openrouter":
            needed.add("openrouter")
        elif c.provider == "deepseek_direct":
            needed.add("deepseek")
    return needed


def create_lm(cfg: ModelConfig, keys: dict, timeout: int = 180) -> dspy.LM:
    """Build a DSPy LM instance for one MODEL_REGISTRY entry.

    Caching is always on — DSPy hashes the full request payload (model string,
    messages, extra_body, temperature, max_tokens) so different thinking-budget
    variants of the same base model land in distinct cache entries.
    """
    if cfg.provider == "deepseek_direct":
        return dspy.LM(
            model=f"deepseek/{cfg.slug}",
            api_key=keys["deepseek"],
            api_base=DEEPSEEK_BASE_URL,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            cache=True,
            timeout=timeout,
        )
    # Default: OpenRouter
    return dspy.LM(
        model=f"openrouter/{cfg.slug}",
        api_key=keys["openrouter"],
        api_base=OPENROUTER_BASE_URL,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        cache=True,
        timeout=timeout,
    )


def build_extra_body(cfg: ModelConfig) -> dict:
    """Compose the per-call extra_body, merging thinking config + provider extras."""
    body = dict(cfg.extra_body)  # shallow copy so we don't mutate the registry
    if cfg.thinking and cfg.slug.startswith("anthropic/"):
        # Anthropic extended thinking schema
        body["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
    return body


def call_model(lm: dspy.LM, cfg: ModelConfig, user_input: str,
               rollout_id: Optional[int]) -> str:
    """Single model call, returns the assistant text."""
    messages = [{"role": "user", "content": user_input}]
    config: dict = {}
    if rollout_id is not None:
        config["rollout_id"] = rollout_id
    extra = build_extra_body(cfg)
    if extra:
        config["extra_body"] = extra

    response = lm(messages=messages, **config)
    if not response or not response[0]:
        raise ValueError("Model returned empty response")
    result = response[0]
    if isinstance(result, dict):
        result = result.get("content", "") or result.get("text", "") or str(result)
    return result.strip()


def _call_with_retry(
    col_name: str, cfg: ModelConfig, lm: dspy.LM,
    user_input: str, rollout_id: Optional[int], max_retries: int = 3,
) -> tuple[str, dict]:
    """Call a single model with retries. Returns (col_name, result_dict)."""
    for attempt in range(max_retries):
        try:
            raw = call_model(lm, cfg, user_input, rollout_id)
            return col_name, {"assistant_response": raw}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return col_name, {"error": str(e)}
    return col_name, {"error": "unreachable"}


def process_row(row: dict, lms: dict, configs: dict[str, ModelConfig],
                rollout_id: Optional[int]) -> Optional[dict]:
    """Send one user_input to all active models in parallel; return the result row."""
    user_input = row.get("user_input", "")
    if not user_input:
        return None

    model_results = {}
    with ThreadPoolExecutor(max_workers=max(1, len(configs))) as executor:
        futures = {
            executor.submit(
                _call_with_retry, col, cfg, lms[col], user_input, rollout_id
            ): col
            for col, cfg in configs.items()
        }
        for future in as_completed(futures):
            col, result = future.result()
            model_results[col] = result

    return {
        "user_input": user_input,
        "measure": row.get("measure", []),
        "synthetic": row.get("synthetic", False),
        "language": row.get("language", "English"),
        "model_responses": model_results,
    }


def run(args):
    keys = load_keys(args.keys)

    # Subset filter — useful for tiny dry runs (e.g. --models gpt_4o,deepseek_chat)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        unknown = wanted - set(MODEL_REGISTRY)
        if unknown:
            raise SystemExit(f"Unknown model column(s): {sorted(unknown)}")
        active = {k: MODEL_REGISTRY[k] for k in MODEL_REGISTRY if k in wanted}
    else:
        active = dict(MODEL_REGISTRY)

    validate_registry(active)

    # Hard-fail if any required key section is empty
    needed = required_keys_for(active)
    missing = [s for s in needed if not keys.get(s)]
    if missing:
        raise SystemExit(
            f"Missing key sections in {args.keys}: {missing} "
            f"(needed for active model set)"
        )

    # Build one DSPy LM per column (caching is per-LM-instance)
    lms = {col: create_lm(cfg, keys) for col, cfg in active.items()}

    # Resume: load already-completed rows by user_input
    completed_rows: dict[str, dict] = {}
    output_path = args.output
    try:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                completed_rows[r["user_input"]] = r
    except FileNotFoundError:
        pass

    # Split input into "new" and "partial" (some models missing)
    rows_new = []
    rows_partial = []  # (input_row, existing_row, missing_cols)
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            ui = row["user_input"]
            if ui in completed_rows:
                existing = completed_rows[ui]
                have = set(existing.get("model_responses", {}).keys())
                missing = set(active) - have
                if missing:
                    rows_partial.append((row, existing, missing))
            else:
                rows_new.append(row)

    print(f"Models active ({len(active)}): {list(active)}")
    print(f"New rows: {len(rows_new)} | Partial rows: {len(rows_partial)} "
          f"| Fully complete (skipped): {len(completed_rows) - len(rows_partial)}")
    print(f"DSPy caching: enabled (rollout_id={args.rollout_id})")

    # New rows — append to the file as we go
    with open(output_path, "a") as f_out:
        for row in tqdm(rows_new, desc="New rows"):
            result = process_row(row, lms, active, args.rollout_id)
            if result:
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                completed_rows[result["user_input"]] = result

    # Partial rows — re-run only the missing columns, then rewrite the whole file
    if rows_partial:
        print(f"Filling {len(rows_partial)} partial rows...")
        for input_row, existing, missing in tqdm(rows_partial, desc="Filling"):
            sub_cfgs = {k: active[k] for k in missing}
            sub_lms = {k: lms[k] for k in missing}
            partial = process_row(input_row, sub_lms, sub_cfgs, args.rollout_id)
            if partial:
                existing["model_responses"].update(partial["model_responses"])

        # Rewrite output file with the merged rows
        with open(output_path, "w") as f_out:
            for r in completed_rows.values():
                f_out.write(json.dumps(r) + "\n")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7.1 — Generate API/direct model responses for the evaluation set."
    )
    parser.add_argument(
        "--input",
        default="data/final_dataset.jsonl",
        help="Input JSONL (default: data/final_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/eval_responses.jsonl",
        help="Output JSONL (default: data/eval_responses.jsonl)",
    )
    parser.add_argument(
        "--keys", default=".keys.json",
        help="Path to .keys.json with {openrouter, deepseek, anthropic} sections",
    )
    parser.add_argument(
        "--models", default="",
        help="Comma-separated subset of MODEL_REGISTRY column names (default: all 23 API/direct models)",
    )
    parser.add_argument(
        "--rollout_id", type=int, default=None,
        help="DSPy rollout ID — bypass cache and generate fresh (default: use cache)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
