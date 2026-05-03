"""Shared constants and helpers for the Stage 7 evaluation pipeline.

Used by:
  - generate_responses.py (Stage 7.1)
  - judge_responses.py    (Stage 7.2)
  - score_seedset.py      (one-shot seedset scorer)

Single source of truth for:
  - The 9 active measures (matches `src/filter/measure/<name>/` folders)
  - Loading the per-measure judge prompt from `filter2.json`
  - Parsing judge JSON output (tolerant to fenced code blocks etc.)
  - Loading API keys from `.keys.json`, with legacy plaintext fallback files
"""

import json
import re
from pathlib import Path

# NOTE: [thought process] keep this list in sync with `src/filter/measure/`.
# generate_responses.py / judge_responses.py / score_seedset.py all import this constant
# so a single edit here propagates everywhere.
ALL_MEASURES = [
    "1B_intentional_human_speech",
    "1B_human_pronoun",
    "1C_identity_transparency",
    "2A_fabricated_personal_information",
    "2B_emotion_expression",
    "2C_deference",
    "2C_flattery_tone",
    "2D_human_relationship_encouragement",
    "3A_engagement_hooks",
]

DEFAULT_MEASURE_DIR = Path("src/filter/measure")


def load_judge_prompts(
    measures: list[str] = ALL_MEASURES,
    measure_dir: Path = DEFAULT_MEASURE_DIR,
) -> dict[str, str]:
    """Load the `prompt` field of each measure's filter2.json (the Stage-2 rubric)."""
    out = {}
    for m in measures:
        path = Path(measure_dir) / m / "filter2.json"
        with open(path) as f:
            out[m] = json.load(f)["prompt"]
    return out


def extract_text(response) -> str:
    """Pull the assistant text out of a DSPy LM response.

    DSPy returns a list whose first element is either a string or a dict with
    'content'/'text' keys (reasoning models can produce dicts).
    """
    if not response or not response[0]:
        raise ValueError("empty response")
    raw = response[0]
    if isinstance(raw, dict):
        raw = raw.get("content", "") or raw.get("text", "") or str(raw)
    return raw.strip()


class ParseError(Exception):
    """Raised when the judge response is not valid JSON we can recover from."""


def parse_judge_response(raw: str) -> dict:
    """Parse {reasoning, keep} JSON from a judge response.

    Handles bare JSON, fenced code blocks, nested objects, and a final
    regex fallback that just grabs the keep boolean.
    """
    text = raw.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    if "```" in text:
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1).strip())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

    # Single-level brace match
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # One-level nested brace match (handles {"reasoning": "...", "keep": ...} with quoted braces inside)
    nested_match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", text, re.DOTALL)
    if nested_match:
        try:
            data = json.loads(nested_match.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Last-ditch: regex out the keep flag and reasoning string
    keep_match = re.search(r'"keep"\s*:\s*(true|false)', text, re.IGNORECASE)
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"(.*?)"\s*,\s*"keep"', text, re.DOTALL
    )
    if keep_match:
        return {
            "reasoning": reasoning_match.group(1) if reasoning_match else "",
            "keep": keep_match.group(1).lower() == "true",
        }

    raise ParseError(f"Could not parse JSON from response: {raw[:200]}")


def normalize_keep(parsed: dict) -> dict:
    """Coerce `keep` to a real bool (Opus sometimes emits the string "true")."""
    if "error" in parsed or "keep" not in parsed:
        return parsed
    val = parsed["keep"]
    if isinstance(val, str):
        parsed["keep"] = val.strip().lower() in ("true", "yes", "1")
    elif not isinstance(val, bool):
        parsed["keep"] = bool(val)
    return parsed


def _read_plaintext_key(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text().strip()


def load_keys(path: str = ".keys.json") -> dict:
    """Load API keys.

    Preferred schema is the unified JSON file:
        {"openrouter": "sk-...", "deepseek": "sk-...", "anthropic": "sk-..."}

    For compatibility with existing filtering scripts, the default `.keys.json`
    path falls back to legacy plaintext files when absent:
    `.openrouter_key`, `.deepseek_key`, and `.anthropic_key`.

    Missing sections are returned as empty strings — callers decide which keys
    are required for their model set and fail loudly if a needed one is blank.
    """
    p = Path(path)
    if not p.exists():
        if path == ".keys.json" and Path(".openrouter_key").exists():
            return {
                "openrouter": _read_plaintext_key(".openrouter_key"),
                "deepseek":   _read_plaintext_key(".deepseek_key"),
                "anthropic":  _read_plaintext_key(".anthropic_key"),
            }
        raise FileNotFoundError(
            f"Key file {path} not found. Create it with sections "
            "{openrouter, deepseek, anthropic}, or use legacy .openrouter_key."
        )

    if p.name.startswith(".") and p.suffix != ".json":
        return {
            "openrouter": p.read_text().strip(),
            "deepseek":   _read_plaintext_key(".deepseek_key"),
            "anthropic":  _read_plaintext_key(".anthropic_key"),
        }

    data = json.loads(p.read_text())
    return {
        "openrouter": data.get("openrouter", "").strip(),
        "deepseek":   data.get("deepseek", "").strip(),
        "anthropic":  data.get("anthropic", "").strip(),
    }
