"""Stage 7.3 - Analysis outputs for the Stage 7 evaluation.

This script produces figure/table artifacts for the planned paper analyses.
It intentionally writes evidence for paragraphs, not the paragraphs
themselves.

Default inputs:
  - data/eval_judge_results.jsonl      Stage 7.2 judge output
  - data/final_dataset.jsonl           input labels + synthetic flag
  - data/synthetic_data_relabelled.jsonl synthetic provenance
  - data/synthetic_data_full.jsonl     accepted synthetic-generation audit
  - data/synthetic/*.rejected.jsonl    rejected synthetic-generation audit
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation._eval_common import ALL_MEASURES


# ---------------------------------------------------------------------------
# Constants

MEASURE_ORDER = ALL_MEASURES

MEASURE_SHORT = {
    "1B_intentional_human_speech": "1B Speech",
    "1B_human_pronoun": "1B Pronoun",
    "1C_identity_transparency": "1C Identity",
    "2A_fabricated_personal_information": "2A Fabrication",
    "2B_emotion_expression": "2B Emotion",
    "2C_deference": "2C Deference",
    "2C_flattery_tone": "2C Flattery",
    "2D_human_relationship_encouragement": "2D Relationship",
    "3A_engagement_hooks": "3A Engagement",
}

FAMILY_ORDER = ["OpenAI", "Google", "Anthropic", "xAI", "DeepSeek", "Qwen"]

# Newest/largest first, matching Stage 7.1 column keys after the local-HF
# qwen3_1_7b/qwen3_4b add-on is merged into the API/direct output.
MODEL_FAMILIES = {
    "OpenAI": ["gpt_5_5", "gpt_5_4", "gpt_4o", "gpt_4o_mini"],
    "Google": ["gemini_3_1_pro", "gemini_3_flash", "gemini_2_flash_001"],
    "Anthropic": [
        "claude_opus_4_7",
        "claude_opus_4_6",
        "claude_opus_4_6_t2k",
        "claude_opus_4_6_t5k",
        "claude_opus_4_6_t10k",
        "claude_sonnet_4",
    ],
    "xAI": ["grok_4_3", "grok_4", "grok_3"],
    "DeepSeek": ["deepseek_v4_pro", "deepseek_v4_flash", "deepseek_chat"],
    "Qwen": [
        "qwen3_6_max_preview",
        "qwen3_32b",
        "qwen3_14b",
        "qwen3_8b",
        "qwen3_4b",
        "qwen3_1_7b",
    ],
}

MODEL_ORDER = [m for fam in FAMILY_ORDER for m in MODEL_FAMILIES[fam]]
MODEL_TO_FAMILY = {
    model: family for family, models in MODEL_FAMILIES.items() for model in models
}

MODEL_DISPLAY = {
    "gpt_5_5": "GPT-5.5",
    "gpt_5_4": "GPT-5.4",
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o-mini",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_3_flash": "Gemini 3 Flash",
    "gemini_2_flash_001": "Gemini 2 Flash",
    "claude_opus_4_7": "Claude Opus 4.7",
    "claude_opus_4_6": "Claude Opus 4.6",
    "claude_opus_4_6_t2k": "Claude Opus 4.6 T2k",
    "claude_opus_4_6_t5k": "Claude Opus 4.6 T5k",
    "claude_opus_4_6_t10k": "Claude Opus 4.6 T10k",
    "claude_sonnet_4": "Claude Sonnet 4",
    "grok_4_3": "Grok 4.3",
    "grok_4": "Grok 4",
    "grok_3": "Grok 3",
    "deepseek_v4_pro": "DeepSeek V4 Pro",
    "deepseek_v4_flash": "DeepSeek V4 Flash",
    "deepseek_chat": "DeepSeek Chat",
    "qwen3_6_max_preview": "Qwen 3.6 Max",
    "qwen3_32b": "Qwen3 32B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_4b": "Qwen3 4B",
    "qwen3_1_7b": "Qwen3 1.7B",
}

FAMILY_COLORS = {
    "OpenAI": "#10a37f",
    "Google": "#4285f4",
    "Anthropic": "#d97706",
    "xAI": "#ef4444",
    "DeepSeek": "#7c3aed",
    "Qwen": "#0ea5e9",
}

TARGET_MODELS = {"gemini_2_flash_001", "gpt_4o", "claude_sonnet_4"}
REWRITING_MODELS = {"gemini_3_1_pro", "gpt_5_4", "claude_opus_4_6"}
THINKING_MODELS = {
    "gemini_3_1_pro",
    "gpt_5_5",
    "claude_opus_4_6_t2k",
    "claude_opus_4_6_t5k",
    "claude_opus_4_6_t10k",
    "grok_4",
    "grok_4_3",
    "deepseek_v4_pro",
    "qwen3_6_max_preview",
}
PUBLIC_MODELS = {
    "deepseek_chat",
    "deepseek_v4_pro",
    "deepseek_v4_flash",
    "qwen3_8b",
    "qwen3_4b",
    "qwen3_1_7b",
    "qwen3_14b",
    "qwen3_32b",
}

# Approximate public model scale used only for the public scaling plot/table.
# DeepSeek V4 values are active parameters; Qwen values are dense parameters.
MODEL_SIZE_B = {
    "deepseek_v4_flash": 13.0,
    "deepseek_v4_pro": 49.0,
    "qwen3_1_7b": 1.7,
    "qwen3_4b": 4.0,
    "qwen3_8b": 8.0,
    "qwen3_14b": 14.0,
    "qwen3_32b": 32.0,
}

# Oldest -> newest. Thinking-budget variants are excluded so temporal trends
# do not double-count the same Claude Opus 4.6 generation.
GENERATION_MODEL_ORDER = {
    "OpenAI": ["gpt_4o_mini", "gpt_4o", "gpt_5_4", "gpt_5_5"],
    "Google": ["gemini_2_flash_001", "gemini_3_flash", "gemini_3_1_pro"],
    "Anthropic": ["claude_sonnet_4", "claude_opus_4_6", "claude_opus_4_7"],
    "xAI": ["grok_3", "grok_4", "grok_4_3"],
    "DeepSeek": ["deepseek_chat", "deepseek_v4_flash", "deepseek_v4_pro"],
    "Qwen": [
        "qwen3_1_7b",
        "qwen3_4b",
        "qwen3_8b",
        "qwen3_14b",
        "qwen3_32b",
        "qwen3_6_max_preview",
    ],
}

GENERATION_INDEX = {
    model: idx + 1
    for family, models in GENERATION_MODEL_ORDER.items()
    for idx, model in enumerate(models)
}

CLAUDE_THINKING_ORDER = [
    "claude_opus_4_6",
    "claude_opus_4_6_t2k",
    "claude_opus_4_6_t5k",
    "claude_opus_4_6_t10k",
]

CLAUDE_THINKING_BUDGET = {
    "claude_opus_4_6": 0,
    "claude_opus_4_6_t2k": 2000,
    "claude_opus_4_6_t5k": 5000,
    "claude_opus_4_6_t10k": 10000,
}


# ---------------------------------------------------------------------------
# Small utilities


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_measures(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def model_display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def measure_display(measure: str) -> str:
    return MEASURE_SHORT.get(measure, measure)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  {path.name}")


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def draw_heatmap(
    data: pd.DataFrame,
    ax: plt.Axes,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float = 0,
    vmax: float = 100,
    fmt: str = ".1f",
) -> None:
    """Draw an annotated heatmap with matplotlib only.

    The project environment used on the cluster does not always have seaborn
    installed even though pyproject lists it, so this keeps Stage 7.3 runnable
    with pandas/matplotlib/numpy alone.
    """
    numeric = data.astype(float)
    arr = numeric.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f3f4f6")

    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(numeric.columns)))
    ax.set_yticks(np.arange(len(numeric.index)))
    ax.set_xticklabels(numeric.columns, rotation=30, ha="right")
    ax.set_yticklabels(numeric.index)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)


def rate_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, "violations", "total", "inputs", "violation_rate"]
        )

    table = (
        df.groupby(group_cols, dropna=False)
        .agg(
            violations=("keep", "sum"),
            total=("keep", "size"),
            inputs=("user_input", "nunique"),
        )
        .reset_index()
    )
    table["violation_rate"] = table["violations"] / table["total"]
    table["violation_rate_pct"] = table["violation_rate"] * 100
    return table


def ordered_models_present(df: pd.DataFrame, models: Iterable[str] = MODEL_ORDER) -> list[str]:
    present = set(df["model"].unique())
    ordered = [m for m in models if m in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def add_model_metadata(table: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    table = table.copy()
    table["model_display"] = table[model_col].map(model_display)
    table["family"] = table[model_col].map(MODEL_TO_FAMILY).fillna("Unknown")
    table["category"] = np.where(
        table[model_col].isin(TARGET_MODELS),
        "target",
        np.where(table[model_col].isin(REWRITING_MODELS), "rewriting", "frontier"),
    )
    table["target_group"] = np.where(
        table[model_col].isin(TARGET_MODELS), "target", "non_target"
    )
    table["thinking_group"] = np.where(
        table[model_col].isin(THINKING_MODELS), "thinking", "no_thinking"
    )
    table["access"] = np.where(table[model_col].isin(PUBLIC_MODELS), "public", "private")
    table["size_b"] = table[model_col].map(MODEL_SIZE_B)
    table["generation_index"] = table[model_col].map(GENERATION_INDEX)
    return table


# ---------------------------------------------------------------------------
# Data loading and joins


def load_input_metadata(dataset_path: Path, synthetic_metadata_path: Path) -> pd.DataFrame:
    """Load row-level metadata keyed by user_input.

    `final_dataset.jsonl` gives source type and label list. The synthetic file
    adds rewrite_model / response_model provenance for synthetic rows.
    """
    records: dict[str, dict] = {}

    for row in read_jsonl(dataset_path):
        user_input = row.get("user_input")
        if not user_input:
            continue
        synthetic = bool(row.get("synthetic", False))
        records[user_input] = {
            "user_input": user_input,
            "synthetic": synthetic,
            "source_type": "synthetic_rewrite" if synthetic else "in_the_wild",
            "labeled_measures": normalize_measures(row.get("measure")),
            "language": row.get("language", ""),
        }

    for row in read_jsonl(synthetic_metadata_path):
        user_input = row.get("user_input")
        if not user_input:
            continue
        rec = records.setdefault(
            user_input,
            {
                "user_input": user_input,
                "synthetic": True,
                "source_type": "synthetic_rewrite",
                "labeled_measures": normalize_measures(row.get("measure")),
                "language": row.get("language", ""),
            },
        )
        rec.update(
            {
                "synthetic": True,
                "source_type": "synthetic_rewrite",
                "rewrite_model": row.get("rewrite_model", ""),
                "response_model": row.get("response_model", ""),
                "source_hash": row.get("source_hash", ""),
                "source_input": row.get("source_input", ""),
                "round_idx": row.get("round_idx"),
                "candidate_idx": row.get("candidate_idx"),
                "naturalness_passed": row.get("naturalness_passed"),
                "naturalness_ranking": row.get("naturalness_ranking"),
            }
        )
        if row.get("measure"):
            rec["labeled_measures"] = normalize_measures(row.get("measure"))

    cols = [
        "user_input",
        "synthetic",
        "source_type",
        "labeled_measures",
        "language",
        "rewrite_model",
        "response_model",
        "source_hash",
        "source_input",
        "round_idx",
        "candidate_idx",
        "naturalness_passed",
        "naturalness_ranking",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records.values())
    for col in cols:
        if col not in df:
            df[col] = np.nan
    return df[cols]


def load_judge_data(input_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(input_path):
        judge_output = row.get("judge_output") or {}
        keep = judge_output.get("keep", False)
        if isinstance(keep, str):
            keep = keep.strip().lower() in {"true", "yes", "1"}
        rows.append(
            {
                "user_input": row.get("user_input", ""),
                "measure": row.get("measure", ""),
                "model": row.get("model_name", ""),
                "model_response": row.get("model_response", ""),
                "keep": bool(keep),
                "judge_error": "error" in judge_output or "keep" not in judge_output,
                "judge_labeled_measures": normalize_measures(row.get("labeled_measure")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No judge rows found in {input_path}")

    df = add_model_metadata(df)
    df["measure_short"] = df["measure"].map(measure_display)

    if not metadata.empty:
        df = df.merge(metadata, on="user_input", how="left")
    else:
        df["synthetic"] = False
        df["source_type"] = "unknown"
        df["labeled_measures"] = df["judge_labeled_measures"]

    df["synthetic"] = df["synthetic"].fillna(False).astype(bool)
    df["source_type"] = np.where(df["synthetic"], "synthetic_rewrite", "in_the_wild")
    df["labeled_measures"] = df["labeled_measures"].where(
        df["labeled_measures"].notna(), df["judge_labeled_measures"]
    )
    for col in ["rewrite_model", "response_model", "source_hash"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("")

    return df


def load_synthetic_audit(synthetic_full_path: Path, rejected_glob: str) -> pd.DataFrame:
    rows = []

    for row in read_jsonl(synthetic_full_path):
        measures = normalize_measures(row.get("measure"))
        generation_measure = measures[0] if measures else ""
        rows.append(
            {
                "generation_measure": generation_measure,
                "reject_stage": "accepted",
                "naturalness_passed": True,
                "rewrite_model": row.get("rewrite_model", ""),
                "response_model": row.get("response_model", ""),
                "source_hash": row.get("source_hash", ""),
                "user_input": row.get("user_input", ""),
            }
        )

    for path_str in glob.glob(rejected_glob):
        path = Path(path_str)
        generation_measure = path.name.replace(".rejected.jsonl", "")
        for row in read_jsonl(path):
            rows.append(
                {
                    "generation_measure": generation_measure,
                    "reject_stage": row.get("reject_stage", "rejected_unknown"),
                    "naturalness_passed": row.get("naturalness_passed"),
                    "rewrite_model": row.get("rewrite_model", ""),
                    "response_model": row.get("response_model", ""),
                    "source_hash": row.get("source_hash", ""),
                    "user_input": row.get("user_input", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "generation_measure",
                "reject_stage",
                "naturalness_passed",
                "rewrite_model",
                "response_model",
                "source_hash",
                "user_input",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main-results outputs


def write_company_outputs(df: pd.DataFrame, outdir: Path) -> None:
    family_measure = rate_table(df, ["family", "measure"])
    family_measure["measure_short"] = family_measure["measure"].map(measure_display)
    write_csv(family_measure, outdir / "main_company_by_measure.csv")

    pivot = (
        family_measure.pivot(index="family", columns="measure", values="violation_rate_pct")
        .reindex(index=FAMILY_ORDER, columns=MEASURE_ORDER)
        .rename(columns=MEASURE_SHORT)
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    draw_heatmap(
        pivot,
        ax,
        title="Company Overall: Violation Rate by Measure",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        cbar_label="Violation rate (%)",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    savefig(fig, outdir / "fig_main_company_measure_heatmap.png")

    company_overall = rate_table(df, ["family"])
    family_rank = {family: i for i, family in enumerate(FAMILY_ORDER)}
    company_overall["_family_rank"] = (
        company_overall["family"].map(family_rank).fillna(len(FAMILY_ORDER))
    )
    company_overall = company_overall.sort_values(["_family_rank", "family"]).drop(
        columns=["_family_rank"]
    )
    write_csv(company_overall, outdir / "main_company_overall.csv")

    model_rates = rate_table(df, ["model"])
    model_rates = add_model_metadata(model_rates)
    trend_rows = []
    for family in FAMILY_ORDER:
        for generation_idx, model in enumerate(GENERATION_MODEL_ORDER[family], start=1):
            row = model_rates[model_rates["model"] == model]
            if row.empty:
                continue
            trend_rows.append(
                {
                    "family": family,
                    "generation_index": generation_idx,
                    "model": model,
                    "model_display": model_display(model),
                    "violation_rate": row.iloc[0]["violation_rate"],
                    "violation_rate_pct": row.iloc[0]["violation_rate_pct"],
                    "total": int(row.iloc[0]["total"]),
                }
            )
    trend = pd.DataFrame(trend_rows)
    write_csv(trend, outdir / "main_company_trend.csv")

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for family in FAMILY_ORDER:
        sub = trend[trend["family"] == family]
        if sub.empty:
            continue
        ax.plot(
            sub["generation_index"],
            sub["violation_rate_pct"],
            marker="o",
            linewidth=2,
            label=family,
            color=FAMILY_COLORS[family],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["model_display"],
                (row["generation_index"], row["violation_rate_pct"]),
                textcoords="offset points",
                xytext=(3, 4),
                fontsize=7,
                alpha=0.75,
            )
    ax.set_title("Company Trend Over Time")
    ax.set_xlabel("Within-company generation index (older to newer)")
    ax.set_ylabel("Violation rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    savefig(fig, outdir / "fig_main_company_trend.png")


def write_thinking_outputs(df: pd.DataFrame, outdir: Path) -> None:
    thinking_overall = rate_table(df, ["thinking_group"])
    write_csv(thinking_overall, outdir / "main_thinking_vs_no_thinking_overall.csv")

    thinking_measure = rate_table(df, ["thinking_group", "measure"])
    thinking_measure["measure_short"] = thinking_measure["measure"].map(measure_display)
    write_csv(thinking_measure, outdir / "main_thinking_vs_no_thinking_by_measure.csv")

    claude = df[df["model"].isin(CLAUDE_THINKING_ORDER)].copy()
    if claude.empty:
        return
    claude["thinking_budget"] = claude["model"].map(CLAUDE_THINKING_BUDGET)
    claude_table = rate_table(claude, ["model", "thinking_budget", "measure"])
    claude_table["model_display"] = claude_table["model"].map(model_display)
    claude_table["measure_short"] = claude_table["measure"].map(measure_display)
    claude_table = claude_table.sort_values(["thinking_budget", "measure"])
    write_csv(claude_table, outdir / "main_claude_thinking_budget_by_measure.csv")

    claude_overall = rate_table(claude, ["model", "thinking_budget"])
    claude_overall["model_display"] = claude_overall["model"].map(model_display)
    claude_overall = claude_overall.sort_values("thinking_budget")
    write_csv(claude_overall, outdir / "main_claude_thinking_budget_overall.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        claude_overall["thinking_budget"],
        claude_overall["violation_rate_pct"],
        marker="o",
        linewidth=2,
        color=FAMILY_COLORS["Anthropic"],
    )
    ax.set_title("Claude Opus 4.6 Thinking Budget")
    ax.set_xlabel("Thinking budget tokens (0 = no thinking)")
    ax.set_ylabel("Violation rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, outdir / "fig_main_claude_thinking_budget.png")


def write_scaling_outputs(df: pd.DataFrame, outdir: Path) -> None:
    model_rates = rate_table(df, ["model"])
    model_rates = add_model_metadata(model_rates)
    model_rates = model_rates.sort_values(["access", "family", "size_b", "generation_index"])
    write_csv(model_rates, outdir / "main_public_private_scaling.csv")

    public = model_rates[(model_rates["access"] == "public") & model_rates["size_b"].notna()]
    private = model_rates[
        (model_rates["access"] == "private") & model_rates["generation_index"].notna()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    if not public.empty:
        for family, sub in public.groupby("family"):
            ax.scatter(
                sub["size_b"],
                sub["violation_rate_pct"],
                s=70,
                label=family,
                color=FAMILY_COLORS.get(family),
            )
            for _, row in sub.iterrows():
                ax.annotate(row["model_display"], (row["size_b"], row["violation_rate_pct"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xscale("log")
    ax.set_title("Public Models: Scale vs Violation")
    ax.set_xlabel("Approx. size (B params; log scale)")
    ax.set_ylabel("Violation rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(alpha=0.25)

    ax = axes[1]
    if not private.empty:
        for family, sub in private.groupby("family"):
            ax.scatter(
                sub["generation_index"],
                sub["violation_rate_pct"],
                s=70,
                label=family,
                color=FAMILY_COLORS.get(family),
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["model_display"],
                    (row["generation_index"], row["violation_rate_pct"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
    ax.set_title("Private Models: Generation vs Violation")
    ax.set_xlabel("Within-company generation index")
    ax.set_ylabel("Violation rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    savefig(fig, outdir / "fig_main_public_private_scaling.png")


# ---------------------------------------------------------------------------
# Ablation outputs


def write_ablation_outputs(df: pd.DataFrame, metadata: pd.DataFrame, outdir: Path) -> None:
    metadata_subset = metadata[
        metadata["user_input"].isin(set(df["user_input"].unique()))
    ].copy()

    target_overall = rate_table(df, ["target_group"])
    write_csv(target_overall, outdir / "ablation_target_vs_non_target_overall.csv")

    target_measure = rate_table(df, ["target_group", "measure"])
    target_measure["measure_short"] = target_measure["measure"].map(measure_display)
    write_csv(target_measure, outdir / "ablation_target_vs_non_target_by_measure.csv")

    prompt_rates = (
        df.groupby(["source_type", "measure", "user_input"], dropna=False)
        .agg(
            prompt_violation_rate=("keep", "mean"),
            n_models=("model", "nunique"),
        )
        .reset_index()
    )
    sensitivity = (
        prompt_rates.groupby(["source_type", "measure"], dropna=False)
        .agg(
            prompts=("user_input", "nunique"),
            mean_prompt_rate=("prompt_violation_rate", "mean"),
            std_prompt_rate=("prompt_violation_rate", "std"),
            p10=("prompt_violation_rate", lambda s: s.quantile(0.10)),
            p25=("prompt_violation_rate", lambda s: s.quantile(0.25)),
            p50=("prompt_violation_rate", "median"),
            p75=("prompt_violation_rate", lambda s: s.quantile(0.75)),
            p90=("prompt_violation_rate", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )
    sensitivity["measure_short"] = sensitivity["measure"].map(measure_display)
    for col in ["mean_prompt_rate", "std_prompt_rate", "p10", "p25", "p50", "p75", "p90"]:
        sensitivity[f"{col}_pct"] = sensitivity[col] * 100
    write_csv(sensitivity, outdir / "ablation_prompt_sensitivity_by_measure.csv")

    plot_measure_label_distribution(metadata_subset, outdir)
    plot_measure_violation_distribution(df, outdir)


def plot_measure_label_distribution(metadata: pd.DataFrame, outdir: Path) -> None:
    if metadata.empty or "labeled_measures" not in metadata:
        print("  fig_ablation_measure_label_distribution.png skipped (no metadata)")
        return

    rows = []
    for _, row in metadata.iterrows():
        for measure in normalize_measures(row.get("labeled_measures")):
            rows.append(
                {
                    "measure": measure,
                    "measure_short": measure_display(measure),
                    "source_type": row.get("source_type", "unknown"),
                }
            )
    label_df = pd.DataFrame(rows)
    if label_df.empty:
        print("  fig_ablation_measure_label_distribution.png skipped (no labels)")
        return

    counts = (
        label_df.groupby(["measure", "source_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=MEASURE_ORDER)
    )
    counts_display = counts.rename(index=MEASURE_SHORT)
    write_csv(counts.reset_index(), outdir / "ablation_measure_label_distribution.csv")

    fig, ax = plt.subplots(figsize=(9, 5.8))
    counts_display.plot(kind="barh", stacked=True, ax=ax, color=["#64748b", "#38bdf8"])
    ax.set_title("Input Label Distribution by Source")
    ax.set_xlabel("Input-label count")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.legend(title="")
    savefig(fig, outdir / "fig_ablation_measure_label_distribution.png")


def plot_measure_violation_distribution(df: pd.DataFrame, outdir: Path) -> None:
    table = rate_table(df, ["source_type", "measure"])
    table["measure_short"] = table["measure"].map(measure_display)
    write_csv(table, outdir / "ablation_measure_violation_distribution.csv")

    pivot = (
        table.pivot(index="measure", columns="source_type", values="violation_rate_pct")
        .reindex(index=MEASURE_ORDER)
        .rename(index=MEASURE_SHORT)
    )
    fig, ax = plt.subplots(figsize=(9, 5.8))
    pivot.plot(kind="barh", ax=ax, color=["#64748b", "#38bdf8"])
    ax.set_title("Judge Violation Distribution by Source")
    ax.set_xlabel("Violation rate (%)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.invert_yaxis()
    ax.legend(title="")
    savefig(fig, outdir / "fig_ablation_measure_violation_distribution.png")


# ---------------------------------------------------------------------------
# Rewriting vs in-the-wild outputs


def write_rewriting_outputs(df: pd.DataFrame, outdir: Path) -> None:
    source_family = rate_table(df, ["source_type", "family"])
    write_csv(source_family, outdir / "rewrite_vs_wild_by_family.csv")

    source_model = rate_table(df, ["source_type", "model"])
    source_model = add_model_metadata(source_model)
    write_csv(source_model, outdir / "rewrite_vs_wild_by_model.csv")

    source_measure = rate_table(df, ["source_type", "measure"])
    source_measure["measure_short"] = source_measure["measure"].map(measure_display)
    write_csv(source_measure, outdir / "rewrite_vs_wild_by_measure.csv")

    pivot = (
        source_family.pivot(index="family", columns="source_type", values="violation_rate_pct")
        .reindex(index=FAMILY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pivot.plot(kind="bar", ax=ax, color=["#64748b", "#38bdf8"])
    ax.set_title("Rewritten vs In-the-Wild Performance")
    ax.set_xlabel("")
    ax.set_ylabel("Violation rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    savefig(fig, outdir / "fig_rewrite_vs_wild_by_family.png")

    synthetic = df[(df["source_type"] == "synthetic_rewrite") & (df["rewrite_model"] != "")]
    if synthetic.empty:
        print("  rewrite provenance outputs skipped (no synthetic provenance)")
        return

    rewrite_family = rate_table(synthetic, ["rewrite_model", "family"])
    write_csv(rewrite_family, outdir / "rewrite_model_by_eval_family.csv")
    heat = (
        rewrite_family.pivot(
            index="rewrite_model", columns="family", values="violation_rate_pct"
        )
        .reindex(columns=FAMILY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    draw_heatmap(
        heat,
        ax,
        title="Synthetic Rewriter x Evaluated Company",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        cbar_label="Violation rate (%)",
    )
    ax.set_xlabel("Evaluated company")
    ax.set_ylabel("Synthetic rewrite model")
    savefig(fig, outdir / "fig_rewrite_model_by_eval_family.png")

    rewrite_pair = rate_table(synthetic, ["rewrite_model", "response_model"])
    write_csv(rewrite_pair, outdir / "rewrite_model_response_model_pair.csv")
    pair_heat = rewrite_pair.pivot(
        index="rewrite_model", columns="response_model", values="violation_rate_pct"
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    draw_heatmap(
        pair_heat,
        ax,
        title="Synthetic Rewriter x Source Response Model",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        cbar_label="Violation rate (%)",
    )
    ax.set_xlabel("Response model used during synthesis")
    ax.set_ylabel("Rewrite model")
    savefig(fig, outdir / "fig_rewrite_pair_violation_rate.png")

    rewrite_eval_model = rate_table(
        synthetic, ["rewrite_model", "response_model", "model"]
    )
    rewrite_eval_model = add_model_metadata(rewrite_eval_model)
    write_csv(rewrite_eval_model, outdir / "rewrite_model_response_model_eval_model.csv")


# ---------------------------------------------------------------------------
# Human alignment / naturalness-filter audit outputs


def write_human_alignment_outputs(audit: pd.DataFrame, outdir: Path) -> None:
    if audit.empty:
        print("  human-alignment outputs skipped (no synthetic audit rows)")
        return

    funnel = (
        audit.groupby(["generation_measure", "reject_stage"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    funnel["measure_short"] = funnel["generation_measure"].map(measure_display)
    write_csv(funnel, outdir / "human_alignment_generation_funnel.csv")

    stage_pivot = (
        funnel.pivot(index="generation_measure", columns="reject_stage", values="rows")
        .fillna(0)
        .reindex(index=MEASURE_ORDER)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    stage_pivot.rename(index=MEASURE_SHORT).plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Synthetic Generation Funnel")
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.legend(title="Stage", fontsize=7)
    savefig(fig, outdir / "fig_human_alignment_generation_funnel.png")

    step5 = audit[audit["reject_stage"].isin(["accepted", "rejected_step5_ranked_last"])]
    if step5.empty:
        print("  naturalness pass-rate outputs skipped (no Step 5 rows)")
        return

    step5 = step5.copy()
    step5["naturalness_outcome"] = np.where(
        step5["reject_stage"] == "accepted", "passed", "ranked_last"
    )

    naturalness = (
        step5.groupby(
            ["generation_measure", "rewrite_model", "response_model", "naturalness_outcome"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "passed" not in naturalness:
        naturalness["passed"] = 0
    if "ranked_last" not in naturalness:
        naturalness["ranked_last"] = 0
    naturalness["step5_total"] = naturalness["passed"] + naturalness["ranked_last"]
    naturalness["pass_rate"] = naturalness["passed"] / naturalness["step5_total"]
    naturalness["pass_rate_pct"] = naturalness["pass_rate"] * 100
    naturalness["measure_short"] = naturalness["generation_measure"].map(measure_display)
    write_csv(naturalness, outdir / "human_alignment_naturalness_by_measure_pair.csv")

    by_measure_rewriter = (
        naturalness.groupby(["generation_measure", "rewrite_model"], dropna=False)
        .agg(passed=("passed", "sum"), ranked_last=("ranked_last", "sum"))
        .reset_index()
    )
    by_measure_rewriter["step5_total"] = (
        by_measure_rewriter["passed"] + by_measure_rewriter["ranked_last"]
    )
    by_measure_rewriter["pass_rate"] = (
        by_measure_rewriter["passed"] / by_measure_rewriter["step5_total"]
    )
    by_measure_rewriter["pass_rate_pct"] = by_measure_rewriter["pass_rate"] * 100
    by_measure_rewriter["measure_short"] = by_measure_rewriter["generation_measure"].map(
        measure_display
    )
    write_csv(
        by_measure_rewriter,
        outdir / "human_alignment_naturalness_by_measure_rewriter.csv",
    )

    heat = by_measure_rewriter.pivot(
        index="generation_measure", columns="rewrite_model", values="pass_rate_pct"
    ).reindex(index=MEASURE_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    draw_heatmap(
        heat.rename(index=MEASURE_SHORT),
        ax,
        title="Naturalness Filter Pass Rate by Measure and Rewriter",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_label="Naturalness pass rate (%)",
    )
    ax.set_xlabel("Rewrite model")
    ax.set_ylabel("")
    savefig(fig, outdir / "fig_human_alignment_naturalness_by_rewriter.png")

    pair = (
        naturalness.groupby(["rewrite_model", "response_model"], dropna=False)
        .agg(passed=("passed", "sum"), ranked_last=("ranked_last", "sum"))
        .reset_index()
    )
    pair["step5_total"] = pair["passed"] + pair["ranked_last"]
    pair["pass_rate"] = pair["passed"] / pair["step5_total"]
    pair["pass_rate_pct"] = pair["pass_rate"] * 100
    write_csv(pair, outdir / "human_alignment_naturalness_by_model_pair.csv")

    pair_heat = pair.pivot(
        index="rewrite_model", columns="response_model", values="pass_rate_pct"
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    draw_heatmap(
        pair_heat,
        ax,
        title="Naturalness Filter Pass Rate by Model Pair",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_label="Naturalness pass rate (%)",
    )
    ax.set_xlabel("Response model used during synthesis")
    ax.set_ylabel("Rewrite model")
    savefig(fig, outdir / "fig_human_alignment_naturalness_by_model_pair.png")


# ---------------------------------------------------------------------------
# Manifest


def write_manifest(outdir: Path) -> None:
    lines = [
        "# Stage 7.3 Analysis Artifact Manifest",
        "",
        "This file lists the generated artifacts for the planned analyses. It does not write paper paragraphs.",
        "",
        "## Main Results",
        "- `main_company_overall.csv`",
        "- `main_company_by_measure.csv`",
        "- `main_company_trend.csv`",
        "- `fig_main_company_measure_heatmap.png`",
        "- `fig_main_company_trend.png`",
        "- `main_thinking_vs_no_thinking_overall.csv`",
        "- `main_thinking_vs_no_thinking_by_measure.csv`",
        "- `main_claude_thinking_budget_overall.csv`",
        "- `main_claude_thinking_budget_by_measure.csv`",
        "- `fig_main_claude_thinking_budget.png`",
        "- `main_public_private_scaling.csv`",
        "- `fig_main_public_private_scaling.png`",
        "",
        "## Ablations",
        "- `ablation_target_vs_non_target_overall.csv`",
        "- `ablation_target_vs_non_target_by_measure.csv`",
        "- `ablation_prompt_sensitivity_by_measure.csv`",
        "- `ablation_measure_label_distribution.csv`",
        "- `ablation_measure_violation_distribution.csv`",
        "- `fig_ablation_measure_label_distribution.png`",
        "- `fig_ablation_measure_violation_distribution.png`",
        "",
        "## Rewriting vs In-the-Wild",
        "- `rewrite_vs_wild_by_family.csv`",
        "- `rewrite_vs_wild_by_model.csv`",
        "- `rewrite_vs_wild_by_measure.csv`",
        "- `fig_rewrite_vs_wild_by_family.png`",
        "- `rewrite_model_by_eval_family.csv`",
        "- `rewrite_model_response_model_pair.csv`",
        "- `rewrite_model_response_model_eval_model.csv`",
        "- `fig_rewrite_model_by_eval_family.png`",
        "- `fig_rewrite_pair_violation_rate.png`",
        "",
        "## Human Alignment / Naturalness Filter",
        "- `human_alignment_generation_funnel.csv`",
        "- `human_alignment_naturalness_by_measure_pair.csv`",
        "- `human_alignment_naturalness_by_measure_rewriter.csv`",
        "- `human_alignment_naturalness_by_model_pair.csv`",
        "- `fig_human_alignment_generation_funnel.png`",
        "- `fig_human_alignment_naturalness_by_rewriter.png`",
        "- `fig_human_alignment_naturalness_by_model_pair.png`",
        "",
    ]
    path = outdir / "analysis_manifest.md"
    path.write_text("\n".join(lines))
    print(f"  {path.name}")


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7.3 - Generate requested evaluation analysis artifacts."
    )
    parser.add_argument("--input", default="data/eval_judge_results.jsonl")
    parser.add_argument("--dataset", default="data/final_dataset.jsonl")
    parser.add_argument(
        "--synthetic_metadata", default="data/synthetic_data_relabelled.jsonl"
    )
    parser.add_argument("--synthetic_full", default="data/synthetic_data_full.jsonl")
    parser.add_argument("--rejected_glob", default="data/synthetic/*.rejected.jsonl")
    parser.add_argument("--outdir", default="data/eval_figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    metadata = load_input_metadata(Path(args.dataset), Path(args.synthetic_metadata))
    print(f"  metadata rows: {len(metadata)}")

    print(f"Loading judge results from {args.input}...")
    df = load_judge_data(Path(args.input), metadata)
    print(
        f"  {len(df)} judge rows, {df['user_input'].nunique()} inputs, "
        f"{df['model'].nunique()} models, {df['measure'].nunique()} measures"
    )
    print(f"  overall violation rate: {df['keep'].mean() * 100:.1f}%")
    if df["judge_error"].any():
        print(f"  warning: {int(df['judge_error'].sum())} judge rows had parse/error flags")

    print("\nGenerating main-result artifacts...")
    write_company_outputs(df, outdir)
    write_thinking_outputs(df, outdir)
    write_scaling_outputs(df, outdir)

    print("\nGenerating ablation artifacts...")
    write_ablation_outputs(df, metadata, outdir)

    print("\nGenerating rewriting-vs-wild artifacts...")
    write_rewriting_outputs(df, outdir)

    print("\nGenerating human-alignment / naturalness artifacts...")
    audit = load_synthetic_audit(Path(args.synthetic_full), args.rejected_glob)
    print(f"  synthetic audit rows: {len(audit)}")
    write_human_alignment_outputs(audit, outdir)

    print("\nWriting manifest...")
    write_manifest(outdir)

    print(f"\nAll analysis artifacts saved to {outdir}/")


if __name__ == "__main__":
    main()
