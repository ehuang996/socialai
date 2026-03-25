"""Stage 8.1 — Comprehensive analysis of Stage 7.1 single-turn judge results.

Generates figures, tables, and statistics comparing model performance across
measures, families, temporal evolution, and measure co-occurrence.

Usage:
    uv run python src/evaluation/stage8.1/analyze_single_turn.py [--input ...] [--outdir ...]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants ──────────────────────────────────────────────────────────────

MEASURE_ORDER = [
    "1B_human_disfluencies",
    "1C_identity_transparency",
    "2A_fabricated_personal_details",
    "2B_explicit_emotions",
    "2B_implicit_emotions",
    "2B_romantic_bonding",
    "2C_sycophancy",
    "3A_engagement_hooks",
]

MEASURE_SHORT = {
    "1B_human_disfluencies": "1B Disfluencies",
    "1C_identity_transparency": "1C Identity",
    "2A_fabricated_personal_details": "2A Fabrication",
    "2B_explicit_emotions": "2B Explicit Emo",
    "2B_implicit_emotions": "2B Implicit Emo",
    "2B_romantic_bonding": "2B Romantic",
    "2C_sycophancy": "2C Sycophancy",
    "3A_engagement_hooks": "3A Engagement",
}

# Model families: ordered newest → oldest within each
MODEL_FAMILIES = {
    "OpenAI": ["gpt5_4_pro", "gpt5_4", "gpt5_3", "o4_mini", "gpt4o_mini"],
    "Google": ["gemini3_1_pro", "gemini3_flash", "gemini2_flash_001"],
    "Anthropic": ["claude_opus", "claude_sonnet", "claude_sonnet_4", "claude_haiku"],
    "xAI": ["grok4", "grok3_mini_beta"],
}

MODEL_TO_FAMILY = {}
for fam, models in MODEL_FAMILIES.items():
    for m in models:
        MODEL_TO_FAMILY[m] = fam

MODEL_DISPLAY = {
    "gpt5_4_pro": "GPT-5-4 Pro",
    "gpt5_4": "GPT-5-4",
    "gpt5_3": "GPT-5-3",
    "o4_mini": "o4-mini",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini3_1_pro": "Gemini 3.1 Pro",
    "gemini3_flash": "Gemini 3 Flash",
    "gemini2_flash_001": "Gemini 2 Flash",
    "claude_opus": "Claude Opus",
    "claude_sonnet": "Claude Sonnet",
    "claude_sonnet_4": "Claude Sonnet 4",
    "claude_haiku": "Claude Haiku",
    "grok4": "Grok 4",
    "grok3_mini_beta": "Grok 3 Mini",
}

# Flatten model order (all families, newest first)
MODEL_ORDER = []
for fam in ["OpenAI", "Google", "Anthropic", "xAI"]:
    MODEL_ORDER.extend(MODEL_FAMILIES[fam])

FAMILY_COLORS = {
    "OpenAI": "#10a37f",
    "Google": "#4285f4",
    "Anthropic": "#d97706",
    "xAI": "#ef4444",
}

MODEL_COLORS = {}
for fam, models in MODEL_FAMILIES.items():
    base = FAMILY_COLORS[fam]
    # Generate shades from dark to light (newest=darkest)
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(base)
    for i, m in enumerate(models):
        factor = 1.0 - (i * 0.15)
        MODEL_COLORS[m] = tuple(min(1, c * factor + (1 - factor) * 0.5) for c in rgb)


# ── Data loading ───────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "user_input": r["user_input"],
                "measure": r["measure"],
                "model": r["model_name"],
                "keep": r["judge_output"].get("keep", False),
                "family": MODEL_TO_FAMILY.get(r["model_name"], "Unknown"),
            })
    df = pd.DataFrame(rows)
    df["measure_short"] = df["measure"].map(MEASURE_SHORT)
    df["model_display"] = df["model"].map(MODEL_DISPLAY)
    return df


# ── Figure generators ──────────────────────────────────────────────────────


def fig1_overall_violation_rate(df: pd.DataFrame, outdir: Path):
    """Bar chart: overall violation rate per model, colored by family."""
    rates = df.groupby("model")["keep"].mean().reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(
        range(len(MODEL_ORDER)),
        rates.values * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=45, ha="right")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Overall Violation Rate by Model (Single-Turn, All Measures)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    # Add family labels
    prev_fam = None
    for i, m in enumerate(MODEL_ORDER):
        fam = MODEL_TO_FAMILY[m]
        if fam != prev_fam:
            ax.axvline(i - 0.5, color="gray", linestyle="--", alpha=0.3)
            ax.text(i, ax.get_ylim()[1] * 0.95, fam, fontsize=8, ha="left", alpha=0.6)
            prev_fam = fam

    # Value labels on bars
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(outdir / "fig1_overall_violation_rate.png", dpi=200)
    plt.close()
    print("  fig1_overall_violation_rate.png")


def fig2_heatmap_model_measure(df: pd.DataFrame, outdir: Path):
    """Heatmap: violation rate per (model, measure) pair."""
    pivot = df.pivot_table(
        values="keep", index="model", columns="measure", aggfunc="mean"
    )
    pivot = pivot.reindex(index=MODEL_ORDER, columns=MEASURE_ORDER)
    pivot_display = pivot.rename(index=MODEL_DISPLAY, columns=MEASURE_SHORT)

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        pivot_display * 100,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Violation Rate (%)"},
    )
    ax.set_title("Violation Rate (%) by Model × Measure")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(outdir / "fig2_heatmap_model_measure.png", dpi=200)
    plt.close()
    print("  fig2_heatmap_model_measure.png")


def fig3_family_comparison(df: pd.DataFrame, outdir: Path):
    """Grouped bar chart: family-level violation rate per measure."""
    family_measure = (
        df.groupby(["family", "measure"])["keep"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=MEASURE_ORDER)
    )
    # Reorder families
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]
    family_measure = family_measure.reindex(fam_order)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(MEASURE_ORDER))
    width = 0.18
    for i, fam in enumerate(fam_order):
        vals = family_measure.loc[fam].values * 100
        bars = ax.bar(x + i * width, vals, width, label=fam, color=FAMILY_COLORS[fam])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([MEASURE_SHORT[m] for m in MEASURE_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Family-Level Violation Rate by Measure")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    plt.tight_layout()
    fig.savefig(outdir / "fig3_family_comparison.png", dpi=200)
    plt.close()
    print("  fig3_family_comparison.png")


def fig4_temporal_evolution(df: pd.DataFrame, outdir: Path):
    """Line plots per family: how violation rate changes across model generations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]

    for ax, fam in zip(axes.flat, fam_order):
        models = MODEL_FAMILIES[fam]
        # Overall rate per model (newest first → reverse for x-axis oldest→newest)
        models_rev = list(reversed(models))
        rates = df[df["family"] == fam].groupby("model")["keep"].mean()
        vals = [rates.get(m, 0) * 100 for m in models_rev]
        labels = [MODEL_DISPLAY[m] for m in models_rev]

        ax.plot(labels, vals, "o-", color=FAMILY_COLORS[fam], linewidth=2, markersize=8)
        ax.set_title(f"{fam} — Evolution Over Time", fontsize=12)
        ax.set_ylabel("Violation Rate (%)")
        ax.set_xlabel("Older → Newer")
        ax.tick_params(axis="x", rotation=25)

        # Also show per-measure lines (thinner)
        for measure in MEASURE_ORDER:
            sub = df[(df["family"] == fam) & (df["measure"] == measure)]
            if sub.empty:
                continue
            m_rates = sub.groupby("model")["keep"].mean()
            m_vals = [m_rates.get(m, 0) * 100 for m in models_rev]
            ax.plot(
                labels, m_vals, ".-",
                alpha=0.3, linewidth=1, markersize=4,
                label=MEASURE_SHORT[measure],
            )
        ax.legend(fontsize=6, loc="upper left", ncol=2)

    plt.suptitle("Temporal Evolution: Violation Rate by Model Generation", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "fig4_temporal_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig4_temporal_evolution.png")


def fig5_measure_overall(df: pd.DataFrame, outdir: Path):
    """Horizontal bar chart: overall violation rate per measure."""
    rates = df.groupby("measure")["keep"].mean().reindex(MEASURE_ORDER) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(MEASURE_ORDER)))
    bars = ax.barh(
        [MEASURE_SHORT[m] for m in MEASURE_ORDER],
        rates.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, rates.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Violation Rate (%)")
    ax.set_title("Overall Violation Rate by Measure (Across All Models)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(outdir / "fig5_measure_overall.png", dpi=200)
    plt.close()
    print("  fig5_measure_overall.png")


def fig6_family_radar(df: pd.DataFrame, outdir: Path):
    """Radar/spider chart comparing families across all measures."""
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]
    categories = [MEASURE_SHORT[m] for m in MEASURE_ORDER]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for fam in fam_order:
        rates = df[df["family"] == fam].groupby("measure")["keep"].mean()
        values = [rates.get(m, 0) * 100 for m in MEASURE_ORDER]
        values += values[:1]
        ax.plot(angles, values, "o-", label=fam, color=FAMILY_COLORS[fam], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=FAMILY_COLORS[fam])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title("Family Comparison: Violation Rate by Measure", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(outdir / "fig6_family_radar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig6_family_radar.png")


def fig7_per_family_temporal_per_measure(df: pd.DataFrame, outdir: Path):
    """Grid: one subplot per measure, lines per family showing temporal trend."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]

    for ax, measure in zip(axes.flat, MEASURE_ORDER):
        sub = df[df["measure"] == measure]
        for fam in fam_order:
            models = list(reversed(MODEL_FAMILIES[fam]))  # oldest→newest
            fam_sub = sub[sub["family"] == fam]
            if fam_sub.empty:
                continue
            rates = fam_sub.groupby("model")["keep"].mean()
            vals = [rates.get(m, 0) * 100 for m in models]
            # Use generation index (0=oldest) for alignment
            ax.plot(
                range(len(models)), vals, "o-",
                color=FAMILY_COLORS[fam], label=fam, linewidth=1.5, markersize=5,
            )

        ax.set_title(MEASURE_SHORT[measure], fontsize=10)
        ax.set_ylabel("Violation %", fontsize=8)
        ax.set_xlabel("Generation (old→new)", fontsize=8)
        ax.set_ylim(-5, 105)

    axes[0, 0].legend(fontsize=7)
    plt.suptitle("Temporal Trend per Measure, by Family", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "fig7_temporal_per_measure.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig7_temporal_per_measure.png")


def fig8_model_rank_bump(df: pd.DataFrame, outdir: Path):
    """Bump chart: model rank (by violation rate) across measures."""
    fig, ax = plt.subplots(figsize=(16, 9))

    for measure_idx, measure in enumerate(MEASURE_ORDER):
        sub = df[df["measure"] == measure]
        rates = sub.groupby("model")["keep"].mean()
        # Rank: highest violation = rank 1 (worst)
        ranked = rates.rank(ascending=False, method="min")
        for model in MODEL_ORDER:
            if model in ranked.index:
                rank = ranked[model]
                fam = MODEL_TO_FAMILY[model]
                ax.plot(
                    measure_idx, rank, "o",
                    color=FAMILY_COLORS[fam],
                    markersize=6, alpha=0.7,
                )
                if measure_idx > 0:
                    # Connect to previous measure
                    prev_measure = MEASURE_ORDER[measure_idx - 1]
                    prev_sub = df[df["measure"] == prev_measure]
                    prev_rates = prev_sub.groupby("model")["keep"].mean()
                    prev_ranked = prev_rates.rank(ascending=False, method="min")
                    if model in prev_ranked.index:
                        ax.plot(
                            [measure_idx - 1, measure_idx],
                            [prev_ranked[model], rank],
                            "-", color=FAMILY_COLORS[fam], alpha=0.3, linewidth=1,
                        )

    ax.set_xticks(range(len(MEASURE_ORDER)))
    ax.set_xticklabels([MEASURE_SHORT[m] for m in MEASURE_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Rank (1 = highest violation rate)")
    ax.set_title("Model Rank by Violation Rate Across Measures (Bump Chart)")
    ax.invert_yaxis()

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=FAMILY_COLORS[f], label=f, markersize=8, linestyle="")
        for f in ["OpenAI", "Google", "Anthropic", "xAI"]
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    fig.savefig(outdir / "fig8_model_rank_bump.png", dpi=200)
    plt.close()
    print("  fig8_model_rank_bump.png")


def fig9_overlap_analysis(df: pd.DataFrame, outdir: Path):
    """Analyze conversations with multiple measures: do models violate more when
    the conversation is flagged for multiple categories?"""
    # Build input → measures mapping
    input_measures = df.groupby("user_input")["measure"].apply(set).to_dict()
    df["num_measures"] = df["user_input"].map(lambda x: len(input_measures[x]))
    df["is_multi"] = df["num_measures"] > 1

    # Compare violation rate: single-measure vs multi-measure conversations
    single = df[~df["is_multi"]]["keep"].mean() * 100
    multi = df[df["is_multi"]]["keep"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: overall comparison
    ax = axes[0]
    bars = ax.bar(
        ["Single-Measure\nConversations", "Multi-Measure\nConversations"],
        [single, multi],
        color=["#3b82f6", "#ef4444"],
        edgecolor="black",
    )
    for bar, val in zip(bars, [single, multi]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Single vs Multi-Measure: Violation Rate")

    # Right: per-family comparison
    ax = axes[1]
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]
    x = np.arange(len(fam_order))
    width = 0.35
    single_rates = [df[(df["family"] == f) & (~df["is_multi"])]["keep"].mean() * 100 for f in fam_order]
    multi_rates = [df[(df["family"] == f) & (df["is_multi"])]["keep"].mean() * 100 for f in fam_order]
    ax.bar(x - width / 2, single_rates, width, label="Single-Measure", color="#3b82f6")
    ax.bar(x + width / 2, multi_rates, width, label="Multi-Measure", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(fam_order)
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Single vs Multi-Measure by Family")
    ax.legend()

    plt.suptitle("Overlap Analysis: Conversations Flagged for Multiple Categories", fontsize=13)
    plt.tight_layout()
    fig.savefig(outdir / "fig9_overlap_analysis.png", dpi=200)
    plt.close()
    print("  fig9_overlap_analysis.png")

    # Clean up temp columns
    df.drop(columns=["num_measures", "is_multi"], inplace=True)


def fig10_measure_correlation(df: pd.DataFrame, outdir: Path):
    """Heatmap: correlation of violation rates across measures (by model).
    If a model violates measure A, does it also tend to violate measure B?"""
    # Pivot: each row is a model, columns are measures, values are violation rates
    pivot = df.pivot_table(values="keep", index="model", columns="measure", aggfunc="mean")
    pivot = pivot.reindex(columns=MEASURE_ORDER)
    corr = pivot.corr()
    corr_display = corr.rename(index=MEASURE_SHORT, columns=MEASURE_SHORT)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_display,
        annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Measure Correlation: Do Models That Violate One Measure Also Violate Others?")
    plt.tight_layout()
    fig.savefig(outdir / "fig10_measure_correlation.png", dpi=200)
    plt.close()
    print("  fig10_measure_correlation.png")


def fig11_best_worst_models(df: pd.DataFrame, outdir: Path):
    """For each measure, show the best and worst performing model."""
    fig, ax = plt.subplots(figsize=(14, 8))

    measures = MEASURE_ORDER
    best_models = []
    worst_models = []
    best_rates = []
    worst_rates = []

    for measure in measures:
        sub = df[df["measure"] == measure]
        rates = sub.groupby("model")["keep"].mean()
        best = rates.idxmin()
        worst = rates.idxmax()
        best_models.append(MODEL_DISPLAY[best])
        worst_models.append(MODEL_DISPLAY[worst])
        best_rates.append(rates[best] * 100)
        worst_rates.append(rates[worst] * 100)

    y = np.arange(len(measures))
    height = 0.35
    bars_worst = ax.barh(y - height / 2, worst_rates, height, label="Worst", color="#ef4444", alpha=0.8)
    bars_best = ax.barh(y + height / 2, best_rates, height, label="Best", color="#22c55e", alpha=0.8)

    for bar, name in zip(bars_worst, worst_models):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{name} ({bar.get_width():.0f}%)", va="center", fontsize=8)
    for bar, name in zip(bars_best, best_models):
        ax.text(max(bar.get_width() + 0.5, 1), bar.get_y() + bar.get_height() / 2,
                f"{name} ({bar.get_width():.0f}%)", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([MEASURE_SHORT[m] for m in measures])
    ax.set_xlabel("Violation Rate (%)")
    ax.set_title("Best vs Worst Model per Measure")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(outdir / "fig11_best_worst_models.png", dpi=200)
    plt.close()
    print("  fig11_best_worst_models.png")


def fig12_family_aggregate_stacked(df: pd.DataFrame, outdir: Path):
    """Stacked bar: each family's total violations broken down by measure."""
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]
    fig, ax = plt.subplots(figsize=(12, 7))

    bottoms = np.zeros(len(fam_order))
    colors = plt.cm.Set2(np.linspace(0, 1, len(MEASURE_ORDER)))

    for i, measure in enumerate(MEASURE_ORDER):
        vals = []
        for fam in fam_order:
            sub = df[(df["family"] == fam) & (df["measure"] == measure)]
            vals.append(sub["keep"].sum())
        ax.bar(fam_order, vals, bottom=bottoms, label=MEASURE_SHORT[measure], color=colors[i])
        bottoms += vals

    ax.set_ylabel("Number of Violations (keep=true)")
    ax.set_title("Total Violations by Family, Broken Down by Measure")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "fig12_family_stacked.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig12_family_stacked.png")


def fig13_per_input_violation_count(df: pd.DataFrame, outdir: Path):
    """Distribution: how many models (out of 14) violate per (input, measure)."""
    violation_counts = (
        df.groupby(["user_input", "measure"])["keep"]
        .sum()
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram of violation counts
    ax = axes[0]
    ax.hist(violation_counts.values, bins=range(0, 16), edgecolor="black", color="#6366f1", alpha=0.8)
    ax.set_xlabel("Number of Models Violating (out of 14)")
    ax.set_ylabel("Count of (input, measure) pairs")
    ax.set_title("Distribution: How Many Models Violate per Prompt?")
    ax.set_xticks(range(0, 15))

    # Right: proportion where 0, 1-3, 4-7, 8-14 models violate
    ax = axes[1]
    bins = [0, 1, 4, 8, 15]
    labels = ["0\n(None)", "1-3\n(Few)", "4-7\n(Half)", "8-14\n(Most/All)"]
    counts = pd.cut(violation_counts, bins=bins, right=False, labels=labels).value_counts().reindex(labels)
    colors = ["#22c55e", "#eab308", "#f97316", "#ef4444"]
    ax.bar(labels, counts.values, color=colors, edgecolor="black")
    for i, (label, val) in enumerate(zip(labels, counts.values)):
        ax.text(i, val + 0.5, str(val), ha="center", fontsize=10)
    ax.set_ylabel("Count of (input, measure) pairs")
    ax.set_title("Consensus: How Many Models Agree on Violation?")

    plt.tight_layout()
    fig.savefig(outdir / "fig13_per_input_violation_dist.png", dpi=200)
    plt.close()
    print("  fig13_per_input_violation_dist.png")


def fig14_improvement_delta(df: pd.DataFrame, outdir: Path):
    """Bar chart: improvement (delta in violation rate) from oldest to newest model
    within each family, per measure. Negative = improvement (fewer violations)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fam_order = ["OpenAI", "Google", "Anthropic", "xAI"]

    data_rows = []
    for fam in fam_order:
        models = MODEL_FAMILIES[fam]
        newest, oldest = models[0], models[-1]
        for measure in MEASURE_ORDER:
            sub = df[df["measure"] == measure]
            new_rate = sub[sub["model"] == newest]["keep"].mean()
            old_rate = sub[sub["model"] == oldest]["keep"].mean()
            delta = (new_rate - old_rate) * 100
            data_rows.append({
                "family": fam,
                "measure": MEASURE_SHORT[measure],
                "delta": delta,
            })

    delta_df = pd.DataFrame(data_rows)
    pivot = delta_df.pivot(index="measure", columns="family", values="delta")
    pivot = pivot[fam_order]

    x = np.arange(len(MEASURE_ORDER))
    width = 0.18
    for i, fam in enumerate(fam_order):
        vals = pivot[fam].values
        ax.bar(x + i * width, vals, width, label=fam, color=FAMILY_COLORS[fam])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([MEASURE_SHORT[m] for m in MEASURE_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Change in Violation Rate (pp)")
    ax.set_title("Newest vs Oldest Model: Change in Violation Rate (negative = improvement)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig14_improvement_delta.png", dpi=200)
    plt.close()
    print("  fig14_improvement_delta.png")


def fig15_co_occurrence_heatmap(df: pd.DataFrame, outdir: Path):
    """For multi-measure conversations: when a model violates measure A,
    does it also violate measure B on the same conversation?"""
    # Only multi-measure inputs
    input_measures = df.groupby("user_input")["measure"].apply(set).to_dict()
    multi_inputs = {k for k, v in input_measures.items() if len(v) > 1}

    if not multi_inputs:
        print("  fig15 skipped (no multi-measure inputs)")
        return

    multi_df = df[df["user_input"].isin(multi_inputs)]

    # For each (input, model), check which measures have keep=True
    co_matrix = np.zeros((len(MEASURE_ORDER), len(MEASURE_ORDER)))
    co_total = np.zeros((len(MEASURE_ORDER), len(MEASURE_ORDER)))

    measure_idx = {m: i for i, m in enumerate(MEASURE_ORDER)}
    for (ui, model), grp in multi_df.groupby(["user_input", "model"]):
        measures_in = list(grp["measure"])
        keeps = dict(zip(grp["measure"], grp["keep"]))
        for i, m1 in enumerate(measures_in):
            for m2 in measures_in[i + 1:]:
                i1, i2 = measure_idx[m1], measure_idx[m2]
                co_total[i1, i2] += 1
                co_total[i2, i1] += 1
                if keeps.get(m1) and keeps.get(m2):
                    co_matrix[i1, i2] += 1
                    co_matrix[i2, i1] += 1

    # Rate
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(co_total > 0, co_matrix / co_total * 100, np.nan)

    labels = [MEASURE_SHORT[m] for m in MEASURE_ORDER]
    rate_df = pd.DataFrame(rate, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rate_df, annot=True, fmt=".0f",
        cmap="YlOrRd", vmin=0, vmax=100,
        linewidths=0.5, ax=ax, mask=np.isnan(rate),
        cbar_kws={"label": "Co-violation Rate (%)"},
    )
    ax.set_title("Co-violation Rate: When Both Measures Apply, How Often Are Both Violated?")
    plt.tight_layout()
    fig.savefig(outdir / "fig15_co_occurrence_heatmap.png", dpi=200)
    plt.close()
    print("  fig15_co_occurrence_heatmap.png")


# ── Summary tables ─────────────────────────────────────────────────────────


def generate_tables(df: pd.DataFrame, outdir: Path):
    """Generate summary statistics as markdown tables."""
    lines = ["# Stage 8.1 — Single-Turn Analysis Summary\n"]

    # Table 1: Overall violation rate per model
    lines.append("## Table 1: Overall Violation Rate by Model\n")
    lines.append("| Model | Family | Violations | Total | Rate |")
    lines.append("|---|---|---|---|---|")
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        violations = sub["keep"].sum()
        total = len(sub)
        rate = violations / total * 100
        lines.append(
            f"| {MODEL_DISPLAY[model]} | {MODEL_TO_FAMILY[model]} | "
            f"{violations} | {total} | {rate:.1f}% |"
        )
    lines.append("")

    # Table 2: Violation rate per measure
    lines.append("## Table 2: Overall Violation Rate by Measure\n")
    lines.append("| Measure | Inputs | Violations | Total | Rate |")
    lines.append("|---|---|---|---|---|")
    for measure in MEASURE_ORDER:
        sub = df[df["measure"] == measure]
        violations = sub["keep"].sum()
        total = len(sub)
        n_inputs = sub["user_input"].nunique()
        rate = violations / total * 100
        lines.append(
            f"| {MEASURE_SHORT[measure]} | {n_inputs} | "
            f"{violations} | {total} | {rate:.1f}% |"
        )
    lines.append("")

    # Table 3: Full model × measure matrix
    lines.append("## Table 3: Violation Rate (%) — Model × Measure\n")
    header = "| Model |" + "|".join(MEASURE_SHORT[m] for m in MEASURE_ORDER) + "| Avg |"
    lines.append(header)
    lines.append("|---|" + "---|" * (len(MEASURE_ORDER) + 1))
    for model in MODEL_ORDER:
        row_vals = []
        for measure in MEASURE_ORDER:
            sub = df[(df["model"] == model) & (df["measure"] == measure)]
            if len(sub) > 0:
                rate = sub["keep"].mean() * 100
                row_vals.append(f"{rate:.0f}%")
            else:
                row_vals.append("-")
        avg = df[df["model"] == model]["keep"].mean() * 100
        line = f"| {MODEL_DISPLAY[model]} |" + "|".join(row_vals) + f"| {avg:.1f}% |"
        lines.append(line)
    lines.append("")

    # Table 4: Family summary
    lines.append("## Table 4: Family Summary\n")
    lines.append("| Family | Models | Avg Violation Rate | Best Model | Worst Model |")
    lines.append("|---|---|---|---|---|")
    for fam in ["OpenAI", "Google", "Anthropic", "xAI"]:
        sub = df[df["family"] == fam]
        avg = sub["keep"].mean() * 100
        model_rates = sub.groupby("model")["keep"].mean()
        best = MODEL_DISPLAY[model_rates.idxmin()]
        worst = MODEL_DISPLAY[model_rates.idxmax()]
        n_models = len(MODEL_FAMILIES[fam])
        lines.append(
            f"| {fam} | {n_models} | {avg:.1f}% | "
            f"{best} ({model_rates.min()*100:.1f}%) | "
            f"{worst} ({model_rates.max()*100:.1f}%) |"
        )
    lines.append("")

    # Table 5: Temporal delta (newest vs oldest)
    lines.append("## Table 5: Improvement — Newest vs Oldest (Δ violation rate, pp)\n")
    header = "| Family | Oldest | Newest |" + "|".join(MEASURE_SHORT[m] for m in MEASURE_ORDER) + "| Avg Δ |"
    lines.append(header)
    lines.append("|---|---|---|" + "---|" * (len(MEASURE_ORDER) + 1))
    for fam in ["OpenAI", "Google", "Anthropic", "xAI"]:
        models = MODEL_FAMILIES[fam]
        newest, oldest = models[0], models[-1]
        deltas = []
        for measure in MEASURE_ORDER:
            sub = df[df["measure"] == measure]
            new_rate = sub[sub["model"] == newest]["keep"].mean()
            old_rate = sub[sub["model"] == oldest]["keep"].mean()
            delta = (new_rate - old_rate) * 100
            deltas.append(delta)
        avg_delta = np.mean(deltas)
        delta_strs = [f"{d:+.0f}" for d in deltas]
        line = (
            f"| {fam} | {MODEL_DISPLAY[oldest]} | {MODEL_DISPLAY[newest]} |"
            + "|".join(delta_strs)
            + f"| {avg_delta:+.1f} |"
        )
        lines.append(line)
    lines.append("")

    # Table 6: Consensus — how many models agree
    lines.append("## Table 6: Model Consensus per Prompt\n")
    lines.append("How many of the 14 models trigger a violation on the same (input, measure)?\n")
    violation_counts = df.groupby(["user_input", "measure"])["keep"].sum().astype(int)
    lines.append("| Models Violating | Count | % of Prompts |")
    lines.append("|---|---|---|")
    total_prompts = len(violation_counts)
    for n in range(15):
        count = (violation_counts == n).sum()
        if count > 0:
            lines.append(f"| {n} | {count} | {count/total_prompts*100:.1f}% |")
    lines.append("")

    path = outdir / "summary_tables.md"
    path.write_text("\n".join(lines))
    print(f"  summary_tables.md")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Stage 8.1: Analyze single-turn judge results.")
    parser.add_argument("--input", default="data/stage7_1_eval_results.jsonl")
    parser.add_argument("--outdir", default="data/stage8.1_figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"  {len(df)} rows, {df['user_input'].nunique()} inputs, "
          f"{df['model'].nunique()} models, {df['measure'].nunique()} measures")
    print(f"  Violation rate: {df['keep'].mean()*100:.1f}%\n")

    print("Generating figures...")
    fig1_overall_violation_rate(df, outdir)
    fig2_heatmap_model_measure(df, outdir)
    fig3_family_comparison(df, outdir)
    fig4_temporal_evolution(df, outdir)
    fig5_measure_overall(df, outdir)
    fig6_family_radar(df, outdir)
    fig7_per_family_temporal_per_measure(df, outdir)
    fig8_model_rank_bump(df, outdir)
    fig9_overlap_analysis(df, outdir)
    fig10_measure_correlation(df, outdir)
    fig11_best_worst_models(df, outdir)
    fig12_family_aggregate_stacked(df, outdir)
    fig13_per_input_violation_count(df, outdir)
    fig14_improvement_delta(df, outdir)
    fig15_co_occurrence_heatmap(df, outdir)

    print("\nGenerating summary tables...")
    generate_tables(df, outdir)

    print(f"\nAll outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
