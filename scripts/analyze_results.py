"""Analysis and publication-quality visualization suite for RocketMind.

Parses evaluation datasets (both historical reward_hack_vs_baseline.csv and
new rigorous evaluation CSVs), computes statistical aggregates, and produces
high-resolution publication figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path
ROCKETMIND_DIR = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path(__file__).resolve().parent
sys.path.insert(0, str(ROCKETMIND_DIR / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


AGENT_PALETTE = {
    "baseline": "#2b5c8f",          # Deep Blue
    "reward_hack": "#d95f02",       # Red-Orange (Exploitative)
    "progress_shaped": "#2ca02c",   # Emerald Green
    "potential_corrected": "#7570b3"# Purple
}

AGENT_LABELS = {
    "baseline": "Baseline PPO\n(Sparse/Standard)",
    "reward_hack": "Reward-Hacked PPO\n(+1 Hover Bonus)",
    "progress_shaped": "Progress-Shaped PPO\n(PBRS Exp C)",
    "potential_corrected": "Corrected PBRS\n(No Truncation Penalty)",
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load evaluation CSV and standardize column names across historical and new runs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column naming
    if "reward" in df.columns and "raw_reward" not in df.columns:
        df["raw_reward"] = df["reward"]

    if "safe_landing" not in df.columns:
        # Approximate safe landing in historical CSV if missing:
        # score >= 200 and not truncated
        df["safe_landing"] = (df["raw_reward"] >= 200.0) & (~df["truncated"])

    if "crash" not in df.columns:
        # Historical approximation: terminated and raw_reward < 0
        df["crash"] = (~df["truncated"]) & (df["raw_reward"] < 50.0)

    return df


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute aggregate statistical metrics per agent and export summary CSV."""
    records = []
    for agent, group in df.groupby("agent"):
        n = len(group)
        rewards = group["raw_reward"].values
        mean_r = np.mean(rewards)
        std_r = np.std(rewards, ddof=1) if n > 1 else 0.0
        sem_r = std_r / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * sem_r

        records.append({
            "Agent": agent,
            "Episodes": n,
            "Mean Return": round(mean_r, 2),
            "Median Return": round(float(np.median(rewards)), 2),
            "Std Dev": round(std_r, 2),
            "95% CI (±)": round(ci95, 2),
            "Return >= 200 (%)": round(group["score_200_plus"].mean() * 100, 1) if "score_200_plus" in group else 0.0,
            "Safe Landings (%)": round(group["safe_landing"].mean() * 100, 1),
            "Time-Limit Truncations (%)": round(group["truncated"].mean() * 100, 1),
            "Mean Hover Frames": round(group["stable_hover_frames"].mean(), 1),
            "Mean Steps": round(group["steps"].mean(), 1),
        })

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[EXPORT] Summary statistics table exported to: {summary_path}")
    return summary_df


def plot_mean_rewards(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Bar Chart of Mean Raw Rewards with 95% Confidence Intervals."""
    agents = df["agent"].unique()
    means = []
    cis = []
    labels = []
    colors = []

    for ag in agents:
        sub = df[df["agent"] == ag]["raw_reward"]
        n = len(sub)
        m = sub.mean()
        sem = sub.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        means.append(m)
        cis.append(1.96 * sem)
        labels.append(AGENT_LABELS.get(ag, ag))
        colors.append(AGENT_PALETTE.get(ag, "#7f7f7f"))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=cis, capsize=6, color=colors, alpha=0.88, edgecolor="black", width=0.55)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(200, color="green", linestyle=":", linewidth=1.2, label="Conventional Target (Return = 200)")
    ax.set_ylabel("Mean Raw Environment Return")
    ax.set_title("RocketMind: Mean Raw Return Comparison (95% CI Error Bars)", pad=12)
    ax.legend(loc="upper left")

    # Annotate bar values
    for bar, m, ci in zip(bars, means, cis):
        y_pos = m + (ci if m >= 0 else -ci - 15)
        ax.annotate(
            f"{m:.1f}\n(±{ci:.1f})",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 3 if m >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Plot saved: {output_path}")


def plot_reward_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """Generate Box & Strip Plot of Episode Returns demonstrating policy divergence."""
    agents = df["agent"].unique()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    data = [df[df["agent"] == ag]["raw_reward"].values for ag in agents]
    labels = [AGENT_LABELS.get(ag, ag) for ag in agents]
    box_colors = [AGENT_PALETTE.get(ag, "#7f7f7f") for ag in agents]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=1.8),
        flierprops=dict(marker='o', markersize=5, markerfacecolor='gray', alpha=0.6)
    )

    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    # Overlay jittered scatter points for empirical transparency
    for i, ag in enumerate(agents):
        y = df[df["agent"] == ag]["raw_reward"].values
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, "o", color=AGENT_PALETTE.get(ag, "black"), alpha=0.55, markersize=5)

    ax.axhline(200, color="green", linestyle=":", linewidth=1.2, label="Target Return (>=200)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Raw Episode Return")
    ax.set_title("RocketMind: Episode Return Distributions (Boxplot & Data Points)", pad=12)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Plot saved: {output_path}")


def plot_behavior_and_hover(df: pd.DataFrame, output_path: Path) -> None:
    """Generate two-panel behavioral breakdown: Hover Frames and Outcome Rates."""
    agents = df["agent"].unique()
    labels = [AGENT_LABELS.get(ag, ag) for ag in agents]
    colors = [AGENT_PALETTE.get(ag, "#7f7f7f") for ag in agents]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Stable Hover Frames per Episode
    hover_means = [df[df["agent"] == ag]["stable_hover_frames"].mean() for ag in agents]
    hover_stds = [df[df["agent"] == ag]["stable_hover_frames"].std() for ag in agents]

    bars = ax1.bar(labels, hover_means, yerr=hover_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax1.set_ylabel("Average Stable Hover Frames / Episode (Max 1000)")
    ax1.set_title("Evidence of Reward Hacking:\nStable Hovering Exploitation", pad=10)
    for bar, val in zip(bars, hover_means):
        ax1.annotate(f"{val:.1f} frames", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold")

    # Panel 2: Episode Outcome Breakdown (Safe Landings vs Truncations vs Crashes)
    trunc_rates = [df[df["agent"] == ag]["truncated"].mean() * 100 for ag in agents]
    landing_rates = [df[df["agent"] == ag]["safe_landing"].mean() * 100 for ag in agents]
    crash_rates = [max(0.0, 100.0 - t - l) for t, l in zip(trunc_rates, landing_rates)]

    ind = np.arange(len(agents))
    width = 0.5

    p1 = ax2.bar(labels, landing_rates, width, label="Safe Landing (Est.)", color="#2ca02c", alpha=0.85)
    p2 = ax2.bar(labels, trunc_rates, width, bottom=landing_rates, label="Time-Limit Truncation (1000 steps)", color="#d95f02", alpha=0.85)
    p3 = ax2.bar(labels, crash_rates, width, bottom=[i+j for i, j in zip(landing_rates, trunc_rates)], label="Crashes / Incomplete", color="#7f7f7f", alpha=0.7)

    ax2.set_ylabel("Outcome Percentage (%)")
    ax2.set_title("Episode Outcome Classifications\n(Landing vs. Timeout vs. Crash)", pad=10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Plot saved: {output_path}")


def run_full_analysis(csv_file: Path | None = None) -> None:
    results_dir = ROCKETMIND_DIR / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_csv = csv_file or (results_dir / "reward_hack_vs_baseline.csv")
    if not target_csv.exists():
        # Check if alternative exists
        alt = results_dir / "rigorous_agent_comparison.csv"
        if alt.exists():
            target_csv = alt
        else:
            raise FileNotFoundError(f"Cannot find evaluation CSV at {target_csv}")

    print(f"\n=======================================================")
    print(f"[INFO] Loading evaluation data: {target_csv.name}")
    print(f"=======================================================\n")

    df = load_dataset(target_csv)

    # 1. Summary Statistics Table
    summary_df = generate_summary_table(df, results_dir)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    # 2. Generate Plots
    print("\n[PLOT] Generating publication figures...")
    plot_mean_rewards(df, figures_dir / "fig1_mean_reward_comparison.png")
    plot_reward_distributions(df, figures_dir / "fig2_return_distributions.png")
    plot_behavior_and_hover(df, figures_dir / "fig3_behavior_and_outcomes.png")

    print(f"\n[SUCCESS] All figures successfully generated in: {figures_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RocketMind experiment results")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to evaluation CSV file (default: results/reward_hack_vs_baseline.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    run_full_analysis(csv_path)


if __name__ == "__main__":
    main()
