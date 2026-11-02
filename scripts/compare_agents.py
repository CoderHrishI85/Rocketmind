"""Reproducible comparative evaluation across RocketMind checkpoints.
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

ROCKETMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROCKETMIND_DIR / "src"))

import pandas as pd

from rocketmind.evaluate import evaluate_agent, save_results_to_csv
from rocketmind.utils import get_model_path, get_results_dir


DEFAULT_MODELS = {
    "baseline": "ppo_lunarlander.zip",
    "reward_hack": "ppo_lunarlander_reward_hack.zip",
    "progress_shaped": "ppo_lunarlander_progress.zip",
}


def run_comparison(
    models_dict: dict[str, str],
    n_episodes: int = 20,
    seed_start: int = 1001,
    output_filename: str = "rigorous_agent_comparison.csv",
) -> pd.DataFrame:
    results_dir = get_results_dir()
    all_episodes = []
    summaries = []

    print("\n" + "=" * 65)
    print(f"[EVAL] Running RocketMind Agent Comparison ({n_episodes} episodes per agent)")
    print(f"[SEEDS] Seeds: {seed_start} to {seed_start + n_episodes - 1}")
    print("=" * 65 + "\n")

    for agent_name, model_file in models_dict.items():
        model_path = get_model_path(model_file)
        if not model_path.exists():
            print(f"[WARN] Checkpoint for '{agent_name}' not found at {model_path}. Skipping.")
            continue

        print(f"[RUN] Evaluating: {agent_name} [{model_path.name}]...")
        ep_results, summary = evaluate_agent(
            model=model_path,
            agent_name=agent_name,
            n_episodes=n_episodes,
            seed_start=seed_start,
            deterministic=True,
        )

        all_episodes.extend(ep_results)
        summaries.append(summary)

        print(
            f"   Mean: {summary['mean_reward']:6.2f} ± {summary['ci95_high']-summary['mean_reward']:.2f} | "
            f"Median: {summary['median_reward']:6.2f} | "
            f"Safe Landings: {summary['safe_landing_rate']*100:4.1f}% | "
            f"Truncations: {summary['truncation_rate']*100:4.1f}% | "
            f"Hover Frames: {summary['mean_hover_frames']:5.1f}"
        )

    # Save per-episode CSV
    csv_path = results_dir / output_filename
    save_results_to_csv(all_episodes, csv_path)
    print(f"\n[SAVE] Per-episode evaluation data saved to: {csv_path}")

    # Display comparison summary table
    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 80)
    print("[SUMMARY] COMPARATIVE SUMMARY TABLE (Raw Environment Rewards)")
    print("=" * 80)
    if not summary_df.empty:
        cols_to_show = [
            "agent",
            "episodes",
            "mean_reward",
            "median_reward",
            "std_reward",
            "safe_landing_rate",
            "truncation_rate",
            "score_200_rate",
            "mean_hover_frames",
        ]
        available_cols = [c for c in cols_to_show if c in summary_df.columns]
        print(summary_df[available_cols].to_string(index=False))

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RocketMind RL agents")
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per agent (default: 20)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1001,
        help="Starting evaluation seed (default: 1001)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rigorous_agent_comparison.csv",
        help="CSV filename to save per-episode data",
    )
    args = parser.parse_args()

    run_comparison(
        models_dict=DEFAULT_MODELS,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()
