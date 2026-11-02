"""Evaluation engine for RocketMind agents.

Runs reproducible deterministic (or stochastic) evaluations in the raw, unmodified
LunarLander-v3 environment across specified seed lists.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rocketmind.metrics import EpisodeMetricsTracker, EpisodeResult
from rocketmind.utils import get_model_path, get_results_dir


def evaluate_agent(
    model: Union[str, Path, PPO],
    agent_name: str,
    n_episodes: int = 20,
    seed_start: int = 1001,
    deterministic: bool = True,
    device: str = "cpu",
    render_mode: Optional[str] = None,
) -> Tuple[List[EpisodeResult], Dict[str, Any]]:
    """Evaluate an RL agent over a sequence of seeded episodes.
    
    Args:
        model: Loaded SB3 PPO model or path / filename to checkpoint.
        agent_name: Descriptive name for logging.
        n_episodes: Number of episodes to evaluate.
        seed_start: Initial seed (episodes use seed_start, seed_start+1, ...).
        deterministic: If True, greedy deterministic action selection is used.
        device: Device to load model onto ('cpu' or 'cuda').
        render_mode: None, 'human', or 'rgb_array'.
        
    Returns:
        (episode_results, summary_statistics)
    """
    if isinstance(model, (str, Path)):
        resolved_path = get_model_path(str(model))
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")
        loaded_model = PPO.load(str(resolved_path), device=device)
    else:
        loaded_model = model

    # Always evaluate on the raw environment without reward wrappers
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    results: List[EpisodeResult] = []

    try:
        for ep_idx in range(1, n_episodes + 1):
            seed = seed_start + ep_idx - 1
            obs, info = env.reset(seed=seed)
            tracker = EpisodeMetricsTracker(
                agent_name=agent_name, episode=ep_idx, seed=seed
            )

            while True:
                action, _ = loaded_model.predict(obs, deterministic=deterministic)
                # action can be a numpy scalar or int
                act_int = int(action)
                obs, reward, terminated, truncated, step_info = env.step(act_int)

                tracker.update(
                    action=act_int,
                    obs=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=step_info,
                )

                if terminated or truncated:
                    break

            results.append(tracker.finalize())
    finally:
        env.close()

    summary = compute_summary_statistics(results)
    return results, summary


def compute_summary_statistics(results: List[EpisodeResult]) -> Dict[str, Any]:
    """Calculate summary statistics including mean, median, std, and 95% confidence intervals."""
    if not results:
        return {}

    rewards = [r.raw_reward for r in results]
    steps = [r.steps for r in results]
    hovers = [r.stable_hover_frames for r in results]
    n = len(results)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards, ddof=1)) if n > 1 else 0.0
    sem_r = std_r / math.sqrt(n) if n > 1 else 0.0
    ci95_r = 1.96 * sem_r

    return {
        "agent": results[0].agent,
        "episodes": n,
        "mean_reward": round(mean_r, 2),
        "median_reward": round(float(np.median(rewards)), 2),
        "std_reward": round(std_r, 2),
        "sem_reward": round(sem_r, 2),
        "ci95_low": round(mean_r - ci95_r, 2),
        "ci95_high": round(mean_r + ci95_r, 2),
        "mean_steps": round(float(np.mean(steps)), 1),
        "mean_hover_frames": round(float(np.mean(hovers)), 1),
        "score_200_rate": round(sum(r.score_200_plus for r in results) / n, 3),
        "safe_landing_rate": round(sum(r.safe_landing for r in results) / n, 3),
        "crash_rate": round(sum(r.crash for r in results) / n, 3),
        "truncation_rate": round(sum(r.time_limit_reached for r in results) / n, 3),
        "persistent_hover_rate": round(sum(r.persistent_hover for r in results) / n, 3),
    }


def save_results_to_csv(results: List[EpisodeResult], filepath: Path) -> None:
    """Save a list of EpisodeResult objects to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

