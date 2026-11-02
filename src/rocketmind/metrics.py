"""Scientific metrics and evaluation analyzers for RocketMind.

Distinguishes between:
1. Episode Return Threshold (return >= 200) - historical benchmark metric.
2. Safe Landing - verified physical touchdown with both legs, low impact velocity, and upright angle.
3. Crash - physical terminal impact resulting in destruction.
4. Persistent Hovering - prolonged airborne stability exploiting the time limit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    agent: str
    episode: int
    seed: int
    raw_reward: float
    steps: int
    terminated: bool
    truncated: bool
    score_200_plus: bool
    safe_landing: bool
    crash: bool
    time_limit_reached: bool
    stable_hover_frames: int
    persistent_hover: bool
    fuel_burn_steps: int
    final_x: float
    final_y: float
    final_vx: float
    final_vy: float
    final_angle: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodeMetricsTracker:
    """Tracks per-step transitions and computes rigorous scientific outcome metrics."""

    def __init__(self, agent_name: str, episode: int, seed: int) -> None:
        self.agent_name = agent_name
        self.episode = episode
        self.seed = seed

        self.steps = 0
        self.total_raw_reward = 0.0
        self.stable_hover_frames = 0
        self.fuel_burn_steps = 0

        self.last_obs: Optional[np.ndarray] = None
        self.last_terminated = False
        self.last_truncated = False

    def update(
        self,
        action: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        self.steps += 1
        raw_r = info.get("original_reward", reward)
        self.total_raw_reward += float(raw_r)

        # Track fuel expenditure (actions 1=left, 2=main, 3=right engine)
        if action != 0:
            self.fuel_burn_steps += 1

        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        angular_vel = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        # Instantaneous proxy condition for stable airborne flight
        is_stable_airborne = (
            abs(vx) < 0.2
            and abs(vy) < 0.2
            and abs(angle) < 0.15
            and abs(angular_vel) < 0.2
            and left_leg < 0.5
            and right_leg < 0.5
            and not terminated
            and not truncated
        )
        if is_stable_airborne:
            self.stable_hover_frames += 1

        self.last_obs = obs
        self.last_terminated = terminated
        self.last_truncated = truncated

    def finalize(self) -> EpisodeResult:
        if self.last_obs is None:
            raise RuntimeError("Tracker has not recorded any transitions.")

        obs = self.last_obs
        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        impact_speed = math.sqrt(vx * vx + vy * vy)
        both_legs_down = (left_leg > 0.5) and (right_leg > 0.5)
        is_upright = abs(angle) < 0.20
        centered = abs(x) < 0.40

        # Rigorous ground truth landing classification:
        # Must terminate naturally (not timeout), touch down on both legs upright,
        # with low speed and a net positive completion reward.
        is_safe_landing = bool(
            self.last_terminated
            and not self.last_truncated
            and both_legs_down
            and is_upright
            and centered
            and impact_speed < 0.35
            and self.total_raw_reward >= 100.0
        )

        # Crash classification: terminated with substantial negative impact or failed contact
        is_crash = bool(
            self.last_terminated
            and not self.last_truncated
            and not is_safe_landing
        )

        time_limit_hit = bool(self.last_truncated)

        # Persistent hovering: lander timed out or survived long (>= 750 steps)
        # while keeping stable airborne state for >= 60% of its duration
        is_persistent_hover = bool(
            self.steps >= 750
            and (self.stable_hover_frames / max(self.steps, 1)) >= 0.60
            and not is_safe_landing
        )

        return EpisodeResult(
            agent=self.agent_name,
            episode=self.episode,
            seed=self.seed,
            raw_reward=round(self.total_raw_reward, 3),
            steps=self.steps,
            terminated=self.last_terminated,
            truncated=self.last_truncated,
            score_200_plus=(self.total_raw_reward >= 200.0),
            safe_landing=is_safe_landing,
            crash=is_crash,
            time_limit_reached=time_limit_hit,
            stable_hover_frames=self.stable_hover_frames,
            persistent_hover=is_persistent_hover,
            fuel_burn_steps=self.fuel_burn_steps,
            final_x=round(x, 4),
            final_y=round(y, 4),
            final_vx=round(vx, 4),
            final_vy=round(vy, 4),
            final_angle=round(angle, 4),
        )

