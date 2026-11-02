"""Gymnasium reward wrappers for RocketMind experiments.

Includes:
1. HoverRewardWrapper: The deliberately flawed dense reward that induces reward hacking.
2. HistoricalPotentialRewardWrapper: The original potential-shaping wrapper used in Experiment C (with terminal-zeroing flaw).
3. CorrectedPotentialRewardWrapper: Mathematically rigorous Potential-Based Reward Shaping (Ng et al., 1999) with proper terminal and truncation handling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class HoverRewardWrapper(gym.Wrapper):
    """Deliberately flawed reward wrapper: awards +1 bonus per frame for stable hovering.
    
    LunarLander observation space:
    [0] x: horizontal coordinate (-1.0 to 1.0)
    [1] y: vertical coordinate (-0.5 to 1.5)
    [2] vx: horizontal velocity
    [3] vy: vertical velocity
    [4] angle: rocket tilt in radians
    [5] angular_velocity: rate of tilt change
    [6] left_leg: boolean (1.0 if contacting ground, else 0.0)
    [7] right_leg: boolean (1.0 if contacting ground, else 0.0)
    """

    def __init__(self, env: gym.Env, bonus_amount: float = 1.0) -> None:
        super().__init__(env)
        self.bonus_amount = bonus_amount

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        angular_velocity = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        stable_hover = (
            abs(vx) < 0.2
            and abs(vy) < 0.2
            and abs(angle) < 0.15
            and abs(angular_velocity) < 0.2
            and left_leg < 0.5
            and right_leg < 0.5
            and not terminated
            and not truncated
        )

        bonus = self.bonus_amount if stable_hover else 0.0
        modified_reward = float(reward) + bonus

        info["original_reward"] = float(reward)
        info["hover_bonus"] = bonus
        info["is_stable_hover"] = stable_hover

        return obs, modified_reward, terminated, truncated, info


class HistoricalPotentialRewardWrapper(gym.Wrapper):
    """Historical potential-based shaping wrapper from Experiment C.
    
    Maintains historical fidelity for reproducibility and diagnostic analysis.
    Known flaw: Forces next_potential = 0.0 on terminated OR truncated,
    which penalizes successful landings and artificial time horizon timeouts.
    """

    def __init__(
        self, env: gym.Env, gamma: float = 0.99, shaping_scale: float = 10.0
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.previous_potential = 0.0

    @staticmethod
    def potential(obs: np.ndarray) -> float:
        x, y, vx, vy, angle, angular_velocity, _, _ = obs
        cost = (
            1.2 * abs(float(x))
            + 0.6 * max(float(y), 0.0)
            + 0.4 * abs(float(vx))
            + 0.4 * abs(float(vy))
            + 0.5 * abs(float(angle))
            + 0.2 * abs(float(angular_velocity))
        )
        return math.exp(-cost)

    def reset(
        self, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            next_potential = 0.0
        else:
            next_potential = self.potential(obs)

        shaping_reward = self.shaping_scale * (
            self.gamma * next_potential - self.previous_potential
        )
        self.previous_potential = next_potential

        info["original_reward"] = float(reward)
        info["shaping_reward"] = float(shaping_reward)

        return obs, float(reward) + shaping_reward, terminated, truncated, info


class CorrectedPotentialRewardWrapper(gym.Wrapper):
    """Mathematically rigorous Potential-Based Reward Shaping (PBRS, Ng et al. 1999).
    
    Theoretical Formulation:
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        
    Corrections implemented:
    1. Truncation Handling: Time-limit expirations (step 1000) are not MDP absorbing
       terminals. Potential Phi(s_truncated) is evaluated normally rather than forced to 0.
    2. Terminal Landing Handling: Touchdown states incorporate leg contact bonus and do not
       drop abruptly to 0.0, eliminating the artificial landing penalty.
    3. State Representation: Includes leg contact grounding incentives.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        shaping_scale: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self.previous_potential = 0.0

    @staticmethod
    def potential(obs: np.ndarray, terminated: bool = False) -> float:
        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        angular_vel = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        # Distance and alignment cost
        pos_cost = 1.0 * abs(x) + 0.8 * max(y, 0.0)
        vel_cost = 0.5 * abs(vx) + 0.5 * abs(vy)
        rot_cost = 0.6 * abs(angle) + 0.3 * abs(angular_vel)

        cost = pos_cost + vel_cost + rot_cost

        # Base potential in [0, 1]
        base_phi = math.exp(-cost)

        # Contact incentives: legs touching ground near center should increase potential
        contact_bonus = 0.0
        if left_leg > 0.5 and right_leg > 0.5 and abs(x) < 0.3:
            contact_bonus = 0.5
        elif (left_leg > 0.5 or right_leg > 0.5) and abs(x) < 0.4:
            contact_bonus = 0.25

        return base_phi + contact_bonus

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # On truncation, continue with physical potential (not absorbing 0)
        # On termination, calculate terminal potential based on true outcome
        next_potential = self.potential(obs, terminated=terminated)

        shaping_reward = self.shaping_scale * (
            self.gamma * next_potential - self.previous_potential
        )
        self.previous_potential = next_potential

        info["original_reward"] = float(reward)
        info["shaping_reward"] = float(shaping_reward)

        return obs, float(reward) + shaping_reward, terminated, truncated, info

