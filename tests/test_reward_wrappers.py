"""Unit tests for reward wrappers and potential function mathematics.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

import gymnasium as gym
from gymnasium import spaces

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rocketmind.reward_wrappers import (
    CorrectedPotentialRewardWrapper,
    HistoricalPotentialRewardWrapper,
    HoverRewardWrapper,
)


class DummyEnv(gym.Env):
    """Mock environment for testing wrappers in isolation."""

    def __init__(self, initial_obs: np.ndarray):
        super().__init__()
        self.obs = initial_obs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=initial_obs.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        return self.obs, {}

    def step(self, action):
        return self.obs, 10.0, False, False, {}


class TestRewardWrappers(unittest.TestCase):
    def test_hover_reward_wrapper_awards_bonus(self):
        # Stable hover state: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
        hovering_obs = np.array([0.0, 0.5, 0.05, -0.05, 0.02, 0.01, 0.0, 0.0])
        wrapper = HoverRewardWrapper(DummyEnv(hovering_obs), bonus_amount=1.0)
        wrapper.reset()
        obs, reward, term, trunc, info = wrapper.step(0)

        self.assertTrue(info["is_stable_hover"])
        self.assertEqual(info["hover_bonus"], 1.0)
        self.assertEqual(reward, 11.0)  # 10 raw + 1 bonus

    def test_hover_reward_wrapper_denies_bonus_on_fast_movement(self):
        fast_obs = np.array([0.0, 0.5, 0.8, -0.9, 0.4, 0.5, 0.0, 0.0])
        wrapper = HoverRewardWrapper(DummyEnv(fast_obs), bonus_amount=1.0)
        wrapper.reset()
        obs, reward, term, trunc, info = wrapper.step(0)

        self.assertFalse(info["is_stable_hover"])
        self.assertEqual(info["hover_bonus"], 0.0)
        self.assertEqual(reward, 10.0)  # No bonus

    def test_historical_potential_calculation(self):
        origin_obs = np.zeros(8)
        phi_zero = HistoricalPotentialRewardWrapper.potential(origin_obs)
        # At origin cost=0 -> exp(0)=1.0
        self.assertAlmostEqual(phi_zero, 1.0)

    def test_corrected_potential_contact_bonus(self):
        grounded_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        airborne_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        phi_grounded = CorrectedPotentialRewardWrapper.potential(grounded_obs)
        phi_airborne = CorrectedPotentialRewardWrapper.potential(airborne_obs)

        self.assertGreater(phi_grounded, phi_airborne)


if __name__ == "__main__":
    unittest.main()
