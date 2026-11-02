"""Unit tests for LunarLander environment creation and spaces.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rocketmind.environments import make_env


class TestEnvironment(unittest.TestCase):
    def test_baseline_env_spaces(self):
        env = make_env("LunarLander-v3", wrapper_type="none")
        self.assertEqual(env.action_space.n, 4)
        self.assertEqual(env.observation_space.shape, (8,))
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 8)
        env.close()

    def test_step_execution(self):
        env = make_env("LunarLander-v3", wrapper_type="none")
        obs, info = env.reset(seed=123)
        next_obs, reward, terminated, truncated, step_info = env.step(0)
        self.assertEqual(len(next_obs), 8)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        env.close()


if __name__ == "__main__":
    unittest.main()

