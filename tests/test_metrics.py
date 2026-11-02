"""Unit tests for metric tracking and episode outcome classification.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rocketmind.metrics import EpisodeMetricsTracker


class TestMetrics(unittest.TestCase):
    def test_safe_landing_classification(self):
        tracker = EpisodeMetricsTracker(agent_name="test", episode=1, seed=42)
        # Safe touchdown observation: grounded, slow, upright, centered
        landing_obs = np.array([0.02, 0.0, 0.01, -0.05, 0.01, 0.0, 1.0, 1.0])

        tracker.update(
            action=0,
            obs=landing_obs,
            reward=120.0,
            terminated=True,
            truncated=False,
            info={"original_reward": 120.0},
        )
        res = tracker.finalize()

        self.assertTrue(res.safe_landing)
        self.assertFalse(res.crash)
        self.assertFalse(res.persistent_hover)
        self.assertFalse(res.time_limit_reached)

    def test_crash_classification(self):
        tracker = EpisodeMetricsTracker(agent_name="test", episode=2, seed=43)
        # Crash observation: high velocity, tilted, terminated
        crash_obs = np.array([0.5, 0.1, 1.2, -1.8, 0.8, 1.5, 0.0, 0.0])

        tracker.update(
            action=2,
            obs=crash_obs,
            reward=-100.0,
            terminated=True,
            truncated=False,
            info={"original_reward": -100.0},
        )
        res = tracker.finalize()

        self.assertFalse(res.safe_landing)
        self.assertTrue(res.crash)

    def test_persistent_hover_classification(self):
        tracker = EpisodeMetricsTracker(agent_name="test", episode=3, seed=44)
        hover_obs = np.array([0.0, 0.8, 0.02, -0.01, 0.01, 0.0, 0.0, 0.0])

        # Simulate 800 steps of hovering
        for _ in range(800):
            tracker.update(
                action=2,
                obs=hover_obs,
                reward=-0.1,
                terminated=False,
                truncated=False,
                info={"original_reward": -0.1},
            )

        # Truncate at step 801
        tracker.update(
            action=0,
            obs=hover_obs,
            reward=-0.1,
            terminated=False,
            truncated=True,
            info={"original_reward": -0.1},
        )
        res = tracker.finalize()

        self.assertTrue(res.persistent_hover)
        self.assertTrue(res.time_limit_reached)
        self.assertFalse(res.safe_landing)


if __name__ == "__main__":
    unittest.main()

