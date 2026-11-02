"""Unit tests for educational PyTorch custom PPO implementation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rocketmind.custom_ppo import ActorCritic, compute_gae


class TestCustomPPO(unittest.TestCase):
    def test_actor_critic_shapes(self):
        model = ActorCritic(state_dim=8, action_dim=4, hidden_dim=32)
        state = torch.randn(5, 8)
        action, log_prob, entropy, value = model.get_action_and_value(state)

        self.assertEqual(action.shape, (5,))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(entropy.shape, (5,))
        self.assertEqual(value.shape, (5,))

    def test_gae_computation(self):
        rewards = torch.tensor([1.0, 1.0, 0.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        next_value = torch.tensor(0.0)

        advantages, returns = compute_gae(rewards, values, dones, next_value)
        self.assertEqual(len(advantages), 3)
        self.assertEqual(len(returns), 3)


if __name__ == "__main__":
    unittest.main()

