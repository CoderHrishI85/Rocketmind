"""Educational Proximal Policy Optimization (PPO) in Pure PyTorch.

This module provides a clean, modular, and mathematically documented implementation
of the PPO-Clip algorithm (Schulman et al., 2017) with Generalized Advantage Estimation (GAE).
Designed for educational clarity and ML interview explanations.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    """Two-headed Actor-Critic Network.
    
    Actor Head: Produces unnormalized logits over discrete actions.
    Critic Head: Predicts state value V(s).
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()

        # Actor Network (Policy pi(a|s))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic Network (Value Function V(s))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action sampling, log probability, entropy, and state value."""
        logits = self.actor(state)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Predict state value V(s) only."""
        return self.critic(state).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE-lambda).
    
    Formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
        Returns_t = A_t + V(s_t)
    """
    n_steps = len(rewards)
    advantages = torch.zeros(n_steps, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class CustomPPOTrainer:
    """Modular trainer for the PPO-Clip algorithm."""

    def __init__(
        self,
        env_id: str = "LunarLander-v3",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.env = gym.make(env_id)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=64,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def train_step(
        self,
        rollout_steps: int = 512,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Collect a trajectory buffer and perform PPO minibatch gradient updates."""
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(rollout_steps):
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = self.model.get_action_and_value(state_t)

            next_obs, reward, terminated, truncated, _ = self.env.step(action_t.item())
            done = terminated or truncated

            states.append(obs)
            actions.append(action_t.item())
            log_probs.append(log_prob_t.item())
            rewards.append(reward)
            dones.append(float(done))
            values.append(value_t.item())

            obs = next_obs if not done else self.env.reset()[0]

        # Convert rollout buffer to tensors
        b_states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        b_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        b_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        b_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        b_values = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            last_state = torch.tensor(obs, dtype=torch.float32, device=self.device)
            next_value = self.model.get_value(last_state)

        # Compute GAE advantages and target returns
        advantages, returns = compute_gae(
            b_rewards, b_values, b_dones, next_value, self.gamma, self.gae_lambda
        )
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Minibatch PPO Optimization
        n_samples = rollout_steps
        indices = np.arange(n_samples)
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                    b_states[mb_idx], b_actions[mb_idx]
                )

                # Ratio: r(theta) = exp(log_prob_new - log_prob_old)
                ratio = torch.exp(new_log_prob - b_log_probs[mb_idx])

                # Clipped Surrogate Objective: min(r*A, clip(r, 1-eps, 1+eps)*A)
                mb_adv = advantages[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss (MSE)
                value_loss = 0.5 * ((new_value - returns[mb_idx]) ** 2).mean()

                # Entropy Bonus for Exploration
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / (n_epochs * (n_samples // batch_size)),
            "value_loss": total_value_loss / (n_epochs * (n_samples // batch_size)),
            "mean_reward": float(np.mean(rewards)),
        }

