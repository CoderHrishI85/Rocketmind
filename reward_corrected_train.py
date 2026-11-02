
from math import exp
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
GAMMA = 0.99
SHAPING_SCALE = 10.0


class PotentialRewardWrapper(gym.Wrapper):
    """Add potential-based shaping; retain the original environment reward."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_potential = 0.0

    @staticmethod
    def potential(obs):
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = obs

        # A larger potential means a state closer to our target:
        # centered, low above the pad, slow, and nearly upright.
        cost = (
            1.2 * abs(float(x))
            + 0.6 * max(float(y), 0.0)
            + 0.4 * abs(float(vx))
            + 0.4 * abs(float(vy))
            + 0.5 * abs(float(angle))
            + 0.2 * abs(float(angular_velocity))
        )

        return exp(-cost)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Terminal potential is zero so shaping stops at episode end.
        if terminated or truncated:
            next_potential = 0.0
        else:
            next_potential = self.potential(obs)

        shaping_reward = SHAPING_SCALE * (
            GAMMA * next_potential - self.previous_potential
        )

        self.previous_potential = next_potential

        info["original_reward"] = float(reward)
        info["shaping_reward"] = float(shaping_reward)

        return (
            obs,
            float(reward) + shaping_reward,
            terminated,
            truncated,
            info,
        )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = PotentialRewardWrapper(gym.make("LunarLander-v3"))

    print("🚀 Training corrected reward-shaping agent...")
    print("Original reward retained; potential-based shaping added.")

    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            seed=43,
            gamma=GAMMA,
            tensorboard_log=str(PROJECT_DIR / "runs"),
        )

        model.learn(
            total_timesteps=300_000,
            tb_log_name="PPO_PotentialShaping",
        )

        save_path = MODEL_DIR / "ppo_lunarlander_progress"
        model.save(str(save_path))

        print("\n✅ Corrected shaping model trained!")
        print(f"Saved to: {save_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
