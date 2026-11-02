
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = PROJECT_DIR / "runs"


class HoverRewardWrapper(gym.Wrapper):
    """Intentionally flawed reward: reward hovering in a stable state."""

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # LunarLander observation:
        # [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
        vx = obs[2]
        vy = obs[3]
        angle = obs[4]
        angular_velocity = obs[5]
        left_leg = obs[6]
        right_leg = obs[7]

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

        bonus = 1.0 if stable_hover else 0.0

        # Deliberately flawed: stable hovering earns extra points.
        modified_reward = float(reward) + bonus

        info["hover_bonus"] = bonus

        return (
            obs,
            modified_reward,
            terminated,
            truncated,
            info,
        )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base_env = gym.make("LunarLander-v3")
    env = HoverRewardWrapper(base_env)

    print("🚀 Starting reward-hacking experiment...")
    print("Extra reward: +1 per stable, airborne frame.")
    print("Baseline model will NOT be overwritten.")

    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            seed=42,
            tensorboard_log=str(LOG_DIR),
        )

        model.learn(
            total_timesteps=300_000,
            tb_log_name="PPO_RewardHacking",
        )

        output_path = MODEL_DIR / "ppo_lunarlander_reward_hack"
        model.save(str(output_path))

        print("\n✅ Reward-hacking training complete!")
        print(f"Model saved to: {output_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
