
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    print("Loading trained RocketMind agent...")

    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load("models/ppo_lunarlander", device="cpu")

    rewards = []

    try:
        for episode in range(1, 6):
            observation, info = env.reset(seed=100 + episode)
            total_reward = 0.0
            steps = 0

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    rewards.append(total_reward)

                    print(
                        f"Episode {episode}/5 | "
                        f"Steps: {steps} | "
                        f"Reward: {total_reward:.2f}"
                    )
                    break

        average = sum(rewards) / len(rewards)
        print(f"\nAverage reward across 5 episodes: {average:.2f}")
        print(f"Episodes with reward >= 200: "
              f"{sum(r >= 200 for r in rewards)}/5")

    finally:
        env.close()


if __name__ == "__main__":
    main()

