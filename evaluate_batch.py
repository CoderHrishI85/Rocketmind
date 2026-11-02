
import csv
import statistics
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # Our model currently lives in D:\PORJESCTS\models.
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "ppo_lunarlander"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    env = gym.make("LunarLander-v3")
    model = PPO.load(str(model_path), device="cpu")

    rewards = []
    csv_path = results_dir / "baseline_eval.csv"

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["episode", "seed", "reward", "steps", "terminated", "truncated"]
            )

            for episode in range(1, 21):
                seed = 1000 + episode
                obs, info = env.reset(seed=seed)
                total_reward = 0.0
                steps = 0

                while True:
                    action, _ = model.predict(
                        obs, deterministic=True
                    )
                    obs, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break

                rewards.append(total_reward)
                writer.writerow([
                    episode, seed, round(total_reward, 2),
                    steps, terminated, truncated
                ])

                print(
                    f"Episode {episode:02d}/20 | "
                    f"Reward: {total_reward:8.2f} | Steps: {steps}"
                )

    finally:
        env.close()

    print("\n--- BASELINE SUMMARY ---")
    print(f"Mean reward: {statistics.mean(rewards):.2f}")
    print(f"Median reward: {statistics.median(rewards):.2f}")
    print(f"Std deviation: {statistics.pstdev(rewards):.2f}")
    print(f"Scores >= 200: {sum(r >= 200 for r in rewards)}/20")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
